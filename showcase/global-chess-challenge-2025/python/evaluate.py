"""
Local Evaluation for Global Chess Challenge 2025

Evaluates a chess-playing LLM using Average Centipawn Loss (ACPL).
Uses Stockfish for ground truth analysis.

Fixed issues (thanks GPT 5.2):
1. Compare best-after vs model-after (not root eval vs model-after)
2. Use real game positions, not random walks
3. Strict parsing - require <uci_move> tags
4. Track legal_rate as gating metric
5. Record evaluation settings
"""
import chess
import chess.engine
import chess.pgn
import json
import random
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from datetime import datetime

# Stockfish path - update for your system or set STOCKFISH_PATH env var
import os
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")


@dataclass
class EvalResult:
    """Result of evaluating a single position"""
    fen: str
    best_move: str
    model_move: str
    best_cp: int  # Centipawn score AFTER best move
    model_cp: int  # Centipawn score AFTER model's move
    cp_loss: int  # Difference (lower is better)
    is_legal: bool
    is_best: bool


@dataclass
class EvalConfig:
    """Evaluation configuration for reproducibility"""
    stockfish_path: str
    stockfish_version: str
    depth: int
    n_positions: int
    position_source: str
    seed: int
    timestamp: str


def get_stockfish_version(path: str) -> str:
    """Get Stockfish version string"""
    try:
        result = subprocess.run([path], input="quit\n", capture_output=True, text=True, timeout=5)
        first_line = result.stdout.split('\n')[0]
        return first_line
    except Exception as e:
        return f"unknown ({e})"


def score_to_cp(score: chess.engine.PovScore) -> int:
    """Convert PovScore to centipawns from white's perspective"""
    s = score.white()
    if s.is_mate():
        return 10000 if s.mate() > 0 else -10000
    return s.score(mate_score=10000)


def parse_uci_move(response: str, strict: bool = True) -> Optional[str]:
    """
    Extract UCI move from model response.

    Args:
        response: Model output text
        strict: If True, require XML-style tags (recommended for eval)
    """
    # Look for <uci_move> tags (standard format)
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        return match.group(1)

    # Also accept <uci> tags (model sometimes outputs this)
    match = re.search(r'<uci>\s*([a-h][1-8][a-h][1-8][qrbn]?)', response)
    if match:
        return match.group(1)

    if strict:
        return None  # Don't fall back in strict mode

    # Fallback: look for any UCI-like move (only if not strict)
    match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
    if match:
        return match.group(1)

    return None


def evaluate_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    model_move_uci: str,
    depth: int = 20
) -> EvalResult:
    """
    Evaluate model's move against Stockfish analysis.

    FIXED: Now compares evaluation AFTER best move vs AFTER model move,
    not root position vs after model move.
    """
    fen = board.fen()
    mover_is_white = (board.turn == chess.WHITE)

    # Get best move from this position
    root = engine.analyse(board, chess.engine.Limit(depth=depth))
    best_move = root["pv"][0]

    # Score AFTER best move
    board.push(best_move)
    best_after = engine.analyse(board, chess.engine.Limit(depth=depth))
    best_after_cp = score_to_cp(best_after["score"])
    board.pop()

    # Parse/check legality
    try:
        model_move = chess.Move.from_uci(model_move_uci)
        is_legal = model_move in board.legal_moves
    except ValueError:
        is_legal = False
        model_move = None

    if not is_legal:
        return EvalResult(
            fen=fen,
            best_move=best_move.uci(),
            model_move=model_move_uci,
            best_cp=best_after_cp,
            model_cp=-10000 if mover_is_white else 10000,
            cp_loss=10000,
            is_legal=False,
            is_best=False,
        )

    # Score AFTER model move
    board.push(model_move)
    model_after = engine.analyse(board, chess.engine.Limit(depth=depth))
    model_after_cp = score_to_cp(model_after["score"])
    board.pop()

    # Loss from mover's perspective (higher is worse)
    if mover_is_white:
        cp_loss = best_after_cp - model_after_cp
    else:
        cp_loss = model_after_cp - best_after_cp

    cp_loss = max(0, cp_loss)

    return EvalResult(
        fen=fen,
        best_move=best_move.uci(),
        model_move=model_move_uci,
        best_cp=best_after_cp,
        model_cp=model_after_cp,
        cp_loss=cp_loss,
        is_legal=True,
        is_best=(model_move == best_move),
    )


def load_positions_from_pgn(
    pgn_path: str,
    n: int = 100,
    min_move: int = 10,
    max_move: int = 60,
    seed: int = 42
) -> List[chess.Board]:
    """
    Load positions from real games (PGN file).

    Much better than random walks - these are realistic positions.
    """
    random.seed(seed)
    positions = []
    all_positions = []

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            move_num = 0
            for move in game.mainline_moves():
                move_num += 1
                board.push(move)

                if min_move <= move_num <= max_move:
                    if not board.is_game_over():
                        all_positions.append(board.copy())

            # Stop if we have enough candidates
            if len(all_positions) >= n * 10:
                break

    # Sample from collected positions
    if len(all_positions) >= n:
        positions = random.sample(all_positions, n)
    else:
        positions = all_positions

    return positions


def load_positions_from_training_data(
    jsonl_path: str,
    n: int = 100,
    seed: int = 42
) -> List[chess.Board]:
    """
    Load positions from training JSONL file.

    Uses the same positions we trained on (or a subset).
    Supports both chat format (messages) and simple format (prompt).
    """
    random.seed(seed)
    all_positions = []

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)

            # Try to extract FEN from different formats
            text_to_search = ""

            # Chat format: messages array
            if "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "user":
                        text_to_search = msg.get("content", "")
                        break
            # Simple format: prompt field
            elif "prompt" in data:
                text_to_search = data["prompt"]

            # Extract FEN
            fen_match = re.search(r'FEN:\s*([^\n]+)', text_to_search)
            if fen_match:
                fen = fen_match.group(1).strip()
                try:
                    board = chess.Board(fen)
                    if not board.is_game_over():
                        all_positions.append(board)
                except:
                    pass

    # Sample
    if len(all_positions) >= n:
        return random.sample(all_positions, n)
    return all_positions


def generate_test_positions_random(n: int = 100, seed: int = 42) -> List[chess.Board]:
    """
    Generate positions via random walks.

    WARNING: These are unrealistic positions. Use load_positions_from_pgn()
    or load_positions_from_training_data() for better evaluation.
    """
    random.seed(seed)
    positions = []

    while len(positions) < n:
        board = chess.Board()

        # Play random moves to reach a position
        num_moves = random.randint(10, 60)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))

        # Skip terminal positions
        if board.is_game_over():
            continue

        positions.append(board.copy())

    return positions


class ChessModelEvaluator:
    """Evaluates a chess-playing model"""

    def __init__(self, model_fn: Callable, stockfish_path: str = STOCKFISH_PATH):
        """
        Args:
            model_fn: Function that takes (fen, legal_moves) and returns response string
            stockfish_path: Path to Stockfish binary
        """
        self.model_fn = model_fn
        self.stockfish_path = stockfish_path
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.stockfish_version = get_stockfish_version(stockfish_path)

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()

    def evaluate(
        self,
        positions: List[chess.Board],
        depth: int = 20,
        strict_parsing: bool = True,
        position_source: str = "unknown"
    ) -> dict:
        """
        Evaluate model on a set of positions.

        Args:
            positions: List of chess positions to evaluate
            depth: Stockfish analysis depth
            strict_parsing: If True, require <uci_move> tags
            position_source: Description of where positions came from

        Returns:
            {
                "acpl": float,  # Average centipawn loss
                "legal_rate": float,  # Fraction of legal moves
                "best_move_rate": float,  # Fraction matching Stockfish
                "n_positions": int,
                "config": EvalConfig,
                "results": List[EvalResult]
            }
        """
        results = []

        for i, board in enumerate(positions):
            # Get legal moves
            legal_moves = [m.uci() for m in board.legal_moves]

            # Query model
            model_response = self.model_fn(board.fen(), legal_moves)
            model_move = parse_uci_move(model_response, strict=strict_parsing)

            if model_move is None:
                model_move = "0000"  # Invalid placeholder

            # Evaluate
            result = evaluate_position(self.engine, board, model_move, depth)
            results.append(result)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(positions)} positions...")

        # Compute metrics
        legal_results = [r for r in results if r.is_legal]
        acpl = sum(r.cp_loss for r in results) / len(results) if results else 0
        legal_rate = len(legal_results) / len(results) if results else 0
        best_rate = sum(1 for r in results if r.is_best) / len(results) if results else 0

        # ACPL excluding illegal moves (for comparison)
        acpl_legal_only = sum(r.cp_loss for r in legal_results) / len(legal_results) if legal_results else 0

        # Config for reproducibility
        config = EvalConfig(
            stockfish_path=self.stockfish_path,
            stockfish_version=self.stockfish_version,
            depth=depth,
            n_positions=len(positions),
            position_source=position_source,
            seed=42,  # Should be passed in
            timestamp=datetime.now().isoformat()
        )

        return {
            "acpl": acpl,
            "acpl_legal_only": acpl_legal_only,
            "legal_rate": legal_rate,
            "best_move_rate": best_rate,
            "n_positions": len(results),
            "n_legal": len(legal_results),
            "n_illegal": len(results) - len(legal_results),
            "config": asdict(config),
            "results": [asdict(r) for r in results]
        }


def demo_random_model(fen: str, legal_moves: List[str]) -> str:
    """Demo: Random move selection"""
    move = random.choice(legal_moves) if legal_moves else "0000"
    return f"<uci_move>{move}</uci_move><rationale>Random selection</rationale>"


if __name__ == "__main__":
    print("Chess Model Evaluator (Fixed)")
    print("=" * 60)

    # Show Stockfish version
    sf_version = get_stockfish_version(STOCKFISH_PATH)
    print(f"Stockfish: {sf_version}")
    print()

    # Try to load real positions first
    training_data = Path(__file__).parent.parent / "data" / "training_elite_30k.jsonl"

    if training_data.exists():
        print(f"Loading 100 positions from training data...")
        positions = load_positions_from_training_data(str(training_data), n=100, seed=42)
        position_source = "training_elite_30k.jsonl"
    else:
        print("WARNING: Using random positions (not recommended)")
        print("  -> For better eval, provide real game positions")
        positions = generate_test_positions_random(n=100, seed=42)
        position_source = "random_walk"

    print(f"Loaded {len(positions)} positions from {position_source}")
    print()

    # Evaluate random baseline
    print("Evaluating random baseline...")
    evaluator = ChessModelEvaluator(demo_random_model)
    results = evaluator.evaluate(
        positions,
        depth=15,
        strict_parsing=True,
        position_source=position_source
    )

    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  ACPL: {results['acpl']:.1f}")
    print(f"  ACPL (legal only): {results['acpl_legal_only']:.1f}")
    print(f"  Legal Move Rate: {results['legal_rate']*100:.1f}%")
    print(f"  Best Move Rate: {results['best_move_rate']*100:.1f}%")
    print(f"  Positions: {results['n_positions']} ({results['n_legal']} legal, {results['n_illegal']} illegal)")
    print()
    print("Config:")
    print(f"  Stockfish: {results['config']['stockfish_version']}")
    print(f"  Depth: {results['config']['depth']}")
    print(f"  Position source: {results['config']['position_source']}")
    print(f"  Timestamp: {results['config']['timestamp']}")

    # Gating check
    print()
    if results['legal_rate'] < 0.995:
        print("WARNING: Legal rate < 99.5% - fix legality before interpreting ACPL")
    else:
        print("Legal rate OK (>= 99.5%)")

    print()
    print("To evaluate your model, implement a model_fn and pass it to ChessModelEvaluator")
