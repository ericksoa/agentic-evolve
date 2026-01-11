"""
Training Data Generation for Global Chess Challenge 2025

Generates fine-tuning data from chess games and puzzles.
This is a key target for evolution - different data strategies yield different results.
"""
import chess
import chess.pgn
import chess.engine
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
import io

# Default Stockfish path
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


@dataclass
class TrainingExample:
    """A single training example for chess LLM fine-tuning"""
    fen: str
    legal_moves: List[str]
    best_move: str
    best_move_san: str
    cp_score: Optional[int]  # Centipawn evaluation
    rationale: str
    game_phase: str  # opening, middlegame, endgame
    difficulty: str  # easy, medium, hard


@dataclass
class DataGenerationConfig:
    """
    Configuration for training data generation.

    These parameters are EVOLUTION TARGETS - different configs yield
    different model performance.
    """
    # Game selection
    min_elo: int = 2000  # Minimum player Elo
    max_elo: int = 2800  # Maximum player Elo
    time_control: str = "any"  # bullet, blitz, rapid, classical, any

    # Position selection
    min_move_number: int = 5  # Skip opening book moves
    max_move_number: int = 80  # Skip very long endgames
    exclude_forced: bool = True  # Skip positions with only 1 legal move

    # Game phase distribution
    opening_fraction: float = 0.2  # Moves 1-15
    middlegame_fraction: float = 0.5  # Moves 16-35
    endgame_fraction: float = 0.3  # Moves 36+

    # Difficulty distribution
    easy_fraction: float = 0.3  # Clear best move (cp diff > 100)
    medium_fraction: float = 0.4  # Good best move (cp diff 30-100)
    hard_fraction: float = 0.3  # Subtle best move (cp diff < 30)

    # Augmentation
    mirror_positions: bool = True  # Flip board a1<->h8
    both_colors: bool = True  # Include positions from both perspectives

    # Stockfish analysis
    stockfish_depth: int = 18

    # Output
    max_examples: int = 100000


def get_game_phase(board: chess.Board) -> str:
    """Determine game phase based on material and move number"""
    move_num = board.fullmove_number

    if move_num <= 15:
        return "opening"
    elif move_num <= 35:
        return "middlegame"
    else:
        return "endgame"


def get_piece_count(board: chess.Board) -> int:
    """Count total pieces (excluding kings)"""
    return len(board.piece_map()) - 2  # Subtract 2 kings


def mirror_fen(fen: str) -> str:
    """Mirror position horizontally (a1 <-> h8)"""
    parts = fen.split()
    position = parts[0]

    # Mirror each rank
    mirrored_ranks = []
    for rank in position.split('/'):
        mirrored_ranks.append(rank[::-1])

    parts[0] = '/'.join(mirrored_ranks)
    return ' '.join(parts)


def generate_rationale(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth: int = 10
) -> str:
    """Generate a simple rationale for a move"""
    san = board.san(move)

    # Check basic patterns
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        return f"Captures {captured.symbol().upper()} on {chess.square_name(move.to_square)}"

    if board.gives_check(move):
        return f"Plays {san} with check"

    # Default rationale
    piece = board.piece_at(move.from_square)
    piece_name = chess.piece_name(piece.piece_type).title()
    return f"Develops {piece_name} to {chess.square_name(move.to_square)}"


def extract_examples_from_game(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    config: DataGenerationConfig
) -> Iterator[TrainingExample]:
    """Extract training examples from a single game"""
    board = game.board()

    for move_num, node in enumerate(game.mainline()):
        move = node.move

        # Skip based on move number
        if move_num < config.min_move_number or move_num > config.max_move_number:
            board.push(move)
            continue

        # Skip forced moves
        legal_moves = list(board.legal_moves)
        if config.exclude_forced and len(legal_moves) <= 1:
            board.push(move)
            continue

        # Analyze with Stockfish
        try:
            analysis = engine.analyse(
                board,
                chess.engine.Limit(depth=config.stockfish_depth)
            )
            best_move = analysis["pv"][0]
            score = analysis["score"].white()

            if score.is_mate():
                cp_score = 10000 if score.mate() > 0 else -10000
            else:
                cp_score = score.score()

            # Determine difficulty (based on second-best move difference)
            # Simplified: use absolute score
            if abs(cp_score) > 300:
                difficulty = "easy"
            elif abs(cp_score) > 100:
                difficulty = "medium"
            else:
                difficulty = "hard"

            yield TrainingExample(
                fen=board.fen(),
                legal_moves=[m.uci() for m in legal_moves],
                best_move=best_move.uci(),
                best_move_san=board.san(best_move),
                cp_score=cp_score,
                rationale=generate_rationale(board, best_move, engine),
                game_phase=get_game_phase(board),
                difficulty=difficulty
            )

        except Exception as e:
            pass  # Skip problematic positions

        board.push(move)


def generate_training_data(
    pgn_path: str,
    output_path: str,
    config: DataGenerationConfig = DataGenerationConfig()
) -> Dict[str, Any]:
    """
    Generate training data from PGN file.

    Returns statistics about generated data.
    """
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    examples = []

    with open(pgn_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Check Elo filter
            white_elo = game.headers.get("WhiteElo", "0")
            black_elo = game.headers.get("BlackElo", "0")
            try:
                if int(white_elo) < config.min_elo or int(black_elo) < config.min_elo:
                    continue
            except ValueError:
                continue

            # Extract examples
            for example in extract_examples_from_game(game, engine, config):
                examples.append(example)

                if len(examples) >= config.max_examples:
                    break

            if len(examples) >= config.max_examples:
                break

    engine.quit()

    # Save to JSONL
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + '\n')

    # Compute statistics
    stats = {
        "total_examples": len(examples),
        "game_phases": {
            "opening": sum(1 for e in examples if e.game_phase == "opening"),
            "middlegame": sum(1 for e in examples if e.game_phase == "middlegame"),
            "endgame": sum(1 for e in examples if e.game_phase == "endgame"),
        },
        "difficulties": {
            "easy": sum(1 for e in examples if e.difficulty == "easy"),
            "medium": sum(1 for e in examples if e.difficulty == "medium"),
            "hard": sum(1 for e in examples if e.difficulty == "hard"),
        }
    }

    return stats


def create_instruction_format(example: TrainingExample) -> Dict[str, str]:
    """
    Convert training example to instruction format for fine-tuning.

    This is an EVOLUTION TARGET - different prompt formats yield different results.
    """
    # System prompt
    system = "You are a chess grandmaster. Given a position and legal moves, select the best move."

    # User prompt (position + moves)
    user = f"""Position (FEN): {example.fen}
Side to move: {'White' if 'w' in example.fen else 'Black'}
Legal moves: {', '.join(example.legal_moves)}

Select the best move."""

    # Assistant response
    assistant = f"""<uci_move>{example.best_move}</uci_move>
<rationale>{example.rationale}</rationale>"""

    return {
        "system": system,
        "user": user,
        "assistant": assistant
    }


if __name__ == "__main__":
    print("Chess Training Data Generator")
    print("=" * 50)

    # Demo: Generate from a sample game
    sample_pgn = """
[Event "Demo"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2400"]
[BlackElo "2400"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. a4 Bg7
16. Bd3 c6 17. Bg5 Qc7 18. Qd2 h6 19. Be3 Rad8 20. axb5 axb5 21. b4 1-0
"""

    # Parse and extract
    game = chess.pgn.read_game(io.StringIO(sample_pgn))

    print(f"Game: {game.headers.get('White')} vs {game.headers.get('Black')}")
    print(f"Processing...")

    # Create a simple config
    config = DataGenerationConfig(
        min_elo=0,
        max_examples=10,
        stockfish_depth=12
    )

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        examples = list(extract_examples_from_game(game, engine, config))
        engine.quit()

        print(f"\nGenerated {len(examples)} training examples")

        if examples:
            print("\nSample example:")
            ex = examples[0]
            print(f"  FEN: {ex.fen[:50]}...")
            print(f"  Best move: {ex.best_move} ({ex.best_move_san})")
            print(f"  Phase: {ex.game_phase}")
            print(f"  Difficulty: {ex.difficulty}")

            print("\nInstruction format:")
            inst = create_instruction_format(ex)
            print(f"  System: {inst['system'][:50]}...")
            print(f"  User: {inst['user'][:50]}...")
            print(f"  Assistant: {inst['assistant']}")

    except FileNotFoundError:
        print(f"\nError: Stockfish not found at {STOCKFISH_PATH}")
        print("Install with: brew install stockfish (macOS)")
        print("Or update STOCKFISH_PATH in this file")
