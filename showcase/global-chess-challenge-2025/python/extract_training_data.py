"""
Training Data Extraction from PGN Files

Extracts high-quality training examples from chess games:
1. Parse PGN games
2. Filter by Elo threshold
3. Analyze positions with Stockfish
4. Format as instruction-tuning data

Usage:
    python extract_training_data.py --pgn data/games/lichess.pgn --output data/training.jsonl
"""
import argparse
import chess
import chess.pgn
import chess.engine
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Stockfish configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
STOCKFISH_DEPTH = 20
STOCKFISH_THREADS = 1

# Scale factor size (one scale per 16 FP4 values in NVFP4, but for chess we use this for batching)
BATCH_SIZE = 100


@dataclass
class ChessExample:
    """A single training example"""
    fen: str
    side_to_move: str
    legal_moves: List[str]
    best_move_uci: str
    best_move_san: str
    evaluation_cp: int
    game_phase: str
    white_elo: int
    black_elo: int
    rationale: str


@dataclass
class DataConfig:
    """Configuration for data extraction"""
    min_elo: int = 2200
    max_elo: int = 2800
    min_move_number: int = 10
    max_move_number: int = 60
    opening_fraction: float = 0.2
    middlegame_fraction: float = 0.6
    endgame_fraction: float = 0.2
    stockfish_depth: int = 20
    max_examples: int = 50000
    exclude_forced: bool = True


def get_game_phase(board: chess.Board) -> str:
    """Determine game phase from position"""
    # Count material
    piece_count = len(board.piece_map())
    move_num = board.fullmove_number

    if move_num <= 12:
        return "opening"
    elif piece_count <= 10 or move_num > 40:
        return "endgame"
    else:
        return "middlegame"


def get_piece_activity(board: chess.Board) -> str:
    """Get a brief description of piece activity"""
    pieces = []
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            square_name = chess.square_name(square)
            piece_name = chess.piece_name(piece.piece_type)
            pieces.append(f"{piece_name} on {square_name}")
    return ", ".join(pieces[:5])  # Limit to 5 pieces


def generate_rationale(board: chess.Board, move: chess.Move, eval_cp: int) -> str:
    """Generate a brief rationale for the move"""
    san = board.san(move)

    # Check for special moves
    if board.is_castling(move):
        return f"Castles for king safety"

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            return f"Captures {chess.piece_name(captured.piece_type)} with {san}"

    if board.gives_check(move):
        return f"Plays {san} with check"

    # Evaluation-based rationale
    if eval_cp > 200:
        return f"Maintains strong advantage with {san}"
    elif eval_cp > 50:
        return f"Keeps slight edge with {san}"
    elif eval_cp > -50:
        return f"Plays solid move {san} in equal position"
    else:
        return f"Best defensive try with {san}"


def analyze_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = STOCKFISH_DEPTH
) -> Tuple[chess.Move, int]:
    """Analyze position with Stockfish, return best move and evaluation"""
    result = engine.analyse(board, chess.engine.Limit(depth=depth))

    best_move = result["pv"][0]
    score = result["score"].white()

    # Convert to centipawns
    if score.is_mate():
        eval_cp = 10000 if score.mate() > 0 else -10000
    else:
        eval_cp = score.score()

    # Adjust for side to move
    if board.turn == chess.BLACK:
        eval_cp = -eval_cp

    return best_move, eval_cp


def process_game(
    game_pgn: str,
    config: DataConfig,
    stockfish_path: str = STOCKFISH_PATH
) -> List[ChessExample]:
    """Process a single game and extract training examples"""
    examples = []

    try:
        import io
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        if game is None:
            return examples

        # Check Elo
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            return examples

        if white_elo < config.min_elo or black_elo < config.min_elo:
            return examples
        if white_elo > config.max_elo or black_elo > config.max_elo:
            return examples

        # Initialize engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": STOCKFISH_THREADS})

        board = game.board()
        move_num = 0

        for node in game.mainline():
            move_num += 1

            # Skip based on move number
            if move_num < config.min_move_number:
                board.push(node.move)
                continue
            if move_num > config.max_move_number:
                break

            # Skip forced moves
            legal_moves = list(board.legal_moves)
            if config.exclude_forced and len(legal_moves) <= 1:
                board.push(node.move)
                continue

            # Skip positions with very few pieces (trivial endgames)
            if len(board.piece_map()) < 6:
                board.push(node.move)
                continue

            # Analyze position
            try:
                best_move, eval_cp = analyze_position(board, engine, config.stockfish_depth)
            except Exception:
                board.push(node.move)
                continue

            # Create example
            example = ChessExample(
                fen=board.fen(),
                side_to_move="White" if board.turn == chess.WHITE else "Black",
                legal_moves=[m.uci() for m in legal_moves],
                best_move_uci=best_move.uci(),
                best_move_san=board.san(best_move),
                evaluation_cp=eval_cp,
                game_phase=get_game_phase(board),
                white_elo=white_elo,
                black_elo=black_elo,
                rationale=generate_rationale(board, best_move, eval_cp)
            )
            examples.append(example)

            board.push(node.move)

        engine.quit()

    except Exception as e:
        print(f"Error processing game: {e}", file=sys.stderr)

    return examples


def read_games_from_pgn(pgn_path: str, max_games: int = 10000) -> Iterator[str]:
    """Read games from PGN file as strings"""
    with open(pgn_path, 'r', errors='ignore') as f:
        game_lines = []
        game_count = 0

        for line in f:
            game_lines.append(line)

            # Empty line after moves signals end of game
            if line.strip() == "" and game_lines and any('[' in l for l in game_lines):
                # Check if we have moves (not just headers)
                has_moves = any(not l.startswith('[') and l.strip() and not l.strip() == ""
                               for l in game_lines)
                if has_moves:
                    yield ''.join(game_lines)
                    game_count += 1
                    if game_count >= max_games:
                        return
                game_lines = []

        # Last game
        if game_lines:
            yield ''.join(game_lines)


def format_for_training(example: ChessExample, prompt_style: str = "concise") -> Dict[str, str]:
    """Format example for instruction tuning"""

    if prompt_style == "concise":
        user_prompt = f"FEN: {example.fen}\nMoves: {', '.join(example.legal_moves)}\nBest?"
    elif prompt_style == "detailed":
        user_prompt = f"""Position: {example.fen}
Side to move: {example.side_to_move}
Legal moves: {', '.join(example.legal_moves)}

Select the best move."""
    else:  # standard
        user_prompt = f"""Chess position (FEN): {example.fen}
{example.side_to_move} to move.
Available moves: {', '.join(example.legal_moves)}

Choose the best move."""

    assistant_response = f"""<uci_move>{example.best_move_uci}</uci_move>
<rationale>{example.rationale}</rationale>"""

    return {
        "messages": [
            {"role": "system", "content": "You are a chess grandmaster. Analyze positions and select the best move."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ],
        "metadata": {
            "fen": example.fen,
            "best_move": example.best_move_uci,
            "eval_cp": example.evaluation_cp,
            "phase": example.game_phase,
            "avg_elo": (example.white_elo + example.black_elo) // 2
        }
    }


def extract_training_data(
    pgn_path: str,
    output_path: str,
    config: DataConfig,
    num_workers: int = 2,  # Keep low for Mac
    prompt_style: str = "concise"
) -> Dict[str, Any]:
    """
    Main extraction function.

    Processes PGN file and outputs JSONL training data.
    """
    print(f"Extracting training data from: {pgn_path}")
    print(f"Config: min_elo={config.min_elo}, depth={config.stockfish_depth}")
    print(f"Using {num_workers} workers")

    # Process games directly with chess.pgn
    all_examples = []
    processed = 0
    skipped_elo = 0

    # Process sequentially for reliability
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1})

    with open(pgn_path, 'r', errors='ignore') as pgn_file:
        while len(all_examples) < config.max_examples:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Check Elo
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                continue

            if white_elo < config.min_elo or black_elo < config.min_elo:
                skipped_elo += 1
                continue

            board = game.board()
            move_num = 0

            for node in game.mainline():
                move_num += 1

                if move_num < config.min_move_number:
                    board.push(node.move)
                    continue
                if move_num > config.max_move_number:
                    break

                legal_moves = list(board.legal_moves)
                if config.exclude_forced and len(legal_moves) <= 1:
                    board.push(node.move)
                    continue

                if len(board.piece_map()) < 6:
                    board.push(node.move)
                    continue

                try:
                    best_move, eval_cp = analyze_position(board, engine, config.stockfish_depth)

                    example = ChessExample(
                        fen=board.fen(),
                        side_to_move="White" if board.turn == chess.WHITE else "Black",
                        legal_moves=[m.uci() for m in legal_moves],
                        best_move_uci=best_move.uci(),
                        best_move_san=board.san(best_move),
                        evaluation_cp=eval_cp,
                        game_phase=get_game_phase(board),
                        white_elo=white_elo,
                        black_elo=black_elo,
                        rationale=generate_rationale(board, best_move, eval_cp)
                    )
                    all_examples.append(example)

                    if len(all_examples) >= config.max_examples:
                        break

                except Exception as e:
                    pass

                board.push(node.move)

            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} games ({skipped_elo} skipped by Elo), {len(all_examples)} examples...")

    engine.quit()
    print(f"Extracted {len(all_examples)} examples from {processed} games (skipped {skipped_elo} by Elo)")

    # Balance by game phase
    phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
    for ex in all_examples:
        phase_counts[ex.game_phase] += 1
    print(f"Phase distribution: {phase_counts}")

    # Format and save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        for example in all_examples:
            formatted = format_for_training(example, prompt_style)
            f.write(json.dumps(formatted) + '\n')

    # Statistics
    stats = {
        "total_examples": len(all_examples),
        "games_processed": processed,
        "phase_distribution": phase_counts,
        "avg_elo": sum(ex.white_elo + ex.black_elo for ex in all_examples) // (2 * len(all_examples)) if all_examples else 0,
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    }

    # Save stats
    stats_path = output_path.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {stats_path}")
    print(f"Average Elo: {stats['avg_elo']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract chess training data from PGN")
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--min-elo", type=int, default=2200, help="Minimum Elo filter")
    parser.add_argument("--max-examples", type=int, default=10000, help="Maximum examples to extract")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish analysis depth")
    parser.add_argument("--prompt-style", default="concise", choices=["concise", "detailed", "standard"])
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (keep low)")

    args = parser.parse_args()

    config = DataConfig(
        min_elo=args.min_elo,
        max_examples=args.max_examples,
        stockfish_depth=args.depth
    )

    stats = extract_training_data(
        args.pgn,
        args.output,
        config,
        num_workers=args.workers,
        prompt_style=args.prompt_style
    )

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Examples: {stats['total_examples']}")
    print(f"Games: {stats['games_processed']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
