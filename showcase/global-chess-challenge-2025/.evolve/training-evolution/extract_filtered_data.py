#!/usr/bin/env python3
"""
Extract Training Data with Quality Filtering

Key insight: Train ONLY on moves where the human move matches Stockfish's
top-N recommendations. This ensures we train on "correct" moves, not mistakes.

This is the main lever for improving ACPL:
- No filter: Train on all moves (including blunders) → ACPL ~200-300
- SF top-10: Train on reasonable moves → ACPL ~100-150 (estimated)
- SF top-3: Train on good moves → ACPL ~80-100 (estimated)
- SF top-1: Train on best moves only → ACPL ~50-80 (estimated)

Usage:
    python extract_filtered_data.py --sf-top 3 --min-elo 2400 --output training_sf3.jsonl
"""
import os
import sys
import json
import chess
import chess.engine
import chess.pgn
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

# Stockfish path
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


@dataclass
class FilterConfig:
    min_elo: int = 2200
    max_elo: int = 3000
    stockfish_top_n: int = 3  # Human move must be in top N
    max_centipawn_loss: int = 100  # Skip moves with CPL > this
    min_move_number: int = 5  # Skip early opening
    max_move_number: int = 80  # Skip very long games
    include_opening: bool = True  # Moves 5-15
    include_middlegame: bool = True  # Moves 16-35
    include_endgame: bool = True  # Moves 36+
    stockfish_depth: int = 12  # Analysis depth


def get_move_phase(move_number: int) -> str:
    """Classify move into game phase"""
    if move_number <= 15:
        return "opening"
    elif move_number <= 35:
        return "middlegame"
    else:
        return "endgame"


def analyze_position(engine, board: chess.Board, depth: int = 12) -> List[dict]:
    """
    Analyze position and return top N moves with evaluations.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=10)

    results = []
    for pv_info in info if isinstance(info, list) else [info]:
        if "pv" in pv_info and len(pv_info["pv"]) > 0:
            move = pv_info["pv"][0]
            score = pv_info.get("score")
            if score:
                cp = score.white().score(mate_score=10000)
                results.append({
                    "move": move.uci(),
                    "cp": cp,
                })

    return results


def format_training_example(
    fen: str,
    legal_moves: List[str],
    best_move: str,
    rationale: str = "",
) -> dict:
    """Format as chat training example"""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a chess grandmaster. Analyze positions and select the best move."
            },
            {
                "role": "user",
                "content": f"FEN: {fen}\nMoves: {', '.join(legal_moves)}\nBest?"
            },
            {
                "role": "assistant",
                "content": f"<uci_move>{best_move}</uci_move>\n<rationale>{rationale}</rationale>"
            }
        ],
        "metadata": {
            "fen": fen,
            "best_move": best_move,
        }
    }


def extract_from_pgn(
    pgn_path: str,
    config: FilterConfig,
    max_examples: int = 50000,
    output_path: str = None,
) -> List[dict]:
    """
    Extract training examples from PGN with Stockfish filtering.
    """
    print(f"Extracting from {pgn_path}")
    print(f"Config: Elo>={config.min_elo}, SF_top={config.stockfish_top_n}, CPL<={config.max_centipawn_loss}")

    # Start Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    examples = []
    games_processed = 0
    positions_analyzed = 0
    positions_accepted = 0

    with open(pgn_path) as f:
        while len(examples) < max_examples:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_processed += 1

            # Check Elo
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            avg_elo = (white_elo + black_elo) // 2

            if avg_elo < config.min_elo or avg_elo > config.max_elo:
                continue

            # Process game
            board = game.board()
            move_number = 0

            for move in game.mainline_moves():
                move_number += 1
                human_move_uci = move.uci()

                # Skip based on move number
                if move_number < config.min_move_number or move_number > config.max_move_number:
                    board.push(move)
                    continue

                # Skip based on phase
                phase = get_move_phase(move_number)
                if phase == "opening" and not config.include_opening:
                    board.push(move)
                    continue
                if phase == "middlegame" and not config.include_middlegame:
                    board.push(move)
                    continue
                if phase == "endgame" and not config.include_endgame:
                    board.push(move)
                    continue

                # Skip terminal positions
                if board.is_game_over():
                    break

                positions_analyzed += 1

                # Analyze with Stockfish
                if config.stockfish_top_n > 0:
                    sf_moves = analyze_position(engine, board, config.stockfish_depth)

                    # Check if human move is in top N
                    top_n_moves = [m["move"] for m in sf_moves[:config.stockfish_top_n]]

                    if human_move_uci not in top_n_moves:
                        board.push(move)
                        continue

                    # Calculate centipawn loss
                    if len(sf_moves) >= 2:
                        best_cp = sf_moves[0]["cp"]
                        human_idx = next((i for i, m in enumerate(sf_moves) if m["move"] == human_move_uci), -1)
                        if human_idx >= 0:
                            human_cp = sf_moves[human_idx]["cp"]
                            # CPL from mover's perspective
                            if board.turn == chess.WHITE:
                                cpl = best_cp - human_cp
                            else:
                                cpl = human_cp - best_cp
                            cpl = max(0, cpl)

                            if cpl > config.max_centipawn_loss:
                                board.push(move)
                                continue

                # Position passes all filters!
                positions_accepted += 1

                fen = board.fen()
                legal_moves = [m.uci() for m in board.legal_moves]

                # Generate rationale based on phase
                if phase == "opening":
                    rationale = f"Solid {phase} move maintaining good position"
                elif phase == "middlegame":
                    rationale = f"Strong {phase} continuation"
                else:
                    rationale = f"Accurate {phase} technique"

                example = format_training_example(fen, legal_moves, human_move_uci, rationale)
                example["metadata"]["avg_elo"] = avg_elo
                example["metadata"]["phase"] = phase
                examples.append(example)

                board.push(move)

                # Progress
                if positions_accepted % 1000 == 0:
                    print(f"  Extracted {positions_accepted} examples (analyzed {positions_analyzed}, {positions_accepted/positions_analyzed*100:.1f}% acceptance)")

            if games_processed % 100 == 0:
                print(f"  Processed {games_processed} games...")

    engine.quit()

    print(f"\nExtraction complete:")
    print(f"  Games processed: {games_processed}")
    print(f"  Positions analyzed: {positions_analyzed}")
    print(f"  Positions accepted: {positions_accepted} ({positions_accepted/positions_analyzed*100:.1f}%)")

    # Save if output path provided
    if output_path:
        # Shuffle
        random.shuffle(examples)

        with open(output_path, 'w') as f:
            for ex in examples[:max_examples]:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {min(len(examples), max_examples)} examples to {output_path}")

    return examples[:max_examples]


def main():
    parser = argparse.ArgumentParser(description="Extract filtered training data")
    parser.add_argument("--pgn", type=str, default="data/lichess_elite.pgn", help="Input PGN file")
    parser.add_argument("--output", type=str, default="training_filtered.jsonl", help="Output JSONL file")
    parser.add_argument("--min-elo", type=int, default=2200, help="Minimum average Elo")
    parser.add_argument("--sf-top", type=int, default=3, help="Stockfish top-N filter (0=disabled)")
    parser.add_argument("--max-cpl", type=int, default=100, help="Maximum centipawn loss")
    parser.add_argument("--max-examples", type=int, default=50000, help="Maximum examples")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish depth")
    args = parser.parse_args()

    config = FilterConfig(
        min_elo=args.min_elo,
        stockfish_top_n=args.sf_top,
        max_centipawn_loss=args.max_cpl,
        stockfish_depth=args.depth,
    )

    pgn_path = Path(__file__).parent.parent.parent / args.pgn
    output_path = Path(__file__).parent.parent.parent / "data" / args.output

    if not pgn_path.exists():
        print(f"PGN file not found: {pgn_path}")
        print("Please download a PGN database first.")
        return

    extract_from_pgn(
        str(pgn_path),
        config,
        max_examples=args.max_examples,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
