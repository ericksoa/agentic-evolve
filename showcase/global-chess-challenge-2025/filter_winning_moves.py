#!/usr/bin/env python3
"""
Winning Moves Filter for Global Chess Challenge 2025

Filters training data to keep only positions where there's ONE clearly best move.
Addresses the issue: 65% of training data has equal positions where multiple moves are good.

Criteria:
1. Best move must be significantly better than 2nd best (>= 50 cp gap)
2. Position is decisive (|eval| > 100 cp) OR tactical (large eval swing)

This creates a "winning moves" dataset where the model learns to find THE best move,
not just A reasonable move.
"""
import json
import chess
import chess.engine
import random
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import time

PROJECT_DIR = Path(__file__).parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "training_sf3_500k.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "training_winning_moves.jsonl"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# Filtering parameters
MIN_EVAL_GAP = 50  # Minimum cp gap between best and 2nd best move
MIN_DECISIVE_EVAL = 100  # Position considered decisive if |eval| > this
STOCKFISH_DEPTH = 15  # Deep enough for reliable multi-pv

# Resource constraint
MAX_WORKERS = 2

# Global engine for each worker
_engine = None


def init_worker():
    """Initialize Stockfish engine for this worker."""
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def cleanup_worker():
    """Cleanup Stockfish engine."""
    global _engine
    if _engine:
        _engine.quit()


def process_position(item: dict) -> dict | None:
    """Process a single position to check if it has a clear winning move.

    Returns the item with winning_move analysis if criteria met, None otherwise.
    """
    global _engine

    try:
        metadata = item.get('metadata', {})
        fen = metadata.get('fen')
        recorded_move = metadata.get('best_move')

        if not fen or not recorded_move:
            return None

        board = chess.Board(fen)

        # Get top 2 moves from Stockfish
        result = _engine.analyse(
            board,
            chess.engine.Limit(depth=STOCKFISH_DEPTH),
            multipv=2
        )

        if not result or len(result) < 2:
            return None

        # Extract scores for top 2 moves
        def get_cp(info):
            score = info.get("score")
            if not score:
                return None
            rel = score.relative
            if rel.is_mate():
                return 10000 if rel.mate() > 0 else -10000
            return rel.score()

        best_cp = get_cp(result[0])
        second_cp = get_cp(result[1])

        if best_cp is None or second_cp is None:
            return None

        best_move = result[0]["pv"][0].uci() if result[0].get("pv") else None
        second_move = result[1]["pv"][0].uci() if result[1].get("pv") else None

        if not best_move:
            return None

        # Calculate the gap between best and second best
        eval_gap = best_cp - second_cp

        # Check criteria:
        # 1. Large gap between best and 2nd best (clear winning move)
        # 2. OR position is decisive (|best_cp| > 100)

        is_clear_best = eval_gap >= MIN_EVAL_GAP
        is_decisive = abs(best_cp) >= MIN_DECISIVE_EVAL

        # Accept if there's a clear best move OR position is decisive
        # But require at least SOME gap (20cp) to avoid totally equal positions
        if not (is_clear_best or (is_decisive and eval_gap >= 20)):
            return None

        # Verify recorded move matches or is close to SF best
        # Accept if recorded move is in top-2
        recorded_in_top2 = recorded_move in [best_move, second_move]

        if not recorded_in_top2:
            # Also accept if recorded move is within 30cp of best
            # (accounting for horizon effects)
            return None

        # Create enriched item
        enriched_item = item.copy()
        enriched_item['metadata'] = metadata.copy()
        enriched_item['metadata']['winning_move_analysis'] = {
            'sf_best': best_move,
            'sf_second': second_move,
            'best_cp': best_cp,
            'second_cp': second_cp,
            'eval_gap': eval_gap,
            'is_clear_best': is_clear_best,
            'is_decisive': is_decisive,
            'recorded_matches_top2': recorded_in_top2,
            'analysis_depth': STOCKFISH_DEPTH
        }

        return enriched_item

    except Exception as e:
        return None


def process_batch(items: list) -> list:
    """Process a batch of items in this worker."""
    results = []
    for item in items:
        result = process_position(item)
        if result:
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Filter for winning moves - positions with clear best move")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Input training file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output filtered training file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N positions (0 = all)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers (max 2)")
    parser.add_argument("--min-gap", type=int, default=MIN_EVAL_GAP,
                        help=f"Minimum cp gap between best and 2nd best (default {MIN_EVAL_GAP})")
    args = parser.parse_args()

    min_gap = args.min_gap
    num_workers = min(args.workers, MAX_WORKERS)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("Winning Moves Filter")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Min eval gap: {min_gap} cp (note: uses module default {MIN_EVAL_GAP})")
    print(f"Decisive threshold: {MIN_DECISIVE_EVAL} cp")
    print(f"Stockfish depth: {STOCKFISH_DEPTH}")
    print(f"Workers: {num_workers}")
    print()

    # Load training data
    print("Loading training data...")
    if not input_path.exists():
        # Try alternative inputs
        alternatives = [
            PROJECT_DIR / "data" / "training_sf3_combined.jsonl",
            PROJECT_DIR / "data" / "training_mega_360k.jsonl",
            PROJECT_DIR / "data" / "training_elite_30k.jsonl"
        ]
        for alt in alternatives:
            if alt.exists():
                print(f"  Input not found, using: {alt}")
                input_path = alt
                break
        else:
            print(f"ERROR: No input file found. Tried: {input_path} and alternatives")
            return

    items = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except:
                continue

    total_available = len(items)
    print(f"Loaded {total_available:,} training examples")

    # Sample if requested
    if args.sample > 0 and args.sample < total_available:
        print(f"Sampling {args.sample:,} positions...")
        random.seed(42)
        items = random.sample(items, args.sample)

    print(f"Processing {len(items):,} positions with {num_workers} workers")
    # Estimate: ~0.5 sec per position with multi-pv 2
    est_time = len(items) * 0.5 / num_workers / 60
    print(f"Estimated time: {est_time:.1f} minutes")
    print()

    start_time = time.time()

    # Split into batches
    batch_size = 20
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    # Process with multiprocessing
    filtered_items = []

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for batch_results in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Filtering"
        ):
            filtered_items.extend(batch_results)

    elapsed_time = time.time() - start_time

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in filtered_items:
            f.write(json.dumps(item) + "\n")

    # Statistics
    acceptance_rate = len(filtered_items) / len(items) * 100 if items else 0

    # Analyze the filtered results
    clear_best_count = sum(
        1 for item in filtered_items
        if item.get('metadata', {}).get('winning_move_analysis', {}).get('is_clear_best', False)
    )
    decisive_count = sum(
        1 for item in filtered_items
        if item.get('metadata', {}).get('winning_move_analysis', {}).get('is_decisive', False)
    )

    avg_gap = sum(
        item.get('metadata', {}).get('winning_move_analysis', {}).get('eval_gap', 0)
        for item in filtered_items
    ) / len(filtered_items) if filtered_items else 0

    print()
    print("=" * 60)
    print("WINNING MOVES FILTERING COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(items):,} positions")
    print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
    print(f"Speed: {len(items)/elapsed_time:.2f} pos/sec")
    print()
    print(f"RESULTS:")
    print(f"  Accepted: {len(filtered_items):,} ({acceptance_rate:.1f}%)")
    print(f"  Rejected: {len(items) - len(filtered_items):,} ({100-acceptance_rate:.1f}%)")
    print()
    print(f"BREAKDOWN:")
    print(f"  Clear best move (gap >= {MIN_EVAL_GAP}cp): {clear_best_count:,}")
    print(f"  Decisive positions (|eval| >= {MIN_DECISIVE_EVAL}cp): {decisive_count:,}")
    print(f"  Average eval gap: {avg_gap:.1f} cp")
    print()
    print(f"Saved to: {output_path}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total_processed": len(items),
        "filtered_count": len(filtered_items),
        "acceptance_rate": acceptance_rate,
        "clear_best_count": clear_best_count,
        "decisive_count": decisive_count,
        "avg_eval_gap": avg_gap,
        "min_gap_threshold": MIN_EVAL_GAP,
        "decisive_threshold": MIN_DECISIVE_EVAL,
        "depth": STOCKFISH_DEPTH,
        "elapsed_seconds": elapsed_time
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
