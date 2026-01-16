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

CHECKPOINTING: Saves progress every 1000 positions. Use --resume to continue from checkpoint.
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
import os

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

# Checkpointing
CHECKPOINT_INTERVAL = 1000  # Save every N positions processed

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


def get_checkpoint_path(output_path: Path) -> Path:
    """Get checkpoint file path."""
    return output_path.with_suffix(".checkpoint.json")


def save_checkpoint(checkpoint_path: Path, processed_count: int, accepted_count: int,
                   rejected_count: int, start_time: float):
    """Save checkpoint state."""
    checkpoint = {
        "processed_count": processed_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "elapsed_seconds": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except:
        return None


def count_output_lines(output_path: Path) -> int:
    """Count lines in output file."""
    if not output_path.exists():
        return 0
    with open(output_path, 'r') as f:
        return sum(1 for _ in f)


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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help=f"Save checkpoint every N positions (default {CHECKPOINT_INTERVAL})")
    args = parser.parse_args()

    min_gap = args.min_gap
    num_workers = min(args.workers, MAX_WORKERS)
    checkpoint_interval = args.checkpoint_interval

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = get_checkpoint_path(output_path)

    print("=" * 60)
    print("Winning Moves Filter (with Checkpointing)")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Min eval gap: {min_gap} cp")
    print(f"Decisive threshold: {MIN_DECISIVE_EVAL} cp")
    print(f"Stockfish depth: {STOCKFISH_DEPTH}")
    print(f"Workers: {num_workers}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print()

    # Check for resume
    start_position = 0
    prior_accepted = 0
    prior_elapsed = 0

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_position = checkpoint["processed_count"]
            prior_accepted = count_output_lines(output_path)
            prior_elapsed = checkpoint.get("elapsed_seconds", 0)
            print(f"RESUMING from checkpoint:")
            print(f"  Processed: {start_position:,}")
            print(f"  Accepted so far: {prior_accepted:,}")
            print(f"  Prior elapsed: {prior_elapsed/60:.1f} minutes")
            print()
        else:
            print("No checkpoint found, starting fresh.")
            # Clear output file if starting fresh
            if output_path.exists():
                output_path.unlink()
    else:
        # Starting fresh - clear any existing output
        if output_path.exists():
            output_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

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

    # Sample if requested (only if not resuming)
    if args.sample > 0 and args.sample < total_available and start_position == 0:
        print(f"Sampling {args.sample:,} positions...")
        random.seed(42)
        items = random.sample(items, args.sample)

    # Skip already processed items
    if start_position > 0:
        print(f"Skipping first {start_position:,} already-processed items...")
        items = items[start_position:]

    remaining = len(items)
    print(f"Processing {remaining:,} remaining positions with {num_workers} workers")
    # Estimate: ~0.5 sec per position with multi-pv 2
    est_time = remaining * 0.5 / num_workers / 60
    print(f"Estimated time: {est_time:.1f} minutes")
    print()

    start_time = time.time()

    # Process in chunks for checkpointing
    batch_size = 20  # For multiprocessing
    chunk_size = checkpoint_interval  # For checkpointing

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in append mode
    total_processed = start_position
    total_accepted = prior_accepted
    total_rejected = start_position - prior_accepted if start_position > 0 else 0

    # Process in chunks
    for chunk_start in range(0, len(items), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(items))
        chunk_items = items[chunk_start:chunk_end]

        # Split chunk into batches for multiprocessing
        batches = [chunk_items[i:i+batch_size] for i in range(0, len(chunk_items), batch_size)]

        chunk_results = []
        with Pool(processes=num_workers, initializer=init_worker) as pool:
            for batch_results in tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc=f"Chunk {chunk_start//chunk_size + 1}",
                leave=False
            ):
                chunk_results.extend(batch_results)

        # Write results to file immediately
        with open(output_path, 'a') as f:
            for item in chunk_results:
                f.write(json.dumps(item) + "\n")

        # Update counters
        chunk_accepted = len(chunk_results)
        chunk_rejected = len(chunk_items) - chunk_accepted
        total_processed += len(chunk_items)
        total_accepted += chunk_accepted
        total_rejected += chunk_rejected

        # Save checkpoint
        save_checkpoint(checkpoint_path, total_processed, total_accepted, total_rejected,
                       start_time - prior_elapsed)

        # Progress report
        elapsed = time.time() - start_time + prior_elapsed
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining_items = (start_position + len(items)) - total_processed
        eta_minutes = remaining_items / rate / 60 if rate > 0 else 0

        print(f"  Checkpoint: {total_processed:,}/{start_position + len(items):,} "
              f"({total_processed/(start_position + len(items))*100:.1f}%) | "
              f"Accepted: {total_accepted:,} ({total_accepted/total_processed*100:.1f}%) | "
              f"ETA: {eta_minutes:.0f}m")

    elapsed_time = time.time() - start_time + prior_elapsed

    # Final statistics
    acceptance_rate = total_accepted / total_processed * 100 if total_processed > 0 else 0

    # Count breakdown from output file
    clear_best_count = 0
    decisive_count = 0
    total_gap = 0
    with open(output_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                analysis = item.get('metadata', {}).get('winning_move_analysis', {})
                if analysis.get('is_clear_best', False):
                    clear_best_count += 1
                if analysis.get('is_decisive', False):
                    decisive_count += 1
                total_gap += analysis.get('eval_gap', 0)
            except:
                continue

    avg_gap = total_gap / total_accepted if total_accepted > 0 else 0

    print()
    print("=" * 60)
    print("WINNING MOVES FILTERING COMPLETE")
    print("=" * 60)
    print(f"Processed: {total_processed:,} positions")
    print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
    print(f"Speed: {total_processed/elapsed_time:.2f} pos/sec")
    print()
    print(f"RESULTS:")
    print(f"  Accepted: {total_accepted:,} ({acceptance_rate:.1f}%)")
    print(f"  Rejected: {total_rejected:,} ({100-acceptance_rate:.1f}%)")
    print()
    print(f"BREAKDOWN:")
    print(f"  Clear best move (gap >= {MIN_EVAL_GAP}cp): {clear_best_count:,}")
    print(f"  Decisive positions (|eval| >= {MIN_DECISIVE_EVAL}cp): {decisive_count:,}")
    print(f"  Average eval gap: {avg_gap:.1f} cp")
    print()
    print(f"Saved to: {output_path}")

    # Save final stats
    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total_processed": total_processed,
        "filtered_count": total_accepted,
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

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint file cleaned up.")


if __name__ == "__main__":
    main()
