#!/usr/bin/env python3
"""
Gen108: Main Runner for Rust-Python Hybrid Pipeline.

Usage:
    python gen108_runner.py                   # Quick benchmark (n=20, 50)
    python gen108_runner.py --n 100           # Single n value
    python gen108_runner.py --max-n 200       # Full submission
    python gen108_runner.py --compare-rust    # Compare with Rust-only

Pipeline:
1. Rust: Generate packing with evolved greedy (fast, good initial placement)
2. Python: SA refinement with GPU-accelerated collision (exploit different moves)
3. Parallel: Run multiple workers, pick best
4. Output: JSON/CSV for analysis or submission
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

from rust_hybrid import (
    RustHybridConfig,
    generate_rust_packing,
    hybrid_pipeline_single_n,
    EXPORT_BIN,
    RUST_DIR,
)
from parallel_refine import ParallelConfig, parallel_refine
from hybrid_collision import HybridCollisionChecker


def run_benchmark(n_values: list = [20, 50], verbose: bool = True):
    """Run quick benchmark on specified n values."""
    print("Gen108 Hybrid Pipeline Benchmark")
    print("=" * 60)

    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        print(f"Build with: cd {RUST_DIR} && cargo build --release --bin export_packing")
        return

    checker = HybridCollisionChecker()

    config = RustHybridConfig(
        rust_num_runs=3,
        sa_iterations=10000,
        sa_restarts=3,
        polish_iterations=2000,
    )

    results = []
    total_start = time.perf_counter()

    for n in n_values:
        refined, final_side, rust_side, improvement = hybrid_pipeline_single_n(
            n, config, checker, verbose=verbose
        )
        results.append((n, rust_side, final_side, improvement))

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'n':>4} | {'Rust':>8} | {'Hybrid':>8} | {'Improvement':>12}")
    print("-" * 40)
    for n, rust_side, final_side, improvement in results:
        print(f"{n:>4} | {rust_side:>8.4f} | {final_side:>8.4f} | {improvement:>+11.2f}%")

    print(f"\nTotal time: {total_time:.1f}s")


def run_single_n(n: int, parallel: bool = False, n_workers: int = 6, verbose: bool = True):
    """Run pipeline for a single n value."""
    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        return

    if parallel:
        config = ParallelConfig(
            n_workers=n_workers,
            rust_runs_per_worker=2,
            sa_iterations=10000,
            sa_restarts=3,
            polish_iterations=2000,
        )
        best_configs, best_side, all_sides = parallel_refine(n, config, verbose=verbose)

        if best_configs is not None:
            print(f"\nFinal: n={n}, side={best_side:.4f}")
            print(f"Range: {min(all_sides):.4f} - {max(all_sides):.4f}")
    else:
        checker = HybridCollisionChecker()
        config = RustHybridConfig(
            rust_num_runs=5,
            sa_iterations=15000,
            sa_restarts=4,
            polish_iterations=3000,
        )
        refined, final_side, rust_side, improvement = hybrid_pipeline_single_n(
            n, config, checker, verbose=verbose
        )


def compare_with_rust(n_values: list = [10, 20, 50, 100]):
    """Compare hybrid pipeline with Rust-only baseline."""
    print("Rust vs Hybrid Comparison")
    print("=" * 60)

    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        return

    checker = HybridCollisionChecker()

    hybrid_config = RustHybridConfig(
        rust_num_runs=1,  # Same as Rust baseline
        sa_iterations=10000,
        sa_restarts=3,
        polish_iterations=2000,
    )

    results = []

    for n in n_values:
        print(f"\n{'='*40}")
        print(f"n = {n}")
        print("=" * 40)

        # Rust only (best of 5)
        print("\nRust only (best of 5):")
        rust_sides = []
        for i in range(5):
            _, rust_side = generate_rust_packing(n, num_runs=1)
            rust_sides.append(rust_side)
            print(f"  Run {i+1}: {rust_side:.4f}")

        best_rust = min(rust_sides)
        print(f"  Best: {best_rust:.4f}")

        # Hybrid (single run + Python refine)
        print("\nHybrid (Rust + Python SA):")
        refined, final_side, rust_side, improvement = hybrid_pipeline_single_n(
            n, hybrid_config, checker, verbose=False
        )
        print(f"  Rust base: {rust_side:.4f}")
        print(f"  After SA:  {final_side:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")

        results.append((n, best_rust, final_side))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'n':>4} | {'Rust Best':>10} | {'Hybrid':>10} | {'Delta':>10}")
    print("-" * 50)

    total_rust_score = 0
    total_hybrid_score = 0

    for n, best_rust, hybrid_side in results:
        delta = (best_rust - hybrid_side) / best_rust * 100
        print(f"{n:>4} | {best_rust:>10.4f} | {hybrid_side:>10.4f} | {delta:>+9.2f}%")

        total_rust_score += best_rust**2 / n
        total_hybrid_score += hybrid_side**2 / n

    print("-" * 50)
    score_improvement = (total_rust_score - total_hybrid_score) / total_rust_score * 100
    print(f"Score: {total_rust_score:.4f} (Rust) vs {total_hybrid_score:.4f} (Hybrid)")
    print(f"Score improvement: {score_improvement:+.2f}%")


def generate_submission(max_n: int = 200, output_path: Optional[Path] = None):
    """Generate full submission using parallel hybrid pipeline."""
    print(f"Generating Submission (n=1 to {max_n})")
    print("=" * 60)

    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        return

    if output_path is None:
        output_path = Path(__file__).parent.parent / "submission_gen108.csv"

    config = ParallelConfig(
        n_workers=6,
        rust_runs_per_worker=2,
        sa_iterations=8000,
        sa_restarts=2,
        polish_iterations=1500,
    )

    total_start = time.perf_counter()
    all_results = []

    # Write CSV header
    with open(output_path, 'w') as f:
        f.write("n,tree_id,x,y,rotation\n")

    for n in range(1, max_n + 1):
        start_n = time.perf_counter()

        best_configs, best_side, all_sides = parallel_refine(n, config, verbose=False)

        elapsed_n = time.perf_counter() - start_n

        if best_configs is not None:
            all_results.append((n, best_side))

            # Append to CSV
            with open(output_path, 'a') as f:
                for i in range(n):
                    x, y, angle = best_configs[i]
                    f.write(f"{n},{i+1},{x:.9f},{y:.9f},{angle:.9f}\n")

            print(f"n={n:>3}: side={best_side:.4f} (range: {min(all_sides):.4f}-{max(all_sides):.4f}) [{elapsed_n:.1f}s]")
        else:
            print(f"n={n:>3}: FAILED")

        # Progress update
        if n % 10 == 0:
            elapsed = time.perf_counter() - total_start
            eta = elapsed / n * (max_n - n)
            print(f"  Progress: {n}/{max_n} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_elapsed = time.perf_counter() - total_start

    # Calculate score
    if all_results:
        score = sum(side**2 / n for n, side in all_results)
        print(f"\n{'='*60}")
        print(f"Submission complete: {output_path}")
        print(f"Total score: {score:.4f}")
        print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/max_n:.1f}s per n)")


def main():
    parser = argparse.ArgumentParser(
        description="Gen108 Rust-Python Hybrid Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--n', type=int,
        help='Run pipeline for single n value'
    )
    parser.add_argument(
        '--max-n', type=int,
        help='Generate full submission up to this n'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Use parallel refinement'
    )
    parser.add_argument(
        '--workers', type=int, default=6,
        help='Number of parallel workers (default: 6)'
    )
    parser.add_argument(
        '--compare-rust', action='store_true',
        help='Compare hybrid with Rust-only baseline'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output path for submission CSV'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    if args.compare_rust:
        compare_with_rust()
    elif args.max_n:
        output_path = Path(args.output) if args.output else None
        generate_submission(args.max_n, output_path)
    elif args.n:
        run_single_n(args.n, args.parallel, args.workers, verbose)
    else:
        # Default: quick benchmark
        run_benchmark([20, 50], verbose)


if __name__ == '__main__':
    main()
