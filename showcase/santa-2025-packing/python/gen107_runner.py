#!/usr/bin/env python3
"""
Gen107: Main runner for hybrid GPU/CPU optimization.

Compares Python pipeline with Rust evolved algorithm
and generates submissions if competitive.
"""

import numpy as np
import json
import subprocess
import time
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
import argparse

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_collision import HybridCollisionChecker
from multi_stage_opt import MultiStageOptimizer, PipelineConfig, create_initial_config
from polygon_collision import transform_trees_batch, polygons_overlap


def run_rust_benchmark(n_range: Tuple[int, int], runs: int = 1) -> Dict[int, float]:
    """
    Run Rust evolved benchmark and return side lengths per n.
    """
    rust_dir = Path(__file__).parent.parent / "rust"

    # Build Rust
    print("Building Rust...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=rust_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Rust build failed: {result.stderr}")
        return {}

    # Run benchmark
    print(f"Running Rust benchmark (n={n_range[0]}-{n_range[1]}, {runs} runs)...")
    result = subprocess.run(
        ["./target/release/benchmark", str(n_range[1]), str(runs)],
        cwd=rust_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Rust benchmark failed: {result.stderr}")
        return {}

    # Parse output (format: "  n= 10: score=5.7982, side=2.2812")
    results = {}
    for line in result.stdout.split('\n'):
        if 'side=' in line and 'n=' in line:
            # Parse "  n= 10: score=5.7982, side=2.2812"
            parts = line.strip().split()
            n = None
            side = None
            for i, part in enumerate(parts):
                if part.startswith('n='):
                    n = int(part.split('=')[1].rstrip(':'))
                elif part.startswith('side='):
                    side = float(part.split('=')[1])
            if n is not None and side is not None:
                results[n] = side

    return results


def run_python_optimization(n: int, config: PipelineConfig) -> Tuple[float, int, float]:
    """
    Run Python optimization for a single n.

    Returns: (side_length, overlaps, time_seconds)
    """
    optimizer = MultiStageOptimizer()

    # Create initial config
    initial = create_initial_config(n, box_size=n * 0.4)

    start = time.perf_counter()
    final_configs, side_length, overlaps = optimizer.optimize(
        initial, config, verbose=False
    )
    elapsed = time.perf_counter() - start

    return side_length, overlaps, elapsed


def benchmark_comparison(n_values: List[int], quick: bool = True):
    """
    Compare Python pipeline with Rust for given n values.
    """
    print("=" * 70)
    print("Gen107 Benchmark: Python Hybrid vs Rust Evolved")
    print("=" * 70)

    # Pipeline config
    if quick:
        config = PipelineConfig(
            rotation_angles=12,
            compaction_passes=5,
            squeeze_passes=3,
            sa_iterations=5000,
            sa_restarts=2,
            polish_iterations=1000,
        )
        print("Mode: Quick (5k SA iters, 2 restarts)")
    else:
        config = PipelineConfig(
            rotation_angles=36,
            compaction_passes=15,
            squeeze_passes=5,
            sa_iterations=40000,
            sa_restarts=6,
            polish_iterations=5000,
        )
        print("Mode: Full (40k SA iters, 6 restarts)")

    # Run Rust benchmark
    n_max = max(n_values)
    rust_results = run_rust_benchmark((1, n_max), runs=1)

    if not rust_results:
        print("Warning: Could not get Rust results, continuing with Python only")

    # Run Python for each n
    results = []

    print(f"\n{'n':>5} | {'Rust':>8} | {'Python':>8} | {'Diff':>8} | {'Time':>6}")
    print("-" * 50)

    for n in n_values:
        rust_side = rust_results.get(n, float('nan'))

        py_side, py_overlaps, py_time = run_python_optimization(n, config)

        if py_overlaps > 0:
            py_side_str = f"{py_side:.3f}*"  # Mark invalid
        else:
            py_side_str = f"{py_side:.3f}"

        if np.isfinite(rust_side):
            diff = py_side - rust_side
            diff_str = f"{diff:+.3f}"
        else:
            diff_str = "N/A"

        print(f"{n:>5} | {rust_side:>8.3f} | {py_side_str:>8} | {diff_str:>8} | {py_time:>5.1f}s")

        results.append({
            'n': n,
            'rust': rust_side,
            'python': py_side,
            'overlaps': py_overlaps,
            'time': py_time,
        })

    # Summary
    print("\nSummary:")
    valid_results = [r for r in results if r['overlaps'] == 0 and np.isfinite(r['rust'])]
    if valid_results:
        rust_total = sum(r['rust'] ** 2 / r['n'] for r in valid_results)
        python_total = sum(r['python'] ** 2 / r['n'] for r in valid_results)
        print(f"  Rust score:   {rust_total:.3f}")
        print(f"  Python score: {python_total:.3f}")
        print(f"  Difference:   {python_total - rust_total:+.3f} ({100*(python_total/rust_total - 1):+.1f}%)")

    return results


def generate_submission(n_range: Tuple[int, int], output_path: Path, config: PipelineConfig):
    """
    Generate a submission CSV for all n in range.
    """
    print(f"Generating submission for n={n_range[0]}-{n_range[1]}")

    optimizer = MultiStageOptimizer()
    all_results = []

    total_score = 0.0
    total_time = 0.0

    for n in range(n_range[0], n_range[1] + 1):
        print(f"\nOptimizing n={n}...", end='', flush=True)

        initial = create_initial_config(n, box_size=n * 0.4)

        start = time.perf_counter()
        final_configs, side_length, overlaps = optimizer.optimize(
            initial, config, verbose=False
        )
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if overlaps > 0:
            print(f" INVALID ({overlaps} overlaps)")
            continue

        score_contrib = side_length ** 2 / n
        total_score += score_contrib

        print(f" side={side_length:.3f}, score={score_contrib:.3f}, time={elapsed:.1f}s")

        # Save result
        for i, (x, y, angle) in enumerate(final_configs):
            all_results.append({
                'n': n,
                'tree_id': i,
                'x': x,
                'y': y,
                'angle_deg': angle,
            })

    # Write CSV
    with open(output_path, 'w') as f:
        f.write("id,x,y,rotation\n")
        result_idx = 0
        for n in range(n_range[0], n_range[1] + 1):
            for tree_id in range(n):
                r = all_results[result_idx]
                f.write(f"{n}_{tree_id},{r['x']:.9f},{r['y']:.9f},{r['angle_deg']:.9f}\n")
                result_idx += 1

    print(f"\nSubmission written to: {output_path}")
    print(f"Total score: {total_score:.3f}")
    print(f"Total time: {total_time / 60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Gen107 Hybrid Optimization Runner')
    parser.add_argument('--mode', choices=['benchmark', 'submit', 'single'], default='benchmark')
    parser.add_argument('--n', type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--quick', action='store_true', help='Use quick settings')
    parser.add_argument('--output', type=str, default='submission_gen107.csv')
    args = parser.parse_args()

    if args.mode == 'benchmark':
        benchmark_comparison(args.n, quick=args.quick)

    elif args.mode == 'single':
        # Single n optimization with verbose output
        n = args.n[0] if args.n else 20

        config = PipelineConfig(
            sa_iterations=10000 if args.quick else 40000,
            sa_restarts=2 if args.quick else 6,
        )

        optimizer = MultiStageOptimizer()
        initial = create_initial_config(n, box_size=n * 0.4)

        print(f"Optimizing n={n} with {'quick' if args.quick else 'full'} settings...")
        final_configs, side_length, overlaps = optimizer.optimize(
            initial, config, verbose=True
        )

        print(f"\nFinal: side={side_length:.3f}, overlaps={overlaps}")

    elif args.mode == 'submit':
        n_range = (1, 200)
        config = PipelineConfig(
            sa_iterations=10000 if args.quick else 40000,
            sa_restarts=2 if args.quick else 6,
        )
        output_path = Path(__file__).parent.parent / args.output
        generate_submission(n_range, output_path, config)


if __name__ == '__main__':
    main()
