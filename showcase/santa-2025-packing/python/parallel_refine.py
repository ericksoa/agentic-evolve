#!/usr/bin/env python3
"""
Gen108: Parallel Refinement Pipeline.

Key insight: Run multiple Rust+Python refinements in parallel and pick best.
Each worker generates a Rust packing and refines it independently.

This exploits:
1. Rust stochasticity - different runs produce different packings
2. Python SA exploration - different seeds find different local minima
3. Parallel execution - utilize all CPU cores
"""

import multiprocessing as mp
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from rust_hybrid import (
    RustHybridConfig,
    generate_rust_packing,
    refine_rust_packing,
    load_rust_packing,
    EXPORT_BIN,
    RUST_DIR,
)
from hybrid_collision import HybridCollisionChecker


@dataclass
class ParallelConfig:
    """Configuration for parallel refinement."""
    n_workers: int = 6
    rust_runs_per_worker: int = 1

    # SA settings (lighter since we have more workers)
    sa_iterations: int = 8000
    sa_restarts: int = 2
    polish_iterations: int = 1000
    squeeze_passes: int = 3


def worker_refine(args: Tuple) -> Tuple[int, np.ndarray, float, float]:
    """
    Worker function for parallel refinement.

    Args:
        args: (worker_id, n, rust_config, sa_config)

    Returns:
        (worker_id, configs, final_side, rust_side)
    """
    worker_id, n, config = args

    # Set random seed for reproducibility per worker
    np.random.seed(worker_id * 1000 + n)

    # Each worker gets its own collision checker
    checker = HybridCollisionChecker()

    rust_hybrid_config = RustHybridConfig(
        rust_num_runs=config.rust_runs_per_worker,
        sa_iterations=config.sa_iterations,
        sa_restarts=config.sa_restarts,
        polish_iterations=config.polish_iterations,
        squeeze_passes=config.squeeze_passes,
    )

    # Generate Rust packing
    try:
        rust_configs, rust_side = generate_rust_packing(
            n,
            num_runs=config.rust_runs_per_worker,
            output_dir=Path(f"/tmp/worker_{worker_id}"),
        )
    except Exception as e:
        print(f"Worker {worker_id}: Rust generation failed: {e}")
        return worker_id, None, float('inf'), float('inf')

    # Refine with Python
    try:
        refined, final_side, overlaps = refine_rust_packing(
            rust_configs, rust_side, rust_hybrid_config, checker, verbose=False
        )

        if overlaps > 0:
            # Invalid result
            return worker_id, None, float('inf'), rust_side

        return worker_id, refined, final_side, rust_side

    except Exception as e:
        print(f"Worker {worker_id}: Refinement failed: {e}")
        return worker_id, None, float('inf'), rust_side


def parallel_refine(
    n: int,
    config: Optional[ParallelConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run parallel refinement pipeline for a single n.

    Args:
        n: Number of trees
        config: Parallel configuration
        verbose: Print progress

    Returns:
        (best_configs, best_side, all_sides)
    """
    if config is None:
        config = ParallelConfig()

    # Ensure temp directories exist
    for i in range(config.n_workers):
        Path(f"/tmp/worker_{i}").mkdir(exist_ok=True)

    if verbose:
        print(f"\nParallel Refinement (n={n}, workers={config.n_workers})")
        print("=" * 60)

    start = time.perf_counter()

    # Prepare worker arguments
    worker_args = [(i, n, config) for i in range(config.n_workers)]

    # Run workers in parallel
    with mp.Pool(config.n_workers) as pool:
        results = pool.map(worker_refine, worker_args)

    elapsed = time.perf_counter() - start

    # Find best result
    best_worker = -1
    best_configs = None
    best_side = float('inf')
    all_sides = []
    rust_sides = []

    for worker_id, configs, final_side, rust_side in results:
        rust_sides.append(rust_side)
        if configs is not None:
            all_sides.append(final_side)
            if final_side < best_side:
                best_side = final_side
                best_configs = configs
                best_worker = worker_id

    if verbose:
        print(f"\nResults from {config.n_workers} workers:")
        for worker_id, configs, final_side, rust_side in results:
            status = "BEST" if worker_id == best_worker else "    "
            if configs is not None:
                improvement = (rust_side - final_side) / rust_side * 100
                print(f"  Worker {worker_id}: rust={rust_side:.4f} -> final={final_side:.4f} ({improvement:+.2f}%) {status}")
            else:
                print(f"  Worker {worker_id}: FAILED")

        print(f"\nBest: {best_side:.4f} (worker {best_worker})")
        print(f"Mean: {np.mean(all_sides):.4f}")
        print(f"Std:  {np.std(all_sides):.4f}")
        print(f"Time: {elapsed:.1f}s ({elapsed/config.n_workers:.1f}s per worker effective)")

    return best_configs, best_side, all_sides


def parallel_refine_all_n(
    max_n: int,
    config: Optional[ParallelConfig] = None,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, float]]:
    """
    Run parallel refinement for n=1 to max_n.

    Note: For each n, runs n_workers parallel refinements.

    Args:
        max_n: Maximum number of trees
        config: Parallel configuration
        verbose: Print progress

    Returns:
        List of (configs, side_length) for n=1 to max_n
    """
    if config is None:
        config = ParallelConfig()

    results = []
    total_start = time.perf_counter()

    for n in range(1, max_n + 1):
        configs, side, all_sides = parallel_refine(n, config, verbose=verbose)
        results.append((configs, side))

        if verbose and n % 10 == 0:
            elapsed = time.perf_counter() - total_start
            eta = elapsed / n * (max_n - n)
            print(f"\nProgress: {n}/{max_n} complete ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_elapsed = time.perf_counter() - total_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed n=1 to {max_n} in {total_elapsed:.1f}s")

        # Calculate score
        score = sum(side**2 / n for n, (_, side) in enumerate(results, start=1))
        print(f"Total score: {score:.4f}")

    return results


def benchmark_parallel():
    """Benchmark parallel refinement."""
    print("Parallel Refinement Benchmark")
    print("=" * 60)

    # Check Rust binary
    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        print(f"Build with: cd {RUST_DIR} && cargo build --release --bin export_packing")
        return

    config = ParallelConfig(
        n_workers=6,
        rust_runs_per_worker=2,
        sa_iterations=5000,
        sa_restarts=2,
        polish_iterations=1000,
    )

    # Test on a few n values
    for n in [20, 50]:
        best_configs, best_side, all_sides = parallel_refine(n, config, verbose=True)

        if best_configs is not None:
            # Verify no overlaps
            checker = HybridCollisionChecker()
            overlaps, side = checker.check_config_overlaps(best_configs)
            print(f"\nVerification: side={side:.4f}, overlaps={overlaps}")


if __name__ == '__main__':
    benchmark_parallel()
