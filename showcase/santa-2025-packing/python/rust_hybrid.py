#!/usr/bin/env python3
"""
Gen108: Rust-Python Hybrid Pipeline.

Key insight: Rust greedy placement (~3.2 side for n=20) is much better than
Python grid initialization (7.5). This module loads Rust packings and applies
Python SA refinement.

Pipeline:
1. Rust: Generate packing with evolved greedy algorithm
2. Python: Load JSON export
3. Python: Apply light SA refinement (skip heavy initialization)
4. Python: Output improved packing
"""

import json
import subprocess
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

from hybrid_collision import HybridCollisionChecker
from gpu_sa import HybridSA, SAConfig
from multi_stage_opt import MultiStageOptimizer, PipelineConfig


# Path to Rust binary
RUST_DIR = Path(__file__).parent.parent / "rust"
EXPORT_BIN = RUST_DIR / "target/release/export_packing"


@dataclass
class RustHybridConfig:
    """Configuration for Rust-Python hybrid pipeline."""
    # Rust generation
    rust_num_runs: int = 1  # Number of Rust runs to pick best from

    # Python SA refinement - conservative settings since Rust is already good
    sa_iterations: int = 10000
    sa_restarts: int = 3
    sa_initial_temp: float = 0.5  # Lower temp - fewer large jumps
    sa_final_temp: float = 0.001  # Lower final for fine tuning

    # Post-refinement
    polish_iterations: int = 2000
    squeeze_enabled: bool = True
    squeeze_passes: int = 3


def load_rust_packing(json_path: Path) -> Tuple[np.ndarray, float]:
    """
    Load packing from Rust JSON export.

    Args:
        json_path: Path to JSON file from export_packing

    Returns:
        (configs, side_length) where configs is (n, 3) array of [x, y, angle_deg]
    """
    with open(json_path) as f:
        data = json.load(f)

    n = data['n']
    side_length = data['side_length']

    configs = np.zeros((n, 3), dtype=np.float64)
    for i, tree in enumerate(data['trees']):
        configs[i, 0] = tree['x']
        configs[i, 1] = tree['y']
        configs[i, 2] = tree['angle_deg']

    return configs, side_length


def generate_rust_packing(n: int, num_runs: int = 1, output_dir: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    """
    Generate packing using Rust evolved algorithm.

    Args:
        n: Number of trees
        num_runs: Run algorithm multiple times and pick best
        output_dir: Directory for temp JSON files (default: /tmp)

    Returns:
        (configs, side_length)
    """
    if not EXPORT_BIN.exists():
        raise RuntimeError(f"Rust binary not found: {EXPORT_BIN}\n"
                          f"Run: cd {RUST_DIR} && cargo build --release --bin export_packing")

    output_dir = output_dir or Path("/tmp")
    json_path = output_dir / f"rust_packing_n{n}.json"

    # Run Rust export
    cmd = [str(EXPORT_BIN), str(n), str(json_path), str(num_runs)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Rust export failed:\n{result.stderr}")

    return load_rust_packing(json_path)


def refine_rust_packing(
    configs: np.ndarray,
    rust_side: float,
    config: Optional[RustHybridConfig] = None,
    checker: Optional[HybridCollisionChecker] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Apply Python SA refinement to Rust packing.

    Args:
        configs: (n, 3) array from Rust
        rust_side: Original side length from Rust
        config: Refinement configuration
        checker: Collision checker (will create if None)
        verbose: Print progress

    Returns:
        (refined_configs, final_side, n_overlaps)
    """
    if config is None:
        config = RustHybridConfig()

    if checker is None:
        checker = HybridCollisionChecker()

    n = configs.shape[0]

    # Initial validation
    n_overlaps, initial_side = checker.check_config_overlaps(configs)

    if verbose:
        print(f"Loaded Rust packing: n={n}, side={rust_side:.4f}")
        print(f"Python verification: side={initial_side:.4f}, overlaps={n_overlaps}")

        if n_overlaps > 0:
            print(f"  WARNING: Rust packing has {n_overlaps} overlaps in Python checker")

    # SA refinement - single pass (no restarts) since Rust solution is tight
    # Restarts perturb the config which creates overlaps
    sa = HybridSA(checker)
    sa_config = SAConfig(
        iterations=config.sa_iterations * config.sa_restarts,  # Longer single run
        initial_temp=config.sa_initial_temp,
        final_temp=config.sa_final_temp,
        restarts=1,  # No restarts - perturbation creates overlaps
        position_scale=0.03,  # Smaller scale - Rust packing is already good
        angle_scale=5.0,      # Smaller rotations - preserve Rust's choices
    )

    if verbose:
        print(f"\nApplying SA refinement ({config.sa_iterations * config.sa_restarts} iters, no restarts)")

    refined, score, overlaps = sa.optimize(
        configs, sa_config, verbose=verbose
    )

    # Post-SA squeeze
    if config.squeeze_enabled and overlaps == 0:
        if verbose:
            print(f"\nApplying squeeze ({config.squeeze_passes} passes)")

        optimizer = MultiStageOptimizer(checker)
        for _ in range(config.squeeze_passes):
            squeezed = optimizer.squeeze_toward_center(refined, factor=0.995, verbose=False)
            new_overlaps, new_side = checker.check_config_overlaps(squeezed)
            if new_overlaps == 0 and new_side < score:
                refined = squeezed
                score = new_side

    # Polish
    if config.polish_iterations > 0:
        if verbose:
            print(f"\nPolish ({config.polish_iterations} iters)")

        polish_config = SAConfig(
            iterations=config.polish_iterations,
            initial_temp=0.3,
            final_temp=0.001,
            restarts=1,
            position_scale=0.02,
            angle_scale=5.0,
        )
        refined, score, overlaps = sa.optimize(refined, polish_config, verbose=False)

    # Final check
    final_overlaps, final_side = checker.check_config_overlaps(refined)

    if verbose:
        improvement = (rust_side - final_side) / rust_side * 100
        print(f"\nResult: side={final_side:.4f}, overlaps={final_overlaps}")
        print(f"Improvement: {rust_side:.4f} -> {final_side:.4f} ({improvement:+.2f}%)")

    return refined, final_side, final_overlaps


def export_packing_to_csv(
    configs: np.ndarray,
    output_path: Path,
    n_start: int = 1,
) -> None:
    """
    Export packing to CSV format for Kaggle submission.

    Note: This exports a single n value. For full submission,
    use this for each n from 1 to 200.
    """
    n = configs.shape[0]

    # Build submission rows
    rows = []
    for i in range(n):
        row = f"{n_start + n - 1},{i + 1}," + ",".join(
            f"{configs[i, j]:.9f}" for j in range(3)
        )
        rows.append(row)

    with open(output_path, 'a') as f:
        for row in rows:
            f.write(row + "\n")


def hybrid_pipeline_single_n(
    n: int,
    config: Optional[RustHybridConfig] = None,
    checker: Optional[HybridCollisionChecker] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Run full Rust-Python hybrid pipeline for a single n.

    Args:
        n: Number of trees
        config: Pipeline configuration
        checker: Collision checker (will create if None)
        verbose: Print progress

    Returns:
        (configs, final_side, rust_side, improvement_pct)
    """
    if config is None:
        config = RustHybridConfig()

    if checker is None:
        checker = HybridCollisionChecker()

    # Generate Rust packing
    if verbose:
        print(f"\n{'='*60}")
        print(f"Hybrid Pipeline for n={n}")
        print("=" * 60)
        print(f"\nStep 1: Generate Rust packing (runs={config.rust_num_runs})")

    start_rust = time.perf_counter()
    rust_configs, rust_side = generate_rust_packing(n, config.rust_num_runs)
    rust_time = time.perf_counter() - start_rust

    if verbose:
        print(f"  Rust: side={rust_side:.4f} ({rust_time:.1f}s)")

    # Refine with Python
    if verbose:
        print(f"\nStep 2: Python SA refinement")

    start_python = time.perf_counter()
    refined, final_side, overlaps = refine_rust_packing(
        rust_configs, rust_side, config, checker, verbose
    )
    python_time = time.perf_counter() - start_python

    improvement = (rust_side - final_side) / rust_side * 100

    if verbose:
        print(f"\nPipeline Summary:")
        print(f"  Rust time:   {rust_time:.1f}s")
        print(f"  Python time: {python_time:.1f}s")
        print(f"  Total time:  {rust_time + python_time:.1f}s")
        print(f"  Rust side:   {rust_side:.4f}")
        print(f"  Final side:  {final_side:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Overlaps:    {overlaps}")

    return refined, final_side, rust_side, improvement


def benchmark_hybrid():
    """Benchmark hybrid pipeline on various n values."""
    print("Rust-Python Hybrid Pipeline Benchmark")
    print("=" * 60)

    # Check Rust binary exists
    if not EXPORT_BIN.exists():
        print(f"ERROR: Rust binary not found at {EXPORT_BIN}")
        print(f"Build with: cd {RUST_DIR} && cargo build --release --bin export_packing")
        return

    checker = HybridCollisionChecker()

    config = RustHybridConfig(
        rust_num_runs=3,
        sa_iterations=5000,
        sa_restarts=2,
        polish_iterations=1000,
    )

    results = []

    for n in [10, 20, 50]:
        refined, final_side, rust_side, improvement = hybrid_pipeline_single_n(
            n, config, checker, verbose=True
        )
        results.append((n, rust_side, final_side, improvement))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'n':>4} | {'Rust':>8} | {'Final':>8} | {'Improvement':>12}")
    print("-" * 40)
    for n, rust_side, final_side, improvement in results:
        print(f"{n:>4} | {rust_side:>8.4f} | {final_side:>8.4f} | {improvement:>+11.2f}%")


if __name__ == '__main__':
    benchmark_hybrid()
