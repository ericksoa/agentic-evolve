#!/usr/bin/env python3
"""
Gen106 Benchmark - Full Scale Testing

Test GPU-accelerated optimization at competition scale (n=200).
"""

import torch
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Tuple
import time

from gpu_primitives import (
    TreeTensor,
    evaluate_configs,
    gpu_count_overlaps,
    get_device,
    TREE_VERTICES_NP,
)
from parallel_sa import ParallelSA, HybridOptimizer


def generate_grid_config(n: int, box_size: float, add_noise: float = 0.0) -> np.ndarray:
    """Generate a grid-based initial configuration."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    spacing = box_size / max(rows, cols)

    config = np.zeros((n, 3), dtype=np.float32)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col - (cols - 1) / 2) * spacing
            y = (row - (rows - 1) / 2) * spacing
            angle = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            config[idx] = [x, y, angle]
            idx += 1

    # Add noise
    if add_noise > 0:
        config[:, :2] += np.random.randn(n, 2).astype(np.float32) * add_noise
        config[:, 2] += np.random.randn(n).astype(np.float32) * 5.0
        config[:, 2] = config[:, 2] % 360.0

    return config


def estimate_box_size(n: int) -> float:
    """Estimate initial box size for n trees."""
    tree_area = 0.35  # Approximate tree area
    total_area = n * tree_area
    # For grid init, use 25% efficiency (very sparse to avoid overlaps)
    side = np.sqrt(total_area / 0.25)
    return side


def run_parallel_sa_benchmark(
    n: int = 200,
    n_chains: int = 64,
    n_iterations: int = 10000,
) -> Tuple[torch.Tensor, float]:
    """Run parallel SA at full scale with compression."""
    print(f"\n{'='*60}")
    print(f"Parallel SA Benchmark (with compression)")
    print(f"  n_trees: {n}")
    print(f"  n_chains: {n_chains}")
    print(f"  iterations: {n_iterations}")
    print(f"{'='*60}\n")

    device = get_device()
    print(f"Device: {device}")

    # Generate initial configs with spread to avoid overlaps
    box_size = estimate_box_size(n) * 1.3
    print(f"Initial box size: {box_size:.2f}")

    initial_configs = [
        generate_grid_config(n, box_size, add_noise=0.1 * i / n_chains)
        for i in range(n_chains)
    ]
    initial_tensor = torch.tensor(
        np.stack(initial_configs),
        dtype=torch.float32,
        device=device
    )

    sa = ParallelSA(n_trees=n, n_chains=n_chains, device=device)
    sa.initialize(initial_tensor)

    start_time = time.perf_counter()

    # Run SA with periodic compression
    best = sa.run(
        n_iterations=n_iterations,
        initial_temp=2.0,
        final_temp=0.001,
        move_scale=0.15,
        angle_scale=20.0,
        overlap_penalty=500.0,
        verbose=True,
        log_interval=1000,
        compression_interval=200,  # Compress every 200 iterations
        compression_strength=0.01,  # 1% towards center each time
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Iterations per second: {n_iterations / elapsed:.0f}")

    return best, sa.best_side_length


def run_compress_refine(
    n: int = 200,
    n_chains: int = 64,
    n_rounds: int = 50,
    sa_iters_per_round: int = 500,
) -> Tuple[torch.Tensor, float]:
    """
    Iterative compress-and-refine approach.

    Each round:
    1. Compress all trees 2% towards center
    2. Run SA to fix overlaps
    3. Repeat until target side length or stuck
    """
    print(f"\n{'='*60}")
    print(f"Compress-and-Refine Optimization")
    print(f"  n_trees: {n}")
    print(f"  n_chains: {n_chains}")
    print(f"  rounds: {n_rounds}")
    print(f"  SA iters/round: {sa_iters_per_round}")
    print(f"{'='*60}\n")

    device = get_device()
    print(f"Device: {device}")

    # Start sparse
    box_size = estimate_box_size(n) * 1.3
    print(f"Initial box size: {box_size:.2f}")

    initial_configs = [
        generate_grid_config(n, box_size, add_noise=0.05)
        for _ in range(n_chains)
    ]
    initial_tensor = torch.tensor(
        np.stack(initial_configs),
        dtype=torch.float32,
        device=device
    )

    sa = ParallelSA(n_trees=n, n_chains=n_chains, device=device)
    sa.initialize(initial_tensor)

    start_time = time.perf_counter()

    for round_idx in range(n_rounds):
        # Get current stats
        side_lengths = sa._get_side_lengths(sa.configs)
        overlaps = sa._get_overlaps(sa.configs)
        best_side = side_lengths.min().item()
        best_overlaps = overlaps[side_lengths.argmin()].item()

        # Adaptive compression: compress more when no overlaps, less when many
        mean_overlaps = overlaps.float().mean().item()
        if mean_overlaps < 1:
            compress_strength = 0.03  # Aggressive when no overlaps
        elif mean_overlaps < 5:
            compress_strength = 0.015  # Moderate
        else:
            compress_strength = 0.005  # Gentle when many overlaps

        # Compress
        sa.compress_towards_center(compress_strength)

        # SA to fix overlaps
        # Higher temp when more overlaps
        temp = 0.5 + 0.1 * mean_overlaps
        sa.run(
            n_iterations=sa_iters_per_round,
            initial_temp=temp,
            final_temp=temp * 0.1,
            move_scale=0.1,
            angle_scale=15.0,
            overlap_penalty=500.0,
            verbose=False,
        )

        # Log progress
        elapsed = time.perf_counter() - start_time
        print(f"Round {round_idx+1:3d}: side={best_side:.4f} "
              f"overlaps={best_overlaps} compress={compress_strength:.3f} "
              f"[{elapsed:.1f}s]")

        # Check if converged
        if best_side < 10.0 and best_overlaps == 0:
            print(f"\n✓ Achieved target side length with no overlaps!")
            break

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    return sa.best_config, sa.best_side_length


def run_hybrid_benchmark(
    n: int = 200,
    pop_size: int = 64,
    n_generations: int = 20,
    sa_iterations: int = 2000,
) -> Tuple[torch.Tensor, float]:
    """Run hybrid optimizer at full scale."""
    print(f"\n{'='*60}")
    print(f"Hybrid Optimizer Benchmark")
    print(f"  n_trees: {n}")
    print(f"  pop_size: {pop_size}")
    print(f"  generations: {n_generations}")
    print(f"  SA iterations/gen: {sa_iterations}")
    print(f"{'='*60}\n")

    device = get_device()
    print(f"Device: {device}")

    box_size = estimate_box_size(n) * 1.3
    print(f"Initial box size: {box_size:.2f}")

    initial_configs = [
        generate_grid_config(n, box_size, add_noise=0.1 * i / pop_size)
        for i in range(pop_size)
    ]
    initial_tensor = torch.tensor(
        np.stack(initial_configs),
        dtype=torch.float32,
        device=device
    )

    start_time = time.perf_counter()

    hybrid = HybridOptimizer(n_trees=n, pop_size=pop_size, device=device)
    best = hybrid.run(
        initial_configs=initial_tensor,
        n_generations=n_generations,
        sa_iterations_per_gen=sa_iterations,
        initial_temp=0.8,
        final_temp=0.01,
        verbose=True,
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nTotal time: {elapsed:.1f}s")

    return best, hybrid.best_side_length


def validate_solution(
    config: torch.Tensor,
    device: torch.device
) -> Tuple[float, int]:
    """Validate a solution and return side_length and overlap count."""
    tree_tensor = TreeTensor(device)
    config_batch = config.unsqueeze(0)

    _, bbox, overlaps, side_lengths = evaluate_configs(tree_tensor, config_batch)
    overlap_count = gpu_count_overlaps(overlaps).item()
    side_length = side_lengths.item()

    return side_length, int(overlap_count)


def export_to_csv(
    config: torch.Tensor,
    output_path: Path,
    n: int
):
    """Export configuration to CSV format for submission."""
    config_np = config.cpu().numpy()

    # Compute vertices for each tree
    tree_tensor = TreeTensor(torch.device('cpu'))
    config_batch = torch.tensor(config_np, dtype=torch.float32).unsqueeze(0)
    transformed, _, _, _ = evaluate_configs(tree_tensor, config_batch)
    vertices = transformed[0].numpy()  # (n_trees, 15, 2)

    # CSV format: tree_id, x_0, y_0, x_1, y_1, ..., x_14, y_14
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        cols = ['tree_id'] + [f'{c}_{i}' for i in range(15) for c in ['x', 'y']]
        writer.writerow(cols)

        for tree_id in range(n):
            row = [tree_id]
            for v in range(15):
                row.extend([vertices[tree_id, v, 0], vertices[tree_id, v, 1]])
            writer.writerow(row)

    print(f"Exported to {output_path}")


def main():
    import sys

    n = 200

    if len(sys.argv) > 1:
        if sys.argv[1] == 'hybrid':
            best, side_length = run_hybrid_benchmark(
                n=n,
                pop_size=64,
                n_generations=30,
                sa_iterations=3000,
            )
        elif sys.argv[1] == 'quick':
            # Quick test
            best, side_length = run_parallel_sa_benchmark(
                n=n,
                n_chains=32,
                n_iterations=2000,
            )
        elif sys.argv[1] == 'compress':
            # Compress-and-refine
            best, side_length = run_compress_refine(
                n=n,
                n_chains=64,
                n_rounds=100,
                sa_iters_per_round=500,
            )
        else:
            print(f"Unknown mode: {sys.argv[1]}")
            return
    else:
        # Default: parallel SA
        best, side_length = run_parallel_sa_benchmark(
            n=n,
            n_chains=64,
            n_iterations=20000,
        )

    # Validate
    device = get_device()
    final_side, final_overlaps = validate_solution(best, device)

    print(f"\n{'='*60}")
    print(f"Final Validation")
    print(f"{'='*60}")
    print(f"Side length: {final_side:.4f}")
    print(f"Overlaps (bbox): {final_overlaps}")

    # Score contribution for n=200
    score_contrib = final_side ** 2 / n
    print(f"Score contribution: {score_contrib:.4f}")

    # Compare to best known
    best_known = 86.17  # Gen103 + best-of-20
    print(f"\nBest known score: {best_known:.2f}")
    print(f"This run contribution: {score_contrib:.2f}")

    if final_overlaps == 0:
        print("\n✓ No bbox overlaps - potentially valid solution")
    else:
        print(f"\n✗ {final_overlaps} bbox overlaps - NOT valid")


if __name__ == '__main__':
    main()
