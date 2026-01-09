#!/usr/bin/env python3
"""
Lightning.ai runner for GPU SA optimization.

Usage on lightning.ai studio (L40S GPU):
1. Create a new studio with L40S GPU
2. Clone the repo
3. Install dependencies: pip install torch numpy
4. Run: python python/gpu/lightning_run.py --n-range 1-50 --chains 500 --iterations 10000

This script runs GPU SA across multiple n values and compares with current best.
"""

import torch
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gpu_sa import (
    GPUSimulatedAnnealing, SAConfig, get_device,
    validate_packing, TREE_VERTICES, load_best_from_csv,
    compute_side_from_solution
)


def load_current_best(csv_path: str) -> Dict[int, float]:
    """Load current best side lengths from submission CSV."""
    current_best = {}

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found")
        return current_best

    # Group trees by n
    trees_by_n: Dict[int, List[Tuple[float, float, float]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'][1:])  # Remove 's' prefix
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            if n not in trees_by_n:
                trees_by_n[n] = []
            trees_by_n[n].append((x, y, deg))

    # Compute side lengths
    for n, trees in trees_by_n.items():
        if not trees:
            continue

        # Get all vertices
        all_xs = []
        all_ys = []
        for x, y, deg in trees:
            import math
            rad = math.radians(deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            for vx, vy in TREE_VERTICES.tolist():
                rx = vx * cos_a - vy * sin_a + x
                ry = vx * sin_a + vy * cos_a + y
                all_xs.append(rx)
                all_ys.append(ry)

        width = max(all_xs) - min(all_xs)
        height = max(all_ys) - min(all_ys)
        current_best[n] = max(width, height)

    return current_best


def run_gpu_sa_for_n(n: int, chains: int, iterations: int, device,
                     best_solutions: Dict[int, List[Tuple[float, float, float]]] = None) -> Tuple[float, bool]:
    """Run GPU SA for a single n value."""
    # Use smaller perturbations when starting from best solutions (refinement mode)
    if best_solutions and n in best_solutions:
        config = SAConfig(
            iterations=iterations,
            initial_temp=0.3,  # Lower for refinement
            cooling_rate=0.999 if iterations > 5000 else 0.998,
            translation_small=0.02,
            translation_large=0.05,
            rotation_max=15.0,
            overlap_margin=0.01
        )
    else:
        # Global search: larger moves, higher temperature
        config = SAConfig(
            iterations=iterations,
            initial_temp=2.0,  # Higher for exploration
            cooling_rate=0.9995 if iterations > 5000 else 0.999,
            translation_small=0.1,
            translation_large=0.3,
            rotation_max=45.0,
            overlap_margin=0.01
        )

    sa = GPUSimulatedAnnealing(n, chains, config, device, best_solutions)
    best_pos, best_ang, best_sides = sa.run(
        verbose=False,
        use_best_init=(best_solutions is not None and n in best_solutions),
        validate_final=True  # Filter invalid solutions
    )

    # Find best valid solution (invalid chains have inf side length)
    valid_mask = best_sides < float('inf')
    if not valid_mask.any():
        return float('inf'), False

    best_idx = best_sides.argmin().item()
    best_side = best_sides[best_idx].item()

    return best_side, True  # Already validated


def main():
    parser = argparse.ArgumentParser(description='Lightning.ai GPU SA runner')
    parser.add_argument('--n-range', type=str, default='1-20',
                        help='Range of n values (e.g., "1-20" or "5,10,15")')
    parser.add_argument('--chains', type=int, default=200,
                        help='Number of parallel SA chains')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='SA iterations per chain')
    parser.add_argument('--compare', type=str, default='submission_best.csv',
                        help='CSV to compare against')
    parser.add_argument('--output', type=str, default='gpu_sa_results.json',
                        help='Output JSON file')
    parser.add_argument('--mode', type=str, default='refine', choices=['refine', 'global'],
                        help='Search mode: "refine" starts from best, "global" starts fresh')
    args = parser.parse_args()

    # Parse n range
    if '-' in args.n_range:
        start, end = map(int, args.n_range.split('-'))
        n_values = list(range(start, end + 1))
    else:
        n_values = [int(x) for x in args.n_range.split(',')]

    # Setup
    device = get_device()
    print(f"{'='*60}")
    print(f"GPU SA Optimization - Lightning.ai")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"N values: {n_values[0]}-{n_values[-1]} ({len(n_values)} total)")
    print(f"Chains: {args.chains}")
    print(f"Iterations: {args.iterations}")
    print()

    # Load current best side lengths for comparison
    compare_path = Path(__file__).parent.parent.parent / args.compare
    current_best = load_current_best(str(compare_path))
    print(f"Loaded {len(current_best)} current best values from {args.compare}")

    # Load full solutions for initialization (only used in refine mode)
    best_solutions = None
    if args.mode == 'refine':
        best_solutions = load_best_from_csv(str(compare_path))
        print(f"Loaded {len(best_solutions)} full solutions for refinement")
    else:
        print("Global search mode - starting from random placements")
    print()

    # Run optimization
    results = {}
    total_improvement = 0.0
    improved_count = 0

    start_total = time.time()

    for n in n_values:
        if args.mode == 'refine' and best_solutions and n in best_solutions:
            init_type = "refine"
        else:
            init_type = "global"
        print(f"n={n:3d} ({init_type}): ", end="", flush=True)

        start = time.time()
        gpu_side, is_valid = run_gpu_sa_for_n(n, args.chains, args.iterations, device,
                                               best_solutions if args.mode == 'refine' else None)
        elapsed = time.time() - start

        current_side = current_best.get(n, float('inf'))
        improvement = (current_side**2 - gpu_side**2) / n if is_valid else 0

        results[n] = {
            'gpu_side': gpu_side,
            'gpu_score': gpu_side**2 / n,
            'current_side': current_side,
            'current_score': current_side**2 / n,
            'improvement': improvement,
            'valid': is_valid,
            'time': elapsed
        }

        status = "VALID" if is_valid else "INVALID"
        if is_valid and improvement > 0:
            improved_count += 1
            total_improvement += improvement
            print(f"side={gpu_side:.4f} ({status}) *** IMPROVEMENT +{improvement:.4f} *** [{elapsed:.1f}s]")
        else:
            diff = gpu_side - current_side
            print(f"side={gpu_side:.4f} ({status}) vs {current_side:.4f} (diff={diff:+.4f}) [{elapsed:.1f}s]")

    total_time = time.time() - start_total

    # Summary
    print()
    print(f"{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"N values improved: {improved_count}/{len(n_values)}")
    print(f"Total score improvement: {total_improvement:.6f}")

    # Save results
    output = {
        'config': {
            'n_range': args.n_range,
            'chains': args.chains,
            'iterations': args.iterations,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
        },
        'summary': {
            'total_time': total_time,
            'improved_count': improved_count,
            'total_improvement': total_improvement
        },
        'results': {str(k): v for k, v in results.items()}
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
