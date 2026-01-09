#!/usr/bin/env python3
"""
Benchmark GPU SA vs baseline approaches.

Tests:
1. GPU SA speedup vs CPU
2. Solution quality across different chain counts
3. Comparison with current best solutions
"""

import torch
import time
import json
import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu.gpu_sa import GPUSimulatedAnnealing, SAConfig, get_device, validate_packing, TREE_VERTICES


def benchmark_speedup(n: int, iterations: int = 5000):
    """Benchmark GPU vs CPU speedup."""
    print(f"\n{'='*60}")
    print(f"Speedup Benchmark: n={n}, iterations={iterations}")
    print('='*60)

    device = get_device()
    config = SAConfig(iterations=iterations, initial_temp=0.5, cooling_rate=0.999)

    # GPU with many chains
    print("\nGPU (100 chains):")
    start = time.time()
    sa_gpu = GPUSimulatedAnnealing(n, 100, config, device)
    _, _, sides_gpu = sa_gpu.run(verbose=False)
    gpu_time = time.time() - start
    gpu_best = sides_gpu.min().item()
    print(f"  Time: {gpu_time:.2f}s")
    print(f"  Best: {gpu_best:.6f}")
    print(f"  Throughput: {100 * iterations / gpu_time:.0f} chain-iterations/s")

    # CPU single chain (simulate by using CPU device)
    print("\nCPU (1 chain):")
    start = time.time()
    sa_cpu = GPUSimulatedAnnealing(n, 1, config, torch.device('cpu'))
    _, _, sides_cpu = sa_cpu.run(verbose=False)
    cpu_time = time.time() - start
    cpu_best = sides_cpu.min().item()
    print(f"  Time: {cpu_time:.2f}s")
    print(f"  Best: {cpu_best:.6f}")
    print(f"  Throughput: {iterations / cpu_time:.0f} chain-iterations/s")

    # Speedup calculation
    # Effective speedup: how many CPU chains could we run in the same time?
    effective_speedup = (100 * iterations / gpu_time) / (iterations / cpu_time)
    print(f"\nEffective speedup: {effective_speedup:.1f}x")
    print(f"(100 GPU chains vs 1 CPU chain in same wall-clock time)")

    return {
        'n': n,
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'gpu_best': gpu_best,
        'cpu_best': cpu_best,
        'effective_speedup': effective_speedup
    }


def benchmark_chain_scaling(n: int, chain_counts: list = [10, 50, 100, 500, 1000]):
    """Test how solution quality scales with chain count."""
    print(f"\n{'='*60}")
    print(f"Chain Scaling Benchmark: n={n}")
    print('='*60)

    device = get_device()
    config = SAConfig(iterations=5000, initial_temp=0.5, cooling_rate=0.999)

    results = []
    for chains in chain_counts:
        print(f"\nChains: {chains}")
        start = time.time()
        sa = GPUSimulatedAnnealing(n, chains, config, device)
        _, _, sides = sa.run(verbose=False)
        elapsed = time.time() - start

        best = sides.min().item()
        mean = sides.mean().item()
        std = sides.std().item()

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Best: {best:.6f}")
        print(f"  Mean: {mean:.6f} ± {std:.6f}")

        results.append({
            'chains': chains,
            'time': elapsed,
            'best': best,
            'mean': mean,
            'std': std
        })

    return results


def compare_with_baseline(n_values: list = [5, 10, 20, 50]):
    """Compare GPU SA with current best solutions."""
    print(f"\n{'='*60}")
    print("Comparison with Current Best")
    print('='*60)

    # Load current best from submission
    submission_path = Path(__file__).parent.parent.parent / 'submission_best.csv'
    current_best = {}

    if submission_path.exists():
        import csv
        with open(submission_path) as f:
            reader = csv.DictReader(f)
            trees_by_n = {}
            for row in reader:
                n = int(row['id'].split('_')[0])
                if n not in trees_by_n:
                    trees_by_n[n] = []
                x = float(row['x'][1:])  # Remove 's' prefix
                y = float(row['y'][1:])
                trees_by_n[n].append((x, y))

            for n, trees in trees_by_n.items():
                # Compute side length from positions (approximate - ignoring angles)
                if trees:
                    xs = [t[0] for t in trees]
                    ys = [t[1] for t in trees]
                    width = max(xs) - min(xs) + 0.7  # Add tree width
                    height = max(ys) - min(ys) + 1.0  # Add tree height
                    current_best[n] = max(width, height)

    device = get_device()
    config = SAConfig(iterations=10000, initial_temp=0.5, cooling_rate=0.9995)

    results = []
    for n in n_values:
        print(f"\nn={n}:")

        # Run GPU SA
        sa = GPUSimulatedAnnealing(n, 200, config, device)
        best_pos, best_ang, sides = sa.run(verbose=False)

        best_idx = sides.argmin().item()
        gpu_best = sides[best_idx].item()

        # Validate
        is_valid, _ = validate_packing(
            best_pos[best_idx], best_ang[best_idx],
            TREE_VERTICES.to(device)
        )

        baseline = current_best.get(n, float('inf'))
        improvement = (baseline - gpu_best) / baseline * 100 if baseline < float('inf') else 0

        print(f"  GPU SA best: {gpu_best:.6f} (valid={is_valid})")
        print(f"  Current best: {baseline:.6f}")
        if improvement > 0:
            print(f"  *** IMPROVEMENT: {improvement:.2f}% ***")
        else:
            print(f"  Difference: {improvement:.2f}%")

        results.append({
            'n': n,
            'gpu_best': gpu_best,
            'current_best': baseline,
            'improvement_pct': improvement,
            'valid': is_valid
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU SA')
    parser.add_argument('--speedup', action='store_true', help='Run speedup benchmark')
    parser.add_argument('--scaling', action='store_true', help='Run chain scaling benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare with current best')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--n', type=int, default=10, help='Number of trees for speedup/scaling')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    if args.all or args.speedup:
        results['speedup'] = benchmark_speedup(args.n)

    if args.all or args.scaling:
        results['scaling'] = benchmark_chain_scaling(args.n)

    if args.all or args.compare:
        results['comparison'] = compare_with_baseline()

    if not any([args.speedup, args.scaling, args.compare, args.all]):
        # Default: quick comparison
        results['comparison'] = compare_with_baseline([5, 10, 20])

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
