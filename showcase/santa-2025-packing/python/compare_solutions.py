#!/usr/bin/env python3
"""
Compare Python optimizer results with theoretical and empirical benchmarks.
"""

import json
import math

# Known theoretical lower bounds for tree packing
# side_min = sqrt(n * tree_area / fill_efficiency)
# Tree area is approximately 0.35 (calculated from polygon)

TREE_AREA = 0.35  # Approximate area of tree polygon


def theoretical_lower_bound(n: int) -> float:
    """
    Theoretical lower bound assuming perfect packing.
    Perfect 2D packing can achieve ~90% fill ratio for convex shapes.
    For our tree shape, ~70-80% is more realistic.
    """
    # Assuming 70% fill efficiency for the irregular tree shape
    fill_efficiency = 0.70
    total_area = n * TREE_AREA
    side = math.sqrt(total_area / fill_efficiency)
    return side


def score_contribution(n: int, side: float) -> float:
    """Calculate score contribution for a single n."""
    return (side ** 2) / n


def analyze_results(results_file: str = "optimized_packings.json"):
    """Analyze optimization results."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return

    print("Santa 2025 Tree Packing - Solution Analysis")
    print("=" * 60)
    print()

    total_score = 0.0
    print(f"{'n':>4} {'Side':>8} {'Score':>10} {'Method':>15} {'vs Lower':>10}")
    print("-" * 60)

    for n_str, data in sorted(results.items(), key=lambda x: int(x[0])):
        n = int(n_str)
        side = data['side']
        method = data['method']
        score = score_contribution(n, side)
        total_score += score

        lb = theoretical_lower_bound(n)
        ratio = side / lb

        print(f"{n:4d} {side:8.4f} {score:10.4f} {method:>15} {ratio:8.2f}x")

    print("-" * 60)
    print(f"Total score (n=1..{max(int(k) for k in results.keys())}): {total_score:.4f}")
    print()

    # Extrapolate to full competition
    max_n = max(int(k) for k in results.keys())
    if max_n < 200:
        # Rough extrapolation based on observed scaling
        # Score grows roughly as n^0.5 for optimal packing
        avg_score_per_n = total_score / max_n
        estimated_full = total_score + avg_score_per_n * (200 - max_n) * 1.2
        print(f"Rough extrapolation to n=200: ~{estimated_full:.1f}")
        print("(Note: This is a very rough estimate)")

    # Compare with Rust champion
    print()
    print("Known benchmarks:")
    print(f"  Rust champion (Gen91b) for n=1-200: ~87-88")
    print(f"  Leaderboard #1 (Rafbill): 69.99")
    print(f"  Gap to close: {88 - 70:.0f} points (~26%)")


if __name__ == '__main__':
    analyze_results()
