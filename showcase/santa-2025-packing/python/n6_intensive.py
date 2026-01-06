#!/usr/bin/env python3
"""
Intensive n=6 optimization with many restarts and long runs.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from typing import List, Tuple, Optional
import json
import sys

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE = Polygon(TREE_VERTICES)


def get_tree_poly(x: float, y: float, angle: float) -> Polygon:
    rotated = affinity.rotate(TREE, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_side(trees: List[Tuple[float, float, float]]) -> float:
    all_coords = []
    for x, y, a in trees:
        poly = get_tree_poly(x, y, a)
        all_coords.extend(list(poly.exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap(trees: List[Tuple[float, float, float]], tol: float = 1e-9) -> bool:
    polys = [get_tree_poly(x, y, a) for x, y, a in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def circular_init(n: int, radius: float = 0.5, base_angle: float = 0) -> List[Tuple[float, float, float]]:
    trees = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        angle = (base_angle + i * (360 / n)) % 360
        trees.append((x, y, angle))
    return trees


def hexagonal_init(n: int = 6) -> List[Tuple[float, float, float]]:
    """Initialize in a hexagonal pattern."""
    # Center + 5 around (or 6 in hex pattern)
    positions = [
        (0, 0),
        (0.5, 0),
        (0.25, 0.433),
        (-0.25, 0.433),
        (-0.5, 0),
        (-0.25, -0.433),
        (0.25, -0.433),
    ][:n]
    trees = []
    for i, (x, y) in enumerate(positions):
        angle = i * 60
        trees.append((x, y, angle))
    return trees


def random_init(n: int, bound: float = 1.0) -> Optional[List[Tuple[float, float, float]]]:
    """Generate random valid configuration."""
    for _ in range(1000):
        trees = []
        for _ in range(n):
            x = np.random.uniform(-bound, bound)
            y = np.random.uniform(-bound, bound)
            a = np.random.uniform(0, 360)
            trees.append((x, y, a))
        if not has_overlap(trees):
            return trees
    return None


def simulated_annealing(
    trees: List[Tuple[float, float, float]],
    iterations: int = 200000,
    T0: float = 0.5,
    Tf: float = 0.00001
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """SA optimization."""
    n = len(trees)
    trees = list(trees)

    best_trees = list(trees)
    current_side = compute_side(trees)
    best_side = current_side

    alpha = (Tf / T0) ** (1 / iterations)
    T = T0

    for i in range(iterations):
        idx = np.random.randint(0, n)
        x, y, a = trees[idx]

        scale = max(0.005, T * 2)
        dx = np.random.normal(0, scale * 0.15)
        dy = np.random.normal(0, scale * 0.15)
        da = np.random.normal(0, scale * 15)

        new_tree = (x + dx, y + dy, (a + da) % 360)
        candidate = trees[:idx] + [new_tree] + trees[idx + 1:]

        if has_overlap(candidate):
            T *= alpha
            continue

        new_side = compute_side(candidate)
        delta = new_side - current_side

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            trees = candidate
            current_side = new_side

            if new_side < best_side:
                best_side = new_side
                best_trees = list(trees)

        T *= alpha

    return best_side, best_trees


def main():
    n = 6
    restarts = 50  # Many restarts
    iterations = 200000  # Long runs

    print(f"=== Intensive n={n} Optimization ===")
    print(f"Restarts: {restarts}, Iterations: {iterations}")

    # Load current best
    current_trees = [
        (0.0403, 0.0479, 156.36),
        (0.4247, -0.4444, 113.60),
        (0.5240, 0.1550, 203.14),
        (-0.4423, 0.3781, 272.63),
        (0.6457, 0.5960, 87.12),
        (-0.5439, -0.4721, 23.74),
    ]
    current_side = compute_side(current_trees)
    print(f"\nCurrent: side={current_side:.6f}, score={current_side**2/n:.6f}")

    best_side = current_side
    best_trees = current_trees

    # First refine current
    print("\n1. Refining current solution...")
    side, trees = simulated_annealing(current_trees, iterations)
    if side < best_side:
        best_side = side
        best_trees = trees
        print(f"   Refined: {side:.6f}")
    else:
        print(f"   No improvement")

    # Try various initializations
    print("\n2. Trying pattern-based initializations...")

    # Circular patterns
    for radius in [0.4, 0.5, 0.6, 0.7]:
        for base_angle in [0, 30, 45, 60, 90]:
            init = circular_init(n, radius, base_angle)
            if not has_overlap(init):
                side, trees = simulated_annealing(init, iterations)
                if side < best_side:
                    best_side = side
                    best_trees = trees
                    print(f"   Circular r={radius} ba={base_angle}: {side:.6f} *NEW BEST*")

    # Hexagonal
    init = hexagonal_init(n)
    if not has_overlap(init):
        side, trees = simulated_annealing(init, iterations)
        if side < best_side:
            best_side = side
            best_trees = trees
            print(f"   Hexagonal: {side:.6f} *NEW BEST*")

    # Random restarts
    print(f"\n3. Random restarts ({restarts} runs)...")
    for r in range(restarts):
        init = random_init(n, bound=0.8)
        if init is None:
            continue
        side, trees = simulated_annealing(init, iterations)
        if side < best_side:
            best_side = side
            best_trees = trees
            print(f"   Restart {r+1}: {side:.6f} *NEW BEST*")
        elif (r + 1) % 10 == 0:
            print(f"   Restart {r+1}: best so far = {best_side:.6f}")

    # Final result
    print(f"\n=== Final Result ===")
    improvement = current_side - best_side
    print(f"Side: {current_side:.6f} -> {best_side:.6f} (delta={improvement:.6f})")
    print(f"Score: {current_side**2/n:.6f} -> {best_side**2/n:.6f}")

    if improvement > 0.0001:
        print("\nImproved configuration:")
        for i, (x, y, a) in enumerate(best_trees):
            print(f"  Tree {i}: x={x:.10f}, y={y:.10f}, angle={a:.10f}")

        # Save
        result = {
            '6': {
                'side': best_side,
                'trees': best_trees
            }
        }

        try:
            with open('python/gen113_optimized.json') as f:
                existing = json.load(f)
        except:
            existing = {}

        existing['6'] = result['6']

        with open('python/gen113_optimized.json', 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved to gen113_optimized.json")
    else:
        print("\nNo improvement found.")


if __name__ == '__main__':
    main()
