#!/usr/bin/env python3
"""
Gen120: Fast search for small n configurations.

Simplified approach: use scipy.optimize with good initialization.
"""

import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from shapely.geometry import Polygon
from shapely import affinity
from typing import List, Tuple
import csv
from collections import defaultdict
import time

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)
TREE_AREA = TREE_POLYGON.area


def get_tree_polygon(x: float, y: float, angle: float) -> Polygon:
    rotated = affinity.rotate(TREE_POLYGON, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_bounding_side(trees: List[Tuple[float, float, float]]) -> float:
    all_points = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(poly.exterior.coords)
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trees_overlap(trees: List[Tuple[float, float, float]], tol: float = 1e-9) -> bool:
    n = len(trees)
    polys = [get_tree_polygon(x, y, a) for x, y, a in trees]
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def objective(params, n, penalty=1000.0):
    """Objective function for n trees."""
    trees = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2] % 360
        trees.append((x, y, a))

    if trees_overlap(trees):
        return penalty

    return compute_bounding_side(trees)


def try_many_starts(n: int, num_starts: int = 50) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Try many random starts with Nelder-Mead."""
    best_side = float('inf')
    best_trees = []

    for start in range(num_starts):
        # Random initialization
        x0 = []
        for i in range(n):
            x0.extend([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(0, 360)
            ])

        result = minimize(
            objective,
            x0,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        if result.fun < best_side:
            trees = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i + 1]
                a = result.x[3*i + 2] % 360
                trees.append((x, y, a))

            if not trees_overlap(trees):
                best_side = result.fun
                best_trees = trees
                print(f"  Start {start+1}: new best {best_side:.6f}")

    return best_side, best_trees


def optimize_from_current(trees: List[Tuple[float, float, float]], num_restarts: int = 10) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Optimize starting from current solution with perturbations."""
    n = len(trees)
    best_side = compute_bounding_side(trees)
    best_trees = trees.copy()

    for r in range(num_restarts):
        # Start from current with small perturbation
        x0 = []
        for x, y, a in trees:
            x0.extend([
                x + np.random.uniform(-0.1, 0.1),
                y + np.random.uniform(-0.1, 0.1),
                (a + np.random.uniform(-20, 20)) % 360
            ])

        result = minimize(
            objective,
            x0,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7}
        )

        if result.fun < best_side:
            new_trees = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i + 1]
                a = result.x[3*i + 2] % 360
                new_trees.append((x, y, a))

            if not trees_overlap(new_trees):
                best_side = result.fun
                best_trees = new_trees
                print(f"  Restart {r+1}: new best {best_side:.6f}")

    return best_side, best_trees


def try_specific_arrangements_n2() -> Tuple[float, List[Tuple[float, float, float]]]:
    """Try specific arrangements for n=2."""
    best_side = float('inf')
    best_trees = []

    # Try various angle pairs
    angle_pairs = [
        (0, 180),    # Opposing directions
        (45, 225),
        (90, 270),
        (0, 90),     # Perpendicular
        (45, 135),
        (0, 45),
        (0, 0),      # Same direction
        (180, 180),
    ]

    for a1, a2 in angle_pairs:
        for offset in [0, 30, 60, 90, 120, 150]:
            angles = [(a1 + offset) % 360, (a2 + offset) % 360]

            # Try different position offsets
            for dx in np.linspace(-1.5, 1.5, 10):
                for dy in np.linspace(-1.5, 1.5, 10):
                    trees = [(0, 0, angles[0]), (dx, dy, angles[1])]

                    if not trees_overlap(trees):
                        side = compute_bounding_side(trees)
                        if side < best_side:
                            best_side = side
                            best_trees = trees

    print(f"  Specific arrangements search found: {best_side:.6f}")
    return best_side, best_trees


def load_submission(csv_path: str) -> dict:
    groups = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'][1:])
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            groups[n].append((x, y, deg))
    return groups


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--input', default='submission_best.csv')
    parser.add_argument('--starts', type=int, default=30)
    args = parser.parse_args()

    print("Gen120: Fast Search for Small n")
    print("=" * 50)

    groups = load_submission(args.input)

    for n in args.n:
        current_trees = groups[n]
        current_side = compute_bounding_side(current_trees)

        print(f"\n=== n={n} ===")
        print(f"Current: {current_side:.6f}")

        best_side = current_side
        best_trees = current_trees

        # Approach 1: Optimize from current
        print("\n1. Refining current solution:")
        opt_side, opt_trees = optimize_from_current(current_trees, num_restarts=20)
        if opt_side < best_side:
            best_side = opt_side
            best_trees = opt_trees

        # Approach 2: Try random starts
        print("\n2. Random restarts:")
        rand_side, rand_trees = try_many_starts(n, num_starts=args.starts)
        if rand_side < best_side:
            best_side = rand_side
            best_trees = rand_trees

        # Approach 3: Specific arrangements (only for n=2)
        if n == 2:
            print("\n3. Specific arrangements:")
            spec_side, spec_trees = try_specific_arrangements_n2()
            if spec_side < best_side:
                best_side = spec_side
                best_trees = spec_trees

        # Summary
        if best_side < current_side - 1e-6:
            improvement = current_side - best_side
            score_delta = (current_side**2 - best_side**2) / n
            print(f"\n*** IMPROVED: {current_side:.6f} -> {best_side:.6f}")
            print(f"    Δside={improvement:.6f}, Δscore={score_delta:.6f}")
        else:
            print(f"\nNo improvement for n={n}")


if __name__ == '__main__':
    main()
