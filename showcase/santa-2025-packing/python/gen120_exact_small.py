#!/usr/bin/env python3
"""
Gen120: Exact optimization for small n (2-5)

For very small n, we can afford exhaustive/systematic search.
Try many angle combinations to find the truly optimal configuration.

Key insight: The tree shape is asymmetric, so angle matters significantly.
For n=2, there are specific angle pairs that might pack much better.
"""

import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from shapely.geometry import Polygon
from shapely import affinity
from typing import List, Tuple
import csv
from collections import defaultdict

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)
TREE_AREA = TREE_POLYGON.area
TREE_BOUNDS = TREE_POLYGON.bounds  # (minx, miny, maxx, maxy)


def get_tree_polygon(x: float, y: float, angle: float) -> Polygon:
    """Get tree polygon at given position and angle."""
    rotated = affinity.rotate(TREE_POLYGON, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_bounding_side(trees: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side length for list of (x, y, angle) tuples."""
    all_points = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(poly.exterior.coords)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trees_overlap(trees: List[Tuple[float, float, float]], tol: float = 1e-10) -> bool:
    """Check if any trees overlap."""
    n = len(trees)
    polys = [get_tree_polygon(x, y, a) for x, y, a in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def strict_segments_intersect(A, B, C, D):
    """Check if segment AB intersects segment CD (strict)."""
    def ccw(P, Q, R):
        return (R[1] - P[1]) * (Q[0] - P[0]) - (Q[1] - P[1]) * (R[0] - P[0])

    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon_strict(point, polygon):
    """Ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygons_overlap_strict(poly1_coords, poly2_coords) -> bool:
    """Strict segment-intersection overlap check."""
    poly1 = list(poly1_coords)[:-1]  # Remove closing point
    poly2 = list(poly2_coords)[:-1]
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if strict_segments_intersect(poly1[i], poly1[(i+1) % n1],
                                         poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon_strict(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon_strict(v, poly1):
            return True

    return False


def trees_overlap_strict(trees: List[Tuple[float, float, float]]) -> bool:
    """Strict segment-intersection overlap check."""
    n = len(trees)
    polys = [get_tree_polygon(x, y, a) for x, y, a in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                if polygons_overlap_strict(polys[i].exterior.coords, polys[j].exterior.coords):
                    return True
    return False


def optimize_positions_for_fixed_angles(angles: List[float]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Given fixed angles, optimize positions to minimize bounding box.
    Uses differential evolution for global search.
    """
    n = len(angles)

    def objective(positions_flat):
        """Objective: bounding side with overlap penalty."""
        trees = []
        for i in range(n):
            x = positions_flat[2*i]
            y = positions_flat[2*i + 1]
            trees.append((x, y, angles[i]))

        # Check overlap
        if trees_overlap(trees):
            return 100.0  # Large penalty

        return compute_bounding_side(trees)

    # Search bounds: positions within reasonable range
    bounds = [(-3, 3)] * (2 * n)

    # Differential evolution for global search
    result = differential_evolution(
        objective,
        bounds,
        maxiter=500,
        popsize=20,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42
    )

    if result.fun < 50:  # Valid solution found
        trees = []
        for i in range(n):
            x = result.x[2*i]
            y = result.x[2*i + 1]
            trees.append((x, y, angles[i]))

        # Verify no strict overlap
        if not trees_overlap_strict(trees):
            return result.fun, trees

    return float('inf'), []


def grid_search_angles(n: int, angle_step: float = 15.0) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Grid search over angle combinations.
    For n=2, this is O(360/step)^2 = feasible.
    """
    best_side = float('inf')
    best_trees = []

    angles_to_try = list(np.arange(0, 360, angle_step))

    if n == 2:
        total = len(angles_to_try) ** 2
        count = 0
        for a1 in angles_to_try:
            for a2 in angles_to_try:
                count += 1
                if count % 100 == 0:
                    print(f"  Grid search: {count}/{total}...", end='\r')

                side, trees = optimize_positions_for_fixed_angles([a1, a2])
                if side < best_side:
                    best_side = side
                    best_trees = trees
                    print(f"  New best: {best_side:.6f} with angles [{a1:.1f}, {a2:.1f}]")

    elif n == 3:
        # Coarser grid for n=3
        angles_to_try = list(np.arange(0, 360, 30))
        total = len(angles_to_try) ** 3
        count = 0
        for a1 in angles_to_try:
            for a2 in angles_to_try:
                for a3 in angles_to_try:
                    count += 1
                    if count % 50 == 0:
                        print(f"  Grid search: {count}/{total}...", end='\r')

                    side, trees = optimize_positions_for_fixed_angles([a1, a2, a3])
                    if side < best_side:
                        best_side = side
                        best_trees = trees
                        print(f"  New best: {best_side:.6f} with angles [{a1:.1f}, {a2:.1f}, {a3:.1f}]")

    return best_side, best_trees


def optimize_full_de(n: int) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Full differential evolution over positions AND angles.
    Parameters: x1, y1, a1, x2, y2, a2, ...
    """
    def objective(params):
        trees = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            trees.append((x, y, a))

        if trees_overlap(trees):
            return 100.0

        return compute_bounding_side(trees)

    # Bounds: positions in [-3, 3], angles in [0, 360]
    bounds = []
    for _ in range(n):
        bounds.extend([(-3, 3), (-3, 3), (0, 360)])

    print(f"  Running DE with {3*n} parameters...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=2000,
        popsize=30,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=1,  # Single thread to avoid pickling issues
        seed=42
    )

    if result.fun < 50:
        trees = []
        for i in range(n):
            x = result.x[3*i]
            y = result.x[3*i + 1]
            a = result.x[3*i + 2]
            trees.append((x, y, a))

        if not trees_overlap_strict(trees):
            return result.fun, trees

    return float('inf'), []


def refine_with_nelder_mead(trees: List[Tuple[float, float, float]]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Refine a valid solution with Nelder-Mead."""
    n = len(trees)

    def objective(params):
        new_trees = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2] % 360
            new_trees.append((x, y, a))

        if trees_overlap(new_trees):
            return 100.0

        return compute_bounding_side(new_trees)

    # Initial guess from current trees
    x0 = []
    for x, y, a in trees:
        x0.extend([x, y, a])

    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )

    if result.fun < 50:
        new_trees = []
        for i in range(n):
            x = result.x[3*i]
            y = result.x[3*i + 1]
            a = result.x[3*i + 2] % 360
            new_trees.append((x, y, a))

        if not trees_overlap_strict(new_trees):
            return result.fun, new_trees

    return float('inf'), trees


def load_submission(csv_path: str) -> dict:
    """Load submission and return groups dictionary."""
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

    parser = argparse.ArgumentParser(description='Gen120: Exact optimization for small n')
    parser.add_argument('--n', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--input', default='submission_best.csv')
    parser.add_argument('--angle-step', type=float, default=15.0)
    args = parser.parse_args()

    print("Gen120: Exact Optimization for Small n")
    print("=" * 50)

    groups = load_submission(args.input)

    for n in args.n:
        current_trees = groups[n]
        current_side = compute_bounding_side(current_trees)
        current_score = current_side ** 2 / n

        print(f"\n=== n={n} ===")
        print(f"Current: side={current_side:.6f}, score={current_score:.6f}")

        # Try multiple approaches
        best_side = current_side
        best_trees = current_trees

        # Approach 1: Full DE
        print("\n1. Full Differential Evolution:")
        de_side, de_trees = optimize_full_de(n)
        if de_side < best_side:
            print(f"   Found better: {de_side:.6f}")
            best_side = de_side
            best_trees = de_trees
        else:
            print(f"   No improvement ({de_side:.6f})")

        # Approach 2: Grid search on angles (only for n<=3)
        if n <= 3:
            print("\n2. Grid Search on Angles:")
            grid_side, grid_trees = grid_search_angles(n, args.angle_step)
            if grid_side < best_side:
                print(f"   Found better: {grid_side:.6f}")
                best_side = grid_side
                best_trees = grid_trees
            else:
                print(f"   No improvement ({grid_side:.6f})")

        # Approach 3: Refine best with Nelder-Mead
        print("\n3. Refinement with Nelder-Mead:")
        nm_side, nm_trees = refine_with_nelder_mead(best_trees)
        if nm_side < best_side:
            print(f"   Refined to: {nm_side:.6f}")
            best_side = nm_side
            best_trees = nm_trees
        else:
            print(f"   No improvement")

        # Summary
        if best_side < current_side - 1e-6:
            improvement = current_side - best_side
            score_delta = (current_side**2 - best_side**2) / n
            print(f"\n*** IMPROVED: {current_side:.6f} -> {best_side:.6f} (Δside={improvement:.6f}, Δscore={score_delta:.6f})")
            print(f"    Trees: {best_trees}")
        else:
            print(f"\nNo improvement found for n={n}")


if __name__ == '__main__':
    main()
