#!/usr/bin/env python3
"""
Gen117 Fast Pattern-Based CMA-ES Optimizer

Uses Shapely for fast overlap checking during optimization,
then strict segment-intersection validation for final acceptance.

Usage:
    python3 python/gen117_fast.py --n 20 21 22 --evals 3000
"""

import math
import numpy as np
import cma
import csv
import json
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely import affinity

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLY = Polygon(TREE_VERTICES)


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_shapely_poly(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLY, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)

    def get_vertices(self) -> List[Tuple[float, float]]:
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return [
            (vx * cos_a - vy * sin_a + self.x,
             vx * sin_a + vy * cos_a + self.y)
            for vx, vy in TREE_VERTICES
        ]


def compute_side(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


# ============ FAST VALIDATION (Shapely-based) ============

def has_any_overlap_fast(trees: List[Tree], tol: float = 1e-9) -> bool:
    """Fast overlap check using Shapely (area-based)."""
    polys = [t.get_shapely_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def compute_overlap_area(trees: List[Tree]) -> float:
    """Compute total overlap area using Shapely."""
    polys = [t.get_shapely_poly() for t in trees]
    n = len(polys)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                total += polys[i].intersection(polys[j]).area
    return total


# ============ STRICT VALIDATION (matches Kaggle) ============

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon(point, polygon):
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


def polygons_overlap_strict(poly1, poly2) -> bool:
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i + 1) % n1],
                                  poly2[j], poly2[(j + 1) % n2]):
                return True
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True
    return False


def has_any_overlap_strict(trees: List[Tree]) -> bool:
    """Strict validation matching Kaggle checker."""
    polys = [t.get_vertices() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_strict(polys[i], polys[j]):
                return True
    return False


# ============ PATTERN INITIALIZATIONS ============

def radial_pattern(n: int, radius_scale: float = 0.5) -> List[Tree]:
    trees = []
    base_radius = radius_scale * math.sqrt(n) * 0.6
    placed = 0
    ring = 0
    while placed < n:
        if ring == 0:
            trees.append(Tree(0, 0, 0))
            placed += 1
        else:
            ring_radius = base_radius * ring / max(1, math.sqrt(n) / 2)
            trees_in_ring = min(6 * ring, n - placed)
            for i in range(trees_in_ring):
                theta = 2 * math.pi * i / trees_in_ring + (ring % 2) * math.pi / trees_in_ring
                x = ring_radius * math.cos(theta)
                y = ring_radius * math.sin(theta)
                angle = (math.degrees(theta) + (i % 4) * 90) % 360
                trees.append(Tree(x, y, angle))
                placed += 1
                if placed >= n:
                    break
        ring += 1
    return trees[:n]


def hexagonal_pattern(n: int, spacing: float = 0.65) -> List[Tree]:
    trees = []
    cols = int(math.ceil(math.sqrt(n * 4 / 3)))
    rows = int(math.ceil(n / cols))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = (row % 2) * spacing / 2
            x = col * spacing + x_offset
            y = row * spacing * 0.866
            angle = ((row + col) % 4) * 90
            trees.append(Tree(x, y, angle))
            idx += 1
    return trees[:n]


def spiral_pattern(n: int) -> List[Tree]:
    trees = []
    phi = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * math.pi * i / phi**2
        r = 0.3 * math.sqrt(i + 1)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        angle = (math.degrees(theta) + (i % 4) * 90) % 360
        trees.append(Tree(x, y, angle))
    return trees


def grid_90_pattern(n: int, spacing: float = 0.6) -> List[Tree]:
    trees = []
    side = int(math.ceil(math.sqrt(n)))
    idx = 0
    for row in range(side):
        for col in range(side):
            if idx >= n:
                break
            x = col * spacing
            y = row * spacing
            angle = ((row * 2 + col) % 4) * 90
            trees.append(Tree(x, y, angle))
            idx += 1
    return trees[:n]


# ============ CMA-ES OPTIMIZATION ============

def trees_to_params(trees: List[Tree]) -> np.ndarray:
    params = []
    for t in trees:
        params.extend([t.x, t.y, t.angle / 360.0])
    return np.array(params)


def params_to_trees(params: np.ndarray) -> List[Tree]:
    n = len(params) // 3
    trees = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        angle = (params[3 * i + 2] % 1.0) * 360.0
        trees.append(Tree(x, y, angle))
    return trees


def objective_function(params: np.ndarray, penalty_weight: float = 100.0) -> float:
    trees = params_to_trees(params)
    side = compute_side(trees)
    overlap = compute_overlap_area(trees)
    return side + penalty_weight * overlap


def optimize_single_pattern(
    n: int,
    init_trees: List[Tree],
    pattern_name: str,
    max_evals: int = 3000,
    sigma0: float = 0.15,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]]]:
    """Optimize from a single pattern initialization."""
    initial_params = trees_to_params(init_trees)
    dim = len(initial_params)

    options = {
        'maxfevals': max_evals,
        'verbose': -9,
        'popsize': max(8, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-9,
        'tolx': 1e-9,
    }

    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    best_valid_side = float('inf')
    best_valid_trees = None

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)

        # Check solutions (using fast Shapely check first)
        for x in solutions:
            trees = params_to_trees(x)
            if not has_any_overlap_fast(trees):
                side = compute_side(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = [Tree(t.x, t.y, t.angle) for t in trees]

        generation += 1

    if verbose and best_valid_trees:
        print(f"    {pattern_name}: best={best_valid_side:.4f}")

    return best_valid_side, best_valid_trees


def optimize_n(
    n: int,
    current_best: Optional[Tuple[float, List[Tuple]]] = None,
    max_evals: int = 3000,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]], str]:
    """Optimize n using multiple patterns."""
    if verbose:
        print(f"\n=== n={n} ===")

    best_side = float('inf')
    best_trees = None
    best_pattern = "none"

    if current_best:
        current_side, current_trees_data = current_best
        best_side = current_side
        best_trees = [Tree(t[0], t[1], t[2]) for t in current_trees_data]
        best_pattern = "current"
        if verbose:
            print(f"  Current: {current_side:.4f}")

    # Patterns to try
    pattern_fns = [
        ("radial", radial_pattern),
        ("hexagonal", hexagonal_pattern),
        ("spiral", spiral_pattern),
        ("grid_90", grid_90_pattern),
    ]

    # Also refine current best
    if current_best:
        current_trees = [Tree(t[0], t[1], t[2]) for t in current_best[1]]
        side, trees = optimize_single_pattern(n, current_trees, "refine", max_evals, verbose=verbose)
        if trees and side < best_side:
            best_side = side
            best_trees = trees
            best_pattern = "refine"

    for pattern_name, pattern_fn in pattern_fns:
        try:
            init_trees = pattern_fn(n)
            side, trees = optimize_single_pattern(n, init_trees, pattern_name, max_evals, verbose=verbose)
            if trees and side < best_side:
                if verbose:
                    print(f"    ** NEW BEST: {side:.4f} via {pattern_name}")
                best_side = side
                best_trees = trees
                best_pattern = pattern_name
        except Exception as e:
            if verbose:
                print(f"    {pattern_name}: failed ({e})")

    return best_side, best_trees, best_pattern


def load_current_submission(csv_path: str) -> dict:
    groups = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_parts = row['id'].split('_')
            n = int(id_parts[0])
            x = float(row['x'].lstrip('s'))
            y = float(row['y'].lstrip('s'))
            deg = float(row['deg'].lstrip('s'))
            if n not in groups:
                groups[n] = []
            groups[n].append((x, y, deg))

    results = {}
    for n, trees_data in groups.items():
        trees = [Tree(x, y, a) for x, y, a in trees_data]
        side = compute_side(trees)
        results[n] = (side, trees_data)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[20, 21, 22, 23, 24, 25])
    parser.add_argument('--evals', type=int, default=3000)
    parser.add_argument('--submission', type=str, default='submission_best.csv')
    parser.add_argument('--output', type=str, default='python/gen117_optimized.json')

    args = parser.parse_args()

    print("Gen117 Fast Pattern-Based Optimizer")
    print("=" * 40)

    current = load_current_submission(args.submission)
    print(f"Loaded {len(current)} solutions")

    results = {}
    improvements = []

    for n in args.n:
        current_best = current.get(n)
        side, trees, pattern = optimize_n(n, current_best, args.evals, verbose=True)

        if trees:
            old_side = current_best[0] if current_best else float('inf')
            improvement = old_side - side

            # STRICT validation before accepting
            if improvement > 0.0001:
                if has_any_overlap_strict(trees):
                    print(f"  WARNING: n={n} fails strict validation, keeping original")
                    continue

                score_delta = (old_side**2 - side**2) / n
                improvements.append((n, improvement, score_delta, pattern))
                results[str(n)] = {
                    'side': side,
                    'pattern': pattern,
                    'trees': [(t.x, t.y, t.angle) for t in trees]
                }
                print(f"  IMPROVED: {old_side:.4f} -> {side:.4f} ({improvement:.4f})")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    if improvements:
        print("\n=== IMPROVEMENTS ===")
        total = 0
        for n, imp, sd, pattern in improvements:
            print(f"  n={n}: -{imp:.4f} side ({sd:.4f} score) via {pattern}")
            total += sd
        print(f"  Total: {total:.4f}")
    else:
        print("\nNo improvements found")


if __name__ == '__main__':
    main()
