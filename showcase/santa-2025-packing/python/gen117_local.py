#!/usr/bin/env python3
"""
Gen117 Local Refinement with Tight CMA-ES

Instead of global pattern search, use very small sigma to locally
refine existing solutions. This avoids escaping to worse local optima.

Key approach:
1. Start from current best
2. Use very small sigma (0.02-0.05)
3. Higher penalty to strictly avoid overlaps
4. Multiple random restarts with small perturbations

Usage:
    python3 python/gen117_local.py --n 20 21 22 23 24 25 --evals 5000
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
    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_any_overlap_fast(trees: List[Tree], tol: float = 1e-9) -> bool:
    polys = [t.get_shapely_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                if polys[i].intersection(polys[j]).area > tol:
                    return True
    return False


def compute_overlap_area(trees: List[Tree]) -> float:
    polys = [t.get_shapely_poly() for t in trees]
    n = len(polys)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                total += polys[i].intersection(polys[j]).area
    return total


# Strict validation
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
    polys = [t.get_vertices() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_strict(polys[i], polys[j]):
                return True
    return False


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


def objective_function(params: np.ndarray, penalty_weight: float = 500.0) -> float:
    """High penalty to strongly discourage overlaps."""
    trees = params_to_trees(params)
    side = compute_side(trees)
    overlap = compute_overlap_area(trees)
    return side + penalty_weight * overlap


def local_refine(
    init_trees: List[Tree],
    max_evals: int = 3000,
    sigma0: float = 0.03,
    verbose: bool = False
) -> Tuple[float, Optional[List[Tree]]]:
    """Local refinement with tight sigma."""
    initial_params = trees_to_params(init_trees)
    initial_side = compute_side(init_trees)
    dim = len(initial_params)

    options = {
        'maxfevals': max_evals,
        'verbose': -9,
        'popsize': max(8, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-10,
        'tolx': 1e-10,
    }

    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    best_valid_side = initial_side
    best_valid_trees = [Tree(t.x, t.y, t.angle) for t in init_trees] if not has_any_overlap_fast(init_trees) else None

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)

        for x in solutions:
            trees = params_to_trees(x)
            if not has_any_overlap_fast(trees):
                side = compute_side(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = [Tree(t.x, t.y, t.angle) for t in trees]

    return best_valid_side, best_valid_trees


def optimize_n(
    n: int,
    current_trees: List[Tree],
    current_side: float,
    restarts: int = 5,
    evals_per_restart: int = 2000,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]]]:
    """Multi-restart local refinement."""
    if verbose:
        print(f"\n=== n={n} (current: {current_side:.4f}) ===")

    best_side = current_side
    best_trees = [Tree(t.x, t.y, t.angle) for t in current_trees]

    # Try different sigmas
    sigmas = [0.02, 0.03, 0.05, 0.08, 0.1]

    for restart in range(restarts):
        sigma = sigmas[restart % len(sigmas)]

        # Add small random perturbation for diversity
        perturbed = []
        for t in current_trees:
            dx = np.random.normal(0, 0.01)
            dy = np.random.normal(0, 0.01)
            da = np.random.normal(0, 5)
            perturbed.append(Tree(t.x + dx, t.y + dy, (t.angle + da) % 360))

        side, trees = local_refine(perturbed, evals_per_restart, sigma)

        if trees and side < best_side:
            # Strict validation
            if not has_any_overlap_strict(trees):
                improvement = best_side - side
                if verbose:
                    print(f"  restart {restart+1} (sigma={sigma}): {side:.4f} (+{improvement:.4f})")
                best_side = side
                best_trees = trees
            elif verbose:
                print(f"  restart {restart+1}: rejected (strict overlap)")

    return best_side, best_trees


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
        results[n] = (side, trees_data, trees)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=list(range(20, 31)))
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--evals', type=int, default=2000)
    parser.add_argument('--submission', type=str, default='submission_best.csv')
    parser.add_argument('--output', type=str, default='python/gen117_optimized.json')

    args = parser.parse_args()

    print("Gen117 Local Refinement Optimizer")
    print("=" * 40)

    current = load_current_submission(args.submission)
    print(f"Loaded {len(current)} solutions")

    results = {}
    improvements = []

    for n in args.n:
        if n not in current:
            print(f"Skipping n={n} (not in submission)")
            continue

        current_side, current_trees_data, current_trees = current[n]

        side, trees = optimize_n(
            n, current_trees, current_side,
            restarts=args.restarts,
            evals_per_restart=args.evals
        )

        if trees and side < current_side - 0.0001:
            improvement = current_side - side
            score_delta = (current_side**2 - side**2) / n
            improvements.append((n, improvement, score_delta))
            results[str(n)] = {
                'side': side,
                'trees': [(t.x, t.y, t.angle) for t in trees]
            }
            print(f"  IMPROVED: {current_side:.4f} -> {side:.4f}")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    if improvements:
        print("\n=== IMPROVEMENTS ===")
        total = 0
        for n, imp, sd in improvements:
            print(f"  n={n}: -{imp:.4f} side ({sd:.4f} score)")
            total += sd
        print(f"  Total: {total:.4f}")
    else:
        print("\nNo improvements found")


if __name__ == '__main__':
    main()
