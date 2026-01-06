#!/usr/bin/env python3
"""
Gen117 Boundary-Focused Optimizer

For large n, the bounding box is determined by just a few trees at the edges.
This optimizer identifies and moves only those boundary trees.

Key insight: Moving interior trees doesn't change the bounding box.
Focus optimization on the 5-10 trees that define the box.

Usage:
    python3 python/gen117_boundary.py --n 100 110 120 --iters 10000
"""

import math
import numpy as np
import csv
import json
import sys
from typing import List, Tuple, Optional, Set
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


def compute_side(trees: List[Tree]) -> Tuple[float, float, float, float, float]:
    """Return (side, minx, maxx, miny, maxy)."""
    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    side = max(maxx - minx, maxy - miny)
    return side, minx, maxx, miny, maxy


def find_boundary_trees(trees: List[Tree], margin: float = 0.1) -> Set[int]:
    """Find trees that contribute to the bounding box."""
    side, minx, maxx, miny, maxy = compute_side(trees)

    boundary = set()
    for i, t in enumerate(trees):
        verts = t.get_vertices()
        for vx, vy in verts:
            if abs(vx - minx) < margin or abs(vx - maxx) < margin or \
               abs(vy - miny) < margin or abs(vy - maxy) < margin:
                boundary.add(i)
                break

    return boundary


def has_any_overlap_fast(trees: List[Tree], tol: float = 1e-9) -> bool:
    polys = [t.get_shapely_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                if polys[i].intersection(polys[j]).area > tol:
                    return True
    return False


def check_single_overlap(trees: List[Tree], idx: int, tol: float = 1e-9) -> bool:
    """Check if tree at idx overlaps with any other tree."""
    poly = trees[idx].get_shapely_poly()
    for i, t in enumerate(trees):
        if i == idx:
            continue
        other = t.get_shapely_poly()
        if poly.intersects(other):
            if poly.intersection(other).area > tol:
                return True
    return False


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


def boundary_sa(
    trees: List[Tree],
    iterations: int = 50000,
    T0: float = 0.3,
    Tf: float = 0.0001,
    verbose: bool = True
) -> Tuple[float, List[Tree]]:
    """Simulated annealing focused on boundary trees."""
    n = len(trees)
    trees = [Tree(t.x, t.y, t.angle) for t in trees]

    side, _, _, _, _ = compute_side(trees)
    best_side = side
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]

    # Find boundary trees
    boundary = find_boundary_trees(trees, margin=0.15)
    boundary_list = list(boundary)

    if verbose:
        print(f"  Found {len(boundary)} boundary trees out of {n}")

    if not boundary_list:
        return best_side, best_trees

    alpha = (Tf / T0) ** (1 / iterations)
    T = T0
    improvements = 0

    for i in range(iterations):
        # Pick random boundary tree
        idx = np.random.choice(boundary_list)
        t = trees[idx]

        # Perturbation - try to move inward
        scale = max(0.001, T * 0.5)
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        da = np.random.normal(0, scale * 20)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        old_tree = trees[idx]
        trees[idx] = new_tree

        # Quick single-tree overlap check
        if check_single_overlap(trees, idx):
            trees[idx] = old_tree
            T *= alpha
            continue

        new_side, _, _, _, _ = compute_side(trees)
        delta = new_side - side

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            side = new_side
            if new_side < best_side:
                best_side = new_side
                best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
                improvements += 1

                # Update boundary set
                boundary = find_boundary_trees(trees, margin=0.15)
                boundary_list = list(boundary)
        else:
            trees[idx] = old_tree

        T *= alpha

        if verbose and (i + 1) % 20000 == 0:
            print(f"    iter {i+1}: best={best_side:.4f}, T={T:.6f}, improvements={improvements}")

    return best_side, best_trees


def optimize_n(
    n: int,
    trees: List[Tree],
    current_side: float,
    iterations: int = 50000,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]]]:
    """Optimize using boundary-focused SA."""
    if verbose:
        print(f"\n=== n={n} (current: {current_side:.4f}) ===")

    best_side, best_trees = boundary_sa(trees, iterations, verbose=verbose)

    if best_side < current_side - 0.0001:
        # Strict validation
        if has_any_overlap_strict(best_trees):
            if verbose:
                print(f"  Result fails strict validation!")
            return current_side, None

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
        side, _, _, _, _ = compute_side(trees)
        results[n] = (side, trees_data, trees)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=list(range(50, 201, 10)))
    parser.add_argument('--iters', type=int, default=50000)
    parser.add_argument('--submission', type=str, default='submission_best.csv')
    parser.add_argument('--output', type=str, default='python/gen117_optimized.json')

    args = parser.parse_args()

    print("Gen117 Boundary-Focused Optimizer")
    print("=" * 40)

    current = load_current_submission(args.submission)
    print(f"Loaded {len(current)} solutions")

    results = {}
    improvements = []

    for n in args.n:
        if n not in current:
            print(f"Skipping n={n}")
            continue

        current_side, _, current_trees = current[n]

        side, trees = optimize_n(n, current_trees, current_side, args.iters)

        if trees and side < current_side - 0.0001:
            improvement = current_side - side
            score_delta = (current_side**2 - side**2) / n
            improvements.append((n, improvement, score_delta))
            results[str(n)] = {
                'side': side,
                'trees': [(t.x, t.y, t.angle) for t in trees]
            }
            print(f"  IMPROVED: {current_side:.4f} -> {side:.4f}")
        else:
            print(f"  No improvement")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

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
