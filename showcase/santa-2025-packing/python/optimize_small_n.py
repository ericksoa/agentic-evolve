#!/usr/bin/env python3
"""
Optimize small n values with intensive search.

Uses:
1. Grid search for initial solution
2. Local refinement with random perturbations
3. Iterative improvement

For n=1 to 15, find optimal or near-optimal solutions.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import sys

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE = Polygon(TREE_VERTICES)


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_poly(self) -> Polygon:
        rotated = affinity.rotate(TREE, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_side(trees: List[Tree]) -> float:
    all_coords = []
    for t in trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs)-min(xs), max(ys)-min(ys))


def has_overlap(trees: List[Tree], tol: float = 1e-8) -> bool:
    polys = [t.get_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def solve_n1() -> Tuple[float, List[Tree]]:
    """n=1 is trivially optimal at 45 degrees."""
    return 0.8132, [Tree(0, 0, 45)]


def solve_n2() -> Tuple[float, List[Tree]]:
    """Grid search + refinement for n=2."""
    angles = np.arange(0, 180, 5)
    best_side = float('inf')
    best_trees = None

    # Grid search
    for a1 in angles:
        t1 = Tree(0, 0, a1)
        p1 = t1.get_poly()

        for dx in np.arange(0.2, 0.7, 0.02):
            for dy in np.arange(-0.7, 0.7, 0.02):
                for a2 in angles:
                    t2 = Tree(dx, dy, a2)

                    if has_overlap([t1, t2]):
                        continue

                    side = compute_side([t1, t2])
                    if side < best_side:
                        best_side = side
                        best_trees = [Tree(0, 0, a1), Tree(dx, dy, a2)]

    # Local refinement
    best_trees = local_refine(best_trees, 5000)
    best_side = compute_side(best_trees)

    return best_side, best_trees


def solve_n_general(n: int, prev_trees: Optional[List[Tree]] = None) -> Tuple[float, List[Tree]]:
    """Solve for n trees by adding one at a time."""
    if n == 1:
        return solve_n1()
    if n == 2:
        return solve_n2()

    # Start from n-1 solution
    if prev_trees is None:
        _, prev_trees = solve_n_general(n - 1)

    best_side = float('inf')
    best_trees = None

    # Get bounds of existing packing
    all_coords = []
    for t in prev_trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Search grid for new tree
    angles = np.arange(0, 360, 22.5)
    margin = 0.6

    for dx in np.arange(min_x - margin, max_x + margin, 0.04):
        for dy in np.arange(min_y - margin, max_y + margin, 0.04):
            for angle in angles:
                new_tree = Tree(dx, dy, angle)
                candidate = prev_trees + [new_tree]

                if has_overlap(candidate):
                    continue

                side = compute_side(candidate)
                if side < best_side:
                    best_side = side
                    best_trees = [Tree(t.x, t.y, t.angle) for t in candidate]

    if best_trees:
        # Local refinement
        best_trees = local_refine(best_trees, 3000)
        best_side = compute_side(best_trees)

    return best_side, best_trees


def local_refine(trees: List[Tree], iterations: int = 2000) -> List[Tree]:
    """Local refinement with random perturbations."""
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_side = compute_side(best_trees)

    for _ in range(iterations):
        idx = np.random.randint(0, len(best_trees))
        t = best_trees[idx]

        dx = np.random.uniform(-0.02, 0.02)
        dy = np.random.uniform(-0.02, 0.02)
        da = np.random.uniform(-5, 5)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx+1:]

        if has_overlap(candidate):
            continue

        side = compute_side(candidate)
        if side < best_side:
            best_side = side
            best_trees = candidate

    return best_trees


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-n', type=int, default=10, help='Max n to solve')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--compare-csv', type=str, help='CSV to compare against')

    args = parser.parse_args()

    print(f"Optimizing n=1 to {args.max_n}")
    print("-" * 50)

    # Load comparison data if provided
    comparison = {}
    if args.compare_csv:
        trees = []
        with open(args.compare_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                trees.append((float(row['x']), float(row['y']), float(row['angle'])))

        tree_idx = 0
        for n in range(1, 201):
            if tree_idx + n > len(trees):
                break
            n_trees = trees[tree_idx:tree_idx + n]
            tree_idx += n

            # Compute side
            tree_objs = [Tree(x, y, a) for x, y, a in n_trees]
            side = compute_side(tree_objs)
            comparison[n] = side

    results = {}
    total_improvement = 0.0
    prev_trees = None

    print(f"\n{'n':>4} | {'New':>10} | {'Old':>10} | {'Improve':>10} | {'Score Δ':>10}")
    print("-" * 58)

    for n in range(1, args.max_n + 1):
        side, trees = solve_n_general(n, prev_trees)
        prev_trees = trees

        old_side = comparison.get(n, float('inf'))
        improvement = old_side - side
        score_delta = (old_side**2 - side**2) / n

        if improvement > 0:
            total_improvement += score_delta

        results[n] = {
            'side': side,
            'trees': [(t.x, t.y, t.angle) for t in trees]
        }

        print(f"{n:>4} | {side:>10.4f} | {old_side:>10.4f} | {improvement:>+10.4f} | {score_delta:>+10.4f}")

    print("-" * 58)
    print(f"Total score improvement: {total_improvement:+.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
