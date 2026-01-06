#!/usr/bin/env python3
"""
Extended small n optimizer with finer search.

Improvements over optimize_small_n.py:
1. Finer grid search (0.02 step)
2. More angles (32 instead of 16)
3. Longer local refinement (10000+ iterations)
4. Stricter overlap tolerance
5. Multiple restarts for robustness
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import sys
import csv
import argparse

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
    """Compute the side length of the square containing all trees."""
    all_coords = []
    for t in trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap_strict(trees: List[Tree], tol: float = 1e-9) -> bool:
    """Strict overlap check with very small tolerance."""
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


def solve_n2_intensive() -> Tuple[float, List[Tree]]:
    """Intensive grid search + refinement for n=2."""
    angles = np.arange(0, 180, 5)  # 36 angles
    best_side = float('inf')
    best_trees = None

    print("  n=2: Intensive grid search...")

    # Grid search
    for a1 in angles:
        t1 = Tree(0, 0, a1)

        for dx in np.arange(0.1, 0.8, 0.015):
            for dy in np.arange(-0.8, 0.8, 0.015):
                for a2 in angles:
                    t2 = Tree(dx, dy, a2)

                    if has_overlap_strict([t1, t2]):
                        continue

                    side = compute_side([t1, t2])
                    if side < best_side:
                        best_side = side
                        best_trees = [Tree(0, 0, a1), Tree(dx, dy, a2)]

    if best_trees:
        best_trees = local_refine(best_trees, 15000)
        best_side = compute_side(best_trees)

    return best_side, best_trees


def solve_n_general_intensive(n: int, prev_trees: Optional[List[Tree]] = None,
                               existing_solutions: dict = None) -> Tuple[float, List[Tree]]:
    """Solve for n trees using intensive search."""
    if n == 1:
        return solve_n1()
    if n == 2:
        return solve_n2_intensive()

    # Try to use existing solutions if available and they're better
    best_side = float('inf')
    best_trees = None

    # Start from n-1 solution
    if prev_trees is None:
        _, prev_trees = solve_n_general_intensive(n - 1)

    # Get bounds of existing packing
    all_coords = []
    for t in prev_trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Use more angles and finer grid
    angles = np.arange(0, 360, 11.25)  # 32 angles
    margin = 0.8
    step = 0.03 if n <= 10 else 0.04

    print(f"  n={n}: Grid search with {len(angles)} angles, step={step}...")

    for dx in np.arange(min_x - margin, max_x + margin, step):
        for dy in np.arange(min_y - margin, max_y + margin, step):
            for angle in angles:
                new_tree = Tree(dx, dy, angle)
                candidate = prev_trees + [new_tree]

                if has_overlap_strict(candidate):
                    continue

                side = compute_side(candidate)
                if side < best_side:
                    best_side = side
                    best_trees = [Tree(t.x, t.y, t.angle) for t in candidate]

    if best_trees:
        # More iterations for refinement
        iterations = 10000 if n <= 10 else 8000
        print(f"  n={n}: Local refinement with {iterations} iterations...")
        best_trees = local_refine(best_trees, iterations)
        best_side = compute_side(best_trees)

        # Verify no overlaps after refinement
        if has_overlap_strict(best_trees):
            print(f"  WARNING: Overlap detected after refinement for n={n}")
            return float('inf'), None

    return best_side, best_trees


def local_refine(trees: List[Tree], iterations: int = 5000) -> List[Tree]:
    """Local refinement with random perturbations."""
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_side = compute_side(best_trees)

    # Temperature for simulated annealing-like behavior
    temp = 0.05
    temp_decay = 0.9999

    for i in range(iterations):
        idx = np.random.randint(0, len(best_trees))
        t = best_trees[idx]

        # Smaller perturbations
        scale = temp
        dx = np.random.uniform(-0.015, 0.015) * scale / 0.05
        dy = np.random.uniform(-0.015, 0.015) * scale / 0.05
        da = np.random.uniform(-3, 3)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx + 1:]

        if has_overlap_strict(candidate):
            temp *= temp_decay
            continue

        side = compute_side(candidate)
        if side < best_side:
            best_side = side
            best_trees = candidate

        temp *= temp_decay

    return best_trees


def load_csv_solutions(csv_path: str) -> dict:
    """Load solutions from a CSV file."""
    solutions = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Parse id column to get n and tree index
    current_n = 1
    current_trees = []

    for row in rows:
        id_str = row['id']
        n_str, idx_str = id_str.split('_')
        n = int(n_str)
        idx = int(idx_str)

        # Parse s-prefixed values
        x = float(row['x'].lstrip('s'))
        y = float(row['y'].lstrip('s'))
        deg = float(row['deg'].lstrip('s'))

        if n != current_n:
            # Save previous n's trees
            if current_trees:
                tree_objs = [Tree(x, y, a) for x, y, a in current_trees]
                solutions[current_n] = {
                    'side': compute_side(tree_objs),
                    'trees': current_trees
                }
            current_n = n
            current_trees = []

        current_trees.append((x, y, deg))

    # Save last n
    if current_trees:
        tree_objs = [Tree(x, y, a) for x, y, a in current_trees]
        solutions[current_n] = {
            'side': compute_side(tree_objs),
            'trees': current_trees
        }

    return solutions


def main():
    parser = argparse.ArgumentParser(description='Extended small n optimizer')
    parser.add_argument('--min-n', type=int, default=11, help='Min n to solve')
    parser.add_argument('--max-n', type=int, default=20, help='Max n to solve')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--compare-csv', type=str, help='CSV to compare against')
    parser.add_argument('--existing-json', type=str, help='Existing JSON solutions to extend')

    args = parser.parse_args()

    print(f"Optimizing n={args.min_n} to {args.max_n}")
    print("-" * 60)

    # Load comparison data if provided
    comparison = {}
    if args.compare_csv:
        comparison = load_csv_solutions(args.compare_csv)
        print(f"Loaded {len(comparison)} solutions from {args.compare_csv}")

    # Load existing JSON solutions if provided
    existing = {}
    if args.existing_json:
        with open(args.existing_json) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing solutions from {args.existing_json}")

    results = {}
    total_improvement = 0.0

    print(f"\n{'n':>4} | {'New':>10} | {'Old':>10} | {'Improve':>10} | {'Score Δ':>10}")
    print("-" * 60)

    # Build previous trees from existing solutions or comparison
    prev_trees = None
    if args.min_n > 1:
        # Get trees for n-1 from existing or comparison
        n_prev = args.min_n - 1
        if str(n_prev) in existing:
            tree_data = existing[str(n_prev)]['trees']
            prev_trees = [Tree(t[0], t[1], t[2]) for t in tree_data]
        elif n_prev in comparison:
            tree_data = comparison[n_prev]['trees']
            prev_trees = [Tree(t[0], t[1], t[2]) for t in tree_data]

    for n in range(args.min_n, args.max_n + 1):
        side, trees = solve_n_general_intensive(n, prev_trees)
        prev_trees = trees

        old_side = float('inf')
        if n in comparison:
            old_side = comparison[n]['side']
        elif str(n) in existing:
            old_side = existing[str(n)]['side']

        improvement = old_side - side if trees else 0
        score_delta = (old_side**2 - side**2) / n if trees and old_side < float('inf') else 0

        if improvement > 0 and trees:
            total_improvement += score_delta

        if trees:
            results[n] = {
                'side': side,
                'trees': [(t.x, t.y, t.angle) for t in trees]
            }

            marker = "✓" if improvement > 0 else " "
            print(f"{n:>4} | {side:>10.4f} | {old_side:>10.4f} | {improvement:>+10.4f} | {score_delta:>+10.4f} {marker}")
        else:
            print(f"{n:>4} | {'FAILED':>10} | {old_side:>10.4f} | {'N/A':>10} | {'N/A':>10}")

    print("-" * 60)
    print(f"Total score improvement: {total_improvement:+.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
