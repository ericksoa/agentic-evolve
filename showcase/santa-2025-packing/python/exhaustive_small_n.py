#!/usr/bin/env python3
"""
Exhaustive search for optimal small n solutions.

For n=1 to 5, we can afford to do exhaustive/grid search over positions and angles.
This should find provably optimal or near-optimal solutions.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools
from shapely.geometry import Polygon
from shapely import affinity
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

TREE_VERTICES = [
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_side_length(trees: List[Tree]) -> float:
    """Compute bounding box side length."""
    all_coords = []
    for t in trees:
        poly = t.get_polygon()
        all_coords.extend(list(poly.exterior.coords))

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap(trees: List[Tree], tolerance: float = 1e-8) -> bool:
    """Check if any pair overlaps."""
    n = len(trees)
    polys = [t.get_polygon() for t in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tolerance:
                    return True
    return False


def solve_n1():
    """
    Find optimal placement for n=1.

    Just need to find the rotation that minimizes bounding box.
    """
    best_side = float('inf')
    best_tree = None

    # Try many rotations
    for angle in np.linspace(0, 180, 361):  # 0.5 degree steps
        tree = Tree(0, 0, angle)
        poly = tree.get_polygon()
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        side = max(width, height)

        if side < best_side:
            best_side = side
            best_tree = tree

    return best_side, [best_tree]


def solve_n2(grid_step: float = 0.02, angle_step: float = 15.0):
    """
    Find optimal placement for n=2.

    Grid search over relative positions and rotations.
    """
    # Fix first tree at origin
    angles = np.arange(0, 360, angle_step)

    best_side = float('inf')
    best_trees = None

    # For each rotation of first tree
    for a1 in [0, 45, 90]:  # Symmetry reduces search
        t1 = Tree(0, 0, a1)
        p1 = t1.get_polygon()
        b1 = p1.bounds

        # Search grid for second tree position
        # Only need to search one quadrant due to symmetry
        for dx in np.arange(0, 2.0, grid_step):
            for dy in np.arange(-1.0, 1.0, grid_step):
                for a2 in angles:
                    t2 = Tree(dx, dy, a2)
                    p2 = t2.get_polygon()

                    # Quick bounds check
                    b2 = p2.bounds
                    all_min_x = min(b1[0], b2[0])
                    all_max_x = max(b1[2], b2[2])
                    all_min_y = min(b1[1], b2[1])
                    all_max_y = max(b1[3], b2[3])
                    side = max(all_max_x - all_min_x, all_max_y - all_min_y)

                    if side >= best_side:
                        continue

                    # Check overlap
                    if p1.intersects(p2):
                        inter = p1.intersection(p2)
                        if inter.area > 1e-8:
                            continue

                    best_side = side
                    best_trees = [t1, t2]

    return best_side, best_trees


def try_placement(args):
    """Try a specific placement configuration (for parallel execution)."""
    trees_config, = args
    trees = [Tree(x, y, a) for x, y, a in trees_config]

    if has_overlap(trees):
        return float('inf'), None

    side = compute_side_length(trees)
    return side, trees_config


def solve_n_general(n: int, grid_step: float = 0.05, angle_step: float = 45.0,
                    max_configs: int = 1000000):
    """
    General solver for n trees using smart sampling.

    Strategy:
    1. Start with n=1 optimal placement
    2. For each additional tree, find best position relative to existing
    3. Then do local optimization
    """
    if n == 1:
        return solve_n1()
    if n == 2:
        return solve_n2()

    # For n >= 3, use incremental greedy with refinement
    # First get n-1 solution
    prev_side, prev_trees = solve_n_general(n - 1, grid_step, angle_step, max_configs)

    if prev_trees is None:
        return float('inf'), None

    # Find best position for nth tree
    angles = np.arange(0, 360, angle_step)

    # Get bounds of existing trees
    all_coords = []
    for t in prev_trees:
        poly = t.get_polygon()
        all_coords.extend(list(poly.exterior.coords))

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Search around the boundary of existing packing
    search_range = 0.5  # How far beyond current bounds to search
    best_side = float('inf')
    best_trees = None

    prev_polys = [t.get_polygon() for t in prev_trees]

    for dx in np.arange(min_x - search_range, max_x + search_range, grid_step):
        for dy in np.arange(min_y - search_range, max_y + search_range, grid_step):
            for angle in angles:
                new_tree = Tree(dx, dy, angle)
                new_poly = new_tree.get_polygon()

                # Check overlap with all existing
                overlaps = False
                for p in prev_polys:
                    if new_poly.intersects(p):
                        inter = new_poly.intersection(p)
                        if inter.area > 1e-8:
                            overlaps = True
                            break

                if overlaps:
                    continue

                # Compute side
                new_bounds = new_poly.bounds
                all_min_x = min(min_x, new_bounds[0])
                all_max_x = max(max_x, new_bounds[2])
                all_min_y = min(min_y, new_bounds[1])
                all_max_y = max(max_y, new_bounds[3])
                side = max(all_max_x - all_min_x, all_max_y - all_min_y)

                if side < best_side:
                    best_side = side
                    best_trees = prev_trees + [new_tree]

    return best_side, best_trees


def local_refine(trees: List[Tree], iterations: int = 5000) -> Tuple[float, List[Tree]]:
    """Refine solution with local search."""
    import random

    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_side = compute_side_length(best_trees)

    for _ in range(iterations):
        # Pick random tree
        idx = random.randint(0, len(best_trees) - 1)
        t = best_trees[idx]

        # Try small perturbation
        dx = random.uniform(-0.02, 0.02)
        dy = random.uniform(-0.02, 0.02)
        da = random.choice([0, 5, -5, 10, -10, 15, -15])

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx+1:]

        if has_overlap(candidate):
            continue

        side = compute_side_length(candidate)
        if side < best_side:
            best_trees = candidate
            best_side = side

    return best_side, best_trees


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-n', type=int, default=5, help='Max n to solve')
    parser.add_argument('--refine-iters', type=int, default=10000, help='Refinement iterations')
    args = parser.parse_args()

    print("Exhaustive search for small n")
    print("-" * 40)

    total_score = 0.0
    results = []

    for n in range(1, args.max_n + 1):
        print(f"\nSolving n={n}...")

        if n == 1:
            side, trees = solve_n1()
        elif n == 2:
            side, trees = solve_n2(grid_step=0.01, angle_step=5.0)
        else:
            side, trees = solve_n_general(n, grid_step=0.03, angle_step=22.5)

        if trees:
            print(f"  Grid search: side={side:.4f}")

            # Refine
            refined_side, refined_trees = local_refine(trees, iterations=args.refine_iters)
            print(f"  After refinement: side={refined_side:.4f}")

            score = refined_side**2 / n
            total_score += score
            print(f"  Score: {score:.4f}")

            results.append((n, refined_side, refined_trees))
        else:
            print(f"  No valid solution found!")

    print("-" * 40)
    print(f"Total score (n=1..{args.max_n}): {total_score:.4f}")

    # Print tree positions
    print("\nBest solutions:")
    for n, side, trees in results:
        print(f"\nn={n}, side={side:.4f}:")
        for i, t in enumerate(trees):
            print(f"  Tree {i}: x={t.x:.4f}, y={t.y:.4f}, angle={t.angle:.1f}")


if __name__ == '__main__':
    main()
