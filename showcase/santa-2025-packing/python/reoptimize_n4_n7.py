#!/usr/bin/env python3
"""
Re-optimize n=4 and n=7 with stricter validation.

These failed validation in Gen110. Try multiple restarts from scratch.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

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
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap_strict(trees: List[Tree], tol: float = 1e-9) -> bool:
    """Strict overlap check."""
    polys = [t.get_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def random_config(n: int, bounds: float = 1.0) -> List[Tree]:
    """Generate random configuration."""
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    trees = []
    for _ in range(n):
        x = np.random.uniform(-bounds, bounds)
        y = np.random.uniform(-bounds, bounds)
        a = np.random.choice(angles)
        trees.append(Tree(x, y, a))
    return trees


def local_refine(trees: List[Tree], iterations: int = 10000, strict: bool = True) -> List[Tree]:
    """Local refinement with strict overlap checking."""
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_side = compute_side(best_trees)

    temp = 0.1
    temp_decay = 0.9995

    for i in range(iterations):
        idx = np.random.randint(0, len(best_trees))
        t = best_trees[idx]

        scale = max(0.001, temp * 0.3)
        dx = np.random.uniform(-scale, scale)
        dy = np.random.uniform(-scale, scale)
        da = np.random.uniform(-5, 5)

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


def solve_n4() -> Tuple[float, List[Tree]]:
    """Solve n=4 with multiple restarts."""
    print("Solving n=4 with intensive search...")

    angles = np.arange(0, 360, 15)  # 24 angles
    best_side = float('inf')
    best_trees = None

    # Grid-based approach: place trees one by one
    for a1 in [0, 45, 90, 135]:
        t1 = Tree(0, 0, a1)

        for dx2 in np.arange(0.2, 0.7, 0.03):
            for dy2 in np.arange(-0.5, 0.5, 0.03):
                for a2 in [0, 45, 90, 135, 180, 225, 270, 315]:
                    t2 = Tree(dx2, dy2, a2)
                    if has_overlap_strict([t1, t2]):
                        continue

                    for dx3 in np.arange(-0.5, 0.7, 0.05):
                        for dy3 in np.arange(0.2, 0.9, 0.05):
                            for a3 in [0, 90, 180, 270]:
                                t3 = Tree(dx3, dy3, a3)
                                if has_overlap_strict([t1, t2, t3]):
                                    continue

                                for dx4 in np.arange(-0.5, 0.7, 0.05):
                                    for dy4 in np.arange(-0.7, 0.2, 0.05):
                                        for a4 in [45, 135, 225, 315]:
                                            t4 = Tree(dx4, dy4, a4)
                                            trees = [t1, t2, t3, t4]

                                            if has_overlap_strict(trees):
                                                continue

                                            side = compute_side(trees)
                                            if side < best_side:
                                                best_side = side
                                                best_trees = [Tree(t.x, t.y, t.angle) for t in trees]

    if best_trees:
        print(f"  Grid search found: {best_side:.4f}")
        best_trees = local_refine(best_trees, 20000)
        best_side = compute_side(best_trees)
        print(f"  After refinement: {best_side:.4f}")

        # Validate
        if has_overlap_strict(best_trees):
            print("  WARNING: Overlap detected!")
            return float('inf'), None

    return best_side, best_trees


def solve_n7() -> Tuple[float, List[Tree]]:
    """Solve n=7 with intensive search."""
    print("Solving n=7 with intensive search...")

    # Start from n=6 and add one tree
    best_side = float('inf')
    best_trees = None

    # Load existing n=6 solution
    try:
        with open('python/optimized_small_n.json') as f:
            data = json.load(f)
        n6_trees = [Tree(t[0], t[1], t[2]) for t in data['6']['trees']]
    except:
        print("  Could not load n=6 solution, starting from scratch")
        return float('inf'), None

    print(f"  Starting from n=6 solution with side={compute_side(n6_trees):.4f}")

    # Get bounds
    all_coords = []
    for t in n6_trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin = 0.7
    angles = np.arange(0, 360, 15)

    count = 0
    for dx in np.arange(min_x - margin, max_x + margin, 0.03):
        for dy in np.arange(min_y - margin, max_y + margin, 0.03):
            for angle in angles:
                new_tree = Tree(dx, dy, angle)
                candidate = n6_trees + [new_tree]

                if has_overlap_strict(candidate):
                    continue

                side = compute_side(candidate)
                if side < best_side:
                    best_side = side
                    best_trees = [Tree(t.x, t.y, t.angle) for t in candidate]
                    count += 1

    if best_trees:
        print(f"  Grid search found {count} valid configs, best: {best_side:.4f}")
        best_trees = local_refine(best_trees, 15000)
        best_side = compute_side(best_trees)
        print(f"  After refinement: {best_side:.4f}")

        # Validate
        if has_overlap_strict(best_trees):
            print("  WARNING: Overlap detected!")
            return float('inf'), None

    return best_side, best_trees


def main():
    results = {}

    # Solve n=4
    side4, trees4 = solve_n4()
    if trees4:
        results['4'] = {
            'side': side4,
            'trees': [(t.x, t.y, t.angle) for t in trees4]
        }
        print(f"\nn=4: side={side4:.4f}")

    # Solve n=7
    side7, trees7 = solve_n7()
    if trees7:
        results['7'] = {
            'side': side7,
            'trees': [(t.x, t.y, t.angle) for t in trees7]
        }
        print(f"\nn=7: side={side7:.4f}")

    # Save results
    with open('python/optimized_n4_n7.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to python/optimized_n4_n7.json")


if __name__ == '__main__':
    main()
