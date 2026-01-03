#!/usr/bin/env python3
"""
Find optimal packings for small N using exhaustive search.
These can be hardcoded into the Rust solution.
"""

import numpy as np
from typing import List, Tuple
import itertools
import time
from dataclasses import dataclass

TREE_VERTICES = np.array([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])

ROTATION_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def rotate_polygon(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return vertices @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])


def cross_product_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a1, a2, b1, b2):
    eps = 1e-9
    d1 = cross_product_2d(b1, b2, a1)
    d2 = cross_product_2d(b1, b2, a2)
    d3 = cross_product_2d(a1, a2, b1)
    d4 = cross_product_2d(a1, a2, b2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False


def point_in_polygon(point, polygon):
    winding = 0
    n = len(polygon)
    px, py = point
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 <= py:
            if y2 > py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross > 1e-10:
                    winding += 1
        else:
            if y2 <= py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross < -1e-10:
                    winding -= 1
    return winding != 0


def polygons_overlap(poly1, poly2):
    b1_min, b1_max = poly1.min(axis=0), poly1.max(axis=0)
    b2_min, b2_max = poly2.min(axis=0), poly2.max(axis=0)
    eps = 1e-9
    if b1_max[0] + eps < b2_min[0] or b2_max[0] + eps < b1_min[0] or \
       b1_max[1] + eps < b2_min[1] or b2_max[1] + eps < b1_min[1]:
        return False

    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return True
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True
    return False


@dataclass
class PlacedTree:
    x: float
    y: float
    rot: int

    def get_vertices(self):
        poly = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[self.rot])
        return poly + np.array([self.x, self.y])


def trees_overlap(t1, t2):
    return polygons_overlap(t1.get_vertices(), t2.get_vertices())


def packing_side(trees):
    if not trees:
        return 0.0
    all_verts = np.vstack([t.get_vertices() for t in trees])
    mn = all_verts.min(axis=0)
    mx = all_verts.max(axis=0)
    return max(mx[0] - mn[0], mx[1] - mn[1])


def check_valid(trees):
    for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            if trees_overlap(trees[i], trees[j]):
                return False
    return True


def local_refine(trees, max_iter=500):
    """Refine tree positions using local search."""
    best_trees = [PlacedTree(t.x, t.y, t.rot) for t in trees]
    best_side = packing_side(best_trees)

    for _ in range(max_iter):
        # Pick random tree
        idx = np.random.randint(len(trees))
        old = trees[idx]

        # Try small perturbation
        dx = np.random.normal(0, 0.02)
        dy = np.random.normal(0, 0.02)
        new_rot = np.random.randint(8) if np.random.random() < 0.1 else old.rot

        trees[idx] = PlacedTree(old.x + dx, old.y + dy, new_rot)

        if check_valid(trees):
            side = packing_side(trees)
            if side < best_side:
                best_trees = [PlacedTree(t.x, t.y, t.rot) for t in trees]
                best_side = side
            elif np.random.random() > 0.9:
                trees[idx] = old
            # else accept with some probability
        else:
            trees[idx] = old

    return best_trees, best_side


def optimize_n1():
    """Optimal n=1 is trivial: single tree at origin."""
    best = None
    best_side = float('inf')

    for rot in range(8):
        trees = [PlacedTree(0, 0, rot)]
        side = packing_side(trees)
        if side < best_side:
            best_side = side
            best = trees

    return best, best_side


def optimize_n2():
    """Find optimal packing for n=2."""
    best = None
    best_side = float('inf')

    # Try all rotation combinations
    for r1 in range(8):
        for r2 in range(8):
            # Binary search for closest valid position in each direction
            for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
                dx = np.cos(angle)
                dy = np.sin(angle)

                # Find minimum distance
                lo, hi = 0.3, 2.0
                while hi - lo > 0.001:
                    mid = (lo + hi) / 2
                    t1 = PlacedTree(0, 0, r1)
                    t2 = PlacedTree(mid * dx, mid * dy, r2)
                    if trees_overlap(t1, t2):
                        lo = mid
                    else:
                        hi = mid

                t1 = PlacedTree(0, 0, r1)
                t2 = PlacedTree(hi * dx, hi * dy, r2)
                if not trees_overlap(t1, t2):
                    trees = [t1, t2]
                    trees, side = local_refine(trees, max_iter=50)
                    if side < best_side:
                        best_side = side
                        best = [PlacedTree(t.x, t.y, t.rot) for t in trees]

    return best, best_side


def optimize_n3():
    """Find optimal packing for n=3."""
    best = None
    best_side = float('inf')

    # Start with n=2 solution
    trees_2, _ = optimize_n2()
    if not trees_2:
        return None, float('inf')

    # Try adding 3rd tree in various positions
    for r3 in range(8):
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Binary search for position
            lo, hi = 0.3, 3.0
            while hi - lo > 0.01:
                mid = (lo + hi) / 2
                t3 = PlacedTree(mid * dx, mid * dy, r3)
                trees = trees_2 + [t3]
                if check_valid(trees):
                    hi = mid
                else:
                    lo = mid

            t3 = PlacedTree(hi * dx, hi * dy, r3)
            trees = [PlacedTree(t.x, t.y, t.rot) for t in trees_2] + [t3]
            if check_valid(trees):
                trees, side = local_refine(trees, max_iter=100)
                if side < best_side:
                    best_side = side
                    best = [PlacedTree(t.x, t.y, t.rot) for t in trees]

    return best, best_side


def optimize_n_general(n, prev_trees=None, max_attempts=50):
    """General optimization for larger n."""
    best = None
    best_side = float('inf')

    if prev_trees is None and n > 1:
        # Start from previous solution
        return None, float('inf')

    base_trees = prev_trees if prev_trees else []

    # Try adding new tree in various positions
    for r in range(8):
        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2*np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Binary search for position
            lo, hi = 0.3, 5.0
            while hi - lo > 0.01:
                mid = (lo + hi) / 2
                new_tree = PlacedTree(mid * dx, mid * dy, r)
                trees = base_trees + [new_tree]
                if check_valid(trees):
                    hi = mid
                else:
                    lo = mid

            new_tree = PlacedTree(hi * dx, hi * dy, r)
            trees = [PlacedTree(t.x, t.y, t.rot) for t in base_trees] + [new_tree]
            if check_valid(trees):
                trees, side = local_refine(trees, max_iter=200)
                if side < best_side:
                    best_side = side
                    best = [PlacedTree(t.x, t.y, t.rot) for t in trees]

    return best, best_side


def center_packing(trees):
    """Center packing around origin."""
    if not trees:
        return trees

    all_verts = np.vstack([t.get_vertices() for t in trees])
    center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2

    return [PlacedTree(t.x - center[0], t.y - center[1], t.rot) for t in trees]


def main():
    print("Santa 2025 - Optimal Small N Search")
    print("=" * 60)

    results = {}
    prev_trees = None

    for n in range(1, 16):
        print(f"\nOptimizing n={n}...")
        start = time.time()

        if n == 1:
            trees, side = optimize_n1()
        elif n == 2:
            trees, side = optimize_n2()
        elif n == 3:
            trees, side = optimize_n3()
        else:
            trees, side = optimize_n_general(n, prev_trees, max_attempts=80)

        elapsed = time.time() - start

        if trees:
            trees = center_packing(trees)
            prev_trees = trees

            # Final refinement
            trees, side = local_refine(trees, max_iter=500)
            trees = center_packing(trees)

            results[n] = {
                'side': side,
                'score': side**2 / n,
                'trees': [(t.x, t.y, t.rot) for t in trees]
            }

            print(f"  side = {side:.4f}, score = {side**2/n:.4f}, time = {elapsed:.1f}s")

            # Print Rust-compatible format
            print(f"  // n={n}: [(x, y, rotation_idx), ...]")
            print(f"  // {[(f'{t.x:.4f}', f'{t.y:.4f}', t.rot) for t in trees]}")
        else:
            print(f"  FAILED")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Rust Hardcode Format:")
    print("=" * 60)

    total_score = 0
    for n, data in sorted(results.items()):
        total_score += data['score']
        print(f"// n={n}: side={data['side']:.4f}")
        coords = ', '.join([f"({t[0]:.4f}, {t[1]:.4f}, {t[2]})" for t in data['trees']])
        print(f"// vec![{coords}]")

    print(f"\nTotal score (n=1..{max(results.keys())}): {total_score:.4f}")


if __name__ == '__main__':
    main()
