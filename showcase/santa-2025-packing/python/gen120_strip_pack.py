#!/usr/bin/env python3
"""
Gen120: Strip Packing with Alternating Orientations

Key insight: The Christmas tree is asymmetric (tall, pointy).
Trees pointing in opposite directions can interlock.

Strategy:
1. Arrange trees in horizontal strips
2. Alternate 0° and 180° orientations within/between strips
3. Offset strips for better interlocking
4. Refine with local search

This is fundamentally different from radial/greedy placement.
"""

import math
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
import csv
from collections import defaultdict

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)
TREE_HEIGHT = 1.0  # From -0.2 to 0.8
TREE_WIDTH = 0.7   # From -0.35 to 0.35


def get_tree_polygon(x: float, y: float, angle: float) -> Polygon:
    rotated = affinity.rotate(TREE_POLYGON, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def trees_overlap(trees: List[Tuple[float, float, float]], tol: float = 1e-9) -> bool:
    n = len(trees)
    polys = [get_tree_polygon(x, y, a) for x, y, a in trees]
    tree_idx = STRtree(polys)

    for i, poly in enumerate(polys):
        candidates = tree_idx.query(poly)
        for j in candidates:
            if i < j:
                if polys[i].intersects(polys[j]):
                    inter = polys[i].intersection(polys[j])
                    if inter.area > tol:
                        return True
    return False


def compute_bounding_side(trees: List[Tuple[float, float, float]]) -> float:
    all_points = []
    for x, y, angle in trees:
        poly = get_tree_polygon(x, y, angle)
        all_points.extend(poly.exterior.coords)
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def strip_pack_n(n: int, dx_base: float = 0.6, dy_base: float = 0.7,
                 offset: float = 0.3) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Pack n trees in horizontal strips with alternating orientations.

    dx_base: Horizontal spacing within strip
    dy_base: Vertical spacing between strips
    offset: Horizontal offset for alternating strips
    """
    # Calculate grid dimensions
    trees_per_row = max(1, int(math.ceil(math.sqrt(n * dy_base / dx_base))))
    num_rows = max(1, int(math.ceil(n / trees_per_row)))

    trees = []
    idx = 0

    for row in range(num_rows):
        y = row * dy_base
        x_offset = (row % 2) * offset  # Offset alternating rows

        for col in range(trees_per_row):
            if idx >= n:
                break

            x = col * dx_base + x_offset

            # Alternate orientation based on position
            if (row + col) % 2 == 0:
                angle = 0.0    # Point up
            else:
                angle = 180.0  # Point down

            trees.append((x, y, angle))
            idx += 1

        if idx >= n:
            break

    return compute_bounding_side(trees), trees


def try_strip_parameters(n: int) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Try various strip packing parameters to find the best."""
    best_side = float('inf')
    best_trees = []

    # Try different spacings
    for dx in np.arange(0.5, 0.9, 0.05):
        for dy in np.arange(0.5, 1.0, 0.05):
            for offset in np.arange(0.0, 0.5, 0.1):
                side, trees = strip_pack_n(n, dx, dy, offset)

                if not trees_overlap(trees):
                    if side < best_side:
                        best_side = side
                        best_trees = trees

    return best_side, best_trees


def alternating_radial(n: int) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Radial arrangement with alternating orientations.
    Trees point outward in alternate rings.
    """
    if n == 0:
        return 0.0, []

    trees = []

    # First tree at center pointing up
    trees.append((0.0, 0.0, 0.0))
    if n == 1:
        return compute_bounding_side(trees), trees

    # Add trees in rings
    ring = 1
    idx = 1

    while idx < n:
        trees_in_ring = min(6 * ring, n - idx)
        radius = ring * 0.7  # Spacing between rings

        for i in range(trees_in_ring):
            if idx >= n:
                break

            theta = 2 * math.pi * i / trees_in_ring
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)

            # Alternate orientation based on ring and position
            if ring % 2 == 1:
                angle = math.degrees(theta)  # Point outward
            else:
                angle = math.degrees(theta) + 180  # Point inward

            trees.append((x, y, angle))
            idx += 1

        ring += 1

    return compute_bounding_side(trees), trees


def hexagonal_pack(n: int) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Hexagonal close-packing with varied orientations.
    """
    if n == 0:
        return 0.0, []

    # Calculate grid size
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))

    dx = 0.72  # Horizontal spacing
    dy = 0.62  # Vertical spacing (sqrt(3)/2 * dx for perfect hex)

    trees = []
    idx = 0

    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break

            x = col * dx
            y = row * dy

            # Offset odd rows for hex packing
            if row % 2 == 1:
                x += dx / 2

            # Vary angle based on position
            angle = (row * 60 + col * 30) % 360

            trees.append((x, y, angle))
            idx += 1

        if idx >= n:
            break

    return compute_bounding_side(trees), trees


def compact_local_search(trees: List[Tuple[float, float, float]],
                         max_iters: int = 1000) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Local search to compact the packing by moving trees inward.
    """
    n = len(trees)
    trees = list(trees)
    best_side = compute_bounding_side(trees)

    for _ in range(max_iters):
        improved = False

        for i in range(n):
            x, y, a = trees[i]

            # Try moving toward center of bounding box
            all_xs = [t[0] for t in trees]
            all_ys = [t[1] for t in trees]
            center_x = sum(all_xs) / n
            center_y = sum(all_ys) / n

            # Direction to center
            dx = center_x - x
            dy = center_y - y

            # Try small steps toward center
            for step in [0.1, 0.05, 0.02, 0.01]:
                new_x = x + step * np.sign(dx)
                new_y = y + step * np.sign(dy)

                new_trees = trees.copy()
                new_trees[i] = (new_x, new_y, a)

                if not trees_overlap(new_trees):
                    new_side = compute_bounding_side(new_trees)
                    if new_side < best_side:
                        trees = new_trees
                        best_side = new_side
                        improved = True
                        break

        if not improved:
            break

    return best_side, trees


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
    parser.add_argument('--n', type=int, nargs='+', default=list(range(2, 51)))
    parser.add_argument('--input', default='submission_best.csv')
    args = parser.parse_args()

    print("Gen120: Strip Packing with Alternating Orientations")
    print("=" * 55)

    groups = load_submission(args.input)

    improvements = []

    for n in args.n:
        current_trees = groups[n]
        current_side = compute_bounding_side(current_trees)

        print(f"\nn={n}: current={current_side:.4f}")

        best_side = current_side
        best_trees = current_trees

        # Try strip packing
        print("  Strip pack...", end="", flush=True)
        strip_side, strip_trees = try_strip_parameters(n)
        if strip_side < best_side and not trees_overlap(strip_trees):
            strip_side, strip_trees = compact_local_search(strip_trees)
            if strip_side < best_side:
                best_side = strip_side
                best_trees = strip_trees
                print(f" {strip_side:.4f} (better!)", end="")
        print()

        # Try hexagonal packing
        print("  Hexagonal...", end="", flush=True)
        hex_side, hex_trees = hexagonal_pack(n)
        if hex_side < best_side and not trees_overlap(hex_trees):
            hex_side, hex_trees = compact_local_search(hex_trees)
            if hex_side < best_side:
                best_side = hex_side
                best_trees = hex_trees
                print(f" {hex_side:.4f} (better!)", end="")
        print()

        # Try alternating radial
        print("  Alt. radial...", end="", flush=True)
        rad_side, rad_trees = alternating_radial(n)
        if rad_side < best_side and not trees_overlap(rad_trees):
            rad_side, rad_trees = compact_local_search(rad_trees)
            if rad_side < best_side:
                best_side = rad_side
                best_trees = rad_trees
                print(f" {rad_side:.4f} (better!)", end="")
        print()

        if best_side < current_side - 1e-6:
            score_delta = (current_side**2 - best_side**2) / n
            improvements.append((n, current_side - best_side, score_delta))
            print(f"  *** IMPROVED: {current_side:.4f} -> {best_side:.4f} (Δscore={score_delta:.4f})")

    if improvements:
        print("\n" + "="*55)
        print("Summary of improvements:")
        total = 0
        for n, imp, sd in improvements:
            print(f"  n={n}: side -{imp:.4f}, score -{sd:.4f}")
            total += sd
        print(f"  Total score improvement: {total:.4f}")
    else:
        print("\nNo improvements found.")


if __name__ == '__main__':
    main()
