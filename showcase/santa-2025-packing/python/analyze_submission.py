#!/usr/bin/env python3
"""
Analyze submission CSV to get side lengths per n.

Submission format: row_id,x,y,angle
Trees are listed consecutively - first tree for n=1, then 2 trees for n=2, etc.
"""

import csv
import sys
import math
import numpy as np
from pathlib import Path

# Tree polygon vertices (same as Rust)
TREE_VERTICES = np.array([
    (0.0, 0.8),      # Tip
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
])


def get_tree_vertices(x: float, y: float, angle_deg: float) -> np.ndarray:
    """Get tree vertices at given position and rotation."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rotate and translate
    rotated = np.zeros_like(TREE_VERTICES)
    rotated[:, 0] = TREE_VERTICES[:, 0] * cos_a - TREE_VERTICES[:, 1] * sin_a + x
    rotated[:, 1] = TREE_VERTICES[:, 0] * sin_a + TREE_VERTICES[:, 1] * cos_a + y

    return rotated


def compute_side_from_trees(trees: list) -> float:
    """Compute bounding square side length from list of (x, y, angle) tuples."""
    all_points = []
    for x, y, angle in trees:
        vertices = get_tree_vertices(x, y, angle)
        all_points.append(vertices)

    all_points = np.vstack(all_points)
    minx, miny = all_points.min(axis=0)
    maxx, maxy = all_points.max(axis=0)
    return max(maxx - minx, maxy - miny)


def analyze_submission(csv_path: str, max_n: int = 200):
    """Analyze submission and print side lengths for each n."""
    trees = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            angle = float(row['angle'])
            trees.append((x, y, angle))

    print(f"\nAnalyzing {csv_path}")
    print(f"Total trees: {len(trees)}")
    print(f"\n{'n':>4} | {'Side':>10} | {'Score (s²/n)':>12}")
    print("-" * 35)

    total_score = 0.0
    tree_idx = 0

    for n in range(1, max_n + 1):
        if tree_idx + n > len(trees):
            print(f"\n[Incomplete: only {len(trees)} trees, expected {tree_idx + n}]")
            break

        # Get trees for this n
        n_trees = trees[tree_idx:tree_idx + n]
        tree_idx += n

        side = compute_side_from_trees(n_trees)
        score = side**2 / n
        total_score += score

        if n <= 20 or n % 10 == 0 or n >= max_n - 5:
            print(f"{n:>4} | {side:>10.4f} | {score:>12.4f}")

    print("-" * 35)
    print(f"\nTotal score (n=1..{n}): {total_score:.4f}")
    return total_score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Submission CSV file')
    parser.add_argument('--max-n', type=int, default=200, help='Max n to analyze')
    args = parser.parse_args()

    analyze_submission(args.csv, args.max_n)


if __name__ == '__main__':
    main()
