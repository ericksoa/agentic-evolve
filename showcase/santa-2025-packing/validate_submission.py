#!/usr/bin/env python3
"""Validate submission CSV using shapely (like Kaggle does)."""

import sys
import math
import csv
from itertools import combinations
from shapely.geometry import Polygon
from shapely.validation import make_valid

TREE_VERTICES = [
    (0.0, 0.8),
    (0.125, 0.5), (0.0625, 0.5),
    (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0), (0.075, -0.2),
    (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25), (-0.2, 0.25),
    (-0.0625, 0.5), (-0.125, 0.5),
]

def rotate_and_translate(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    verts = []
    for vx, vy in TREE_VERTICES:
        rx = vx * cos_a - vy * sin_a
        ry = vx * sin_a + vy * cos_a
        verts.append((rx + x, ry + y))
    return verts

def parse_value(s):
    if s.startswith('s'):
        return float(s[1:])
    return float(s)

def validate_submission(csv_path, verbose=False):
    """Validate submission and return (is_valid, problems)."""
    # Read all groups from submission
    groups = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = row['id'].split('_')
            n = int(parts[0])
            idx = int(parts[1])
            x = parse_value(row['x'])
            y = parse_value(row['y'])
            deg = parse_value(row['deg'])
            if n not in groups:
                groups[n] = []
            groups[n].append((idx, x, y, deg))

    print(f"Loaded {len(groups)} groups from {csv_path}")

    # Check each group
    problems = []
    close_pairs_count = 0

    for n in sorted(groups.keys()):
        trees = sorted(groups[n], key=lambda t: t[0])

        # Check tree count
        if len(trees) != n:
            problems.append(f"Group {n}: expected {n} trees, got {len(trees)}")
            continue

        polygons = []
        for i, x, y, deg in trees:
            verts = rotate_and_translate(x, y, deg)
            poly = Polygon(verts)
            if not poly.is_valid:
                poly = make_valid(poly)
            polygons.append((i, poly))

        # Check for overlaps
        for (i, poly1), (j, poly2) in combinations(polygons, 2):
            if poly1.intersects(poly2):
                intersection = poly1.intersection(poly2)
                area = intersection.area if hasattr(intersection, 'area') else 0
                if area > 0:
                    problems.append(f"Group {n}: trees {i} and {j} overlap (area={area:.2e})")

            # Count close pairs
            if verbose:
                dist = poly1.distance(poly2)
                if dist < 1e-5:
                    close_pairs_count += 1

    if verbose and close_pairs_count > 0:
        print(f"  Close pairs (dist < 1e-5): {close_pairs_count}")

    return len(problems) == 0, problems

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission.csv> [-v]")
        sys.exit(1)

    csv_path = sys.argv[1]
    verbose = '-v' in sys.argv

    is_valid, problems = validate_submission(csv_path, verbose)

    if is_valid:
        print("\n✓ Submission is VALID - no overlaps detected!")
    else:
        print(f"\n✗ Found {len(problems)} problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
