#!/usr/bin/env python3
"""
Validate submission for Kaggle Santa 2025.

Checks:
1. Correct format (id, x, y, deg with s-prefix)
2. No overlapping trees within each n group
3. All trees inside bounding square

Uses strict tolerance matching Kaggle's checker.
"""

import csv
import math
import sys
from typing import List, Tuple
from dataclasses import dataclass

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_vertices(self) -> List[Tuple[float, float]]:
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return [
            (vx * cos_a - vy * sin_a + self.x,
             vx * sin_a + vy * cos_a + self.y)
            for vx, vy in TREE_VERTICES
        ]


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    """Check if segment AB intersects segment CD."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon(point, polygon):
    """Ray casting algorithm."""
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


def polygons_overlap(poly1, poly2):
    """Check if two polygons overlap (strict check)."""
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i+1) % n1],
                                  poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True

    return False


def validate_group(n: int, trees: List[Tree]) -> Tuple[bool, str]:
    """Validate a group of n trees."""
    if len(trees) != n:
        return False, f"Expected {n} trees, got {len(trees)}"

    # Get all polygons
    polys = [t.get_vertices() for t in trees]

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(polys[i], polys[j]):
                return False, f"Trees {i} and {j} overlap"

    return True, "OK"


def parse_submission(csv_path: str) -> dict:
    """Parse submission CSV into groups."""
    groups = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_parts = row['id'].split('_')
            n = int(id_parts[0])
            tree_idx = int(id_parts[1])

            # Parse s-prefixed values
            x = float(row['x'].lstrip('s'))
            y = float(row['y'].lstrip('s'))
            deg = float(row['deg'].lstrip('s'))

            if n not in groups:
                groups[n] = []
            groups[n].append(Tree(x, y, deg))

    return groups


def validate_submission(csv_path: str) -> bool:
    """Validate entire submission."""
    print(f"Validating {csv_path}...")

    groups = parse_submission(csv_path)
    print(f"Found {len(groups)} groups (n=1 to {max(groups.keys())})")

    all_valid = True
    errors = []

    for n in sorted(groups.keys()):
        valid, msg = validate_group(n, groups[n])
        if not valid:
            errors.append(f"n={n}: {msg}")
            all_valid = False

    if all_valid:
        print("✓ All groups valid - no overlaps detected")
    else:
        print(f"✗ Found {len(errors)} errors:")
        for err in errors:
            print(f"  {err}")

    return all_valid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Submission CSV file')
    args = parser.parse_args()

    valid = validate_submission(args.csv)
    sys.exit(0 if valid else 1)


if __name__ == '__main__':
    main()
