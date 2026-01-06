#!/usr/bin/env python3
"""
Create hybrid submission by taking the best solution for each n
from multiple sources (Rust, Python optimizer, etc.)
"""

import csv
import json
import math
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely import affinity

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
    """Check overlap using Shapely."""
    polys = [t.get_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def has_overlap_strict(trees: List[Tree]) -> bool:
    """Strict overlap check using segment intersection (matches Kaggle)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        d1, d2 = ccw(A, B, C), ccw(A, B, D)
        d3, d4 = ccw(C, D, A), ccw(C, D, B)
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

    def get_vertices(tree):
        import math
        angle_rad = math.radians(tree.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        VERTS = [(0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
                 (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
                 (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5)]
        return [(vx * cos_a - vy * sin_a + tree.x, vx * sin_a + vy * cos_a + tree.y) for vx, vy in VERTS]

    polys = [get_vertices(t) for t in trees]
    n = len(polys)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = polys[i], polys[j]
            n1, n2 = len(p1), len(p2)
            # Check edge intersections
            for ei in range(n1):
                for ej in range(n2):
                    if segments_intersect(p1[ei], p1[(ei+1) % n1], p2[ej], p2[(ej+1) % n2]):
                        return True
            # Check containment
            for v in p1:
                if point_in_polygon(v, p2):
                    return True
            for v in p2:
                if point_in_polygon(v, p1):
                    return True
    return False


def load_submission_csv(csv_path: str) -> Dict[int, List[Tree]]:
    """Load all solutions from submission CSV."""
    trees = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle 's' prefix and both 'deg'/'angle' column names
            x_val = row.get('x', '')
            y_val = row.get('y', '')
            deg_val = row.get('deg', row.get('angle', ''))
            # Strip 's' prefix if present
            x = float(x_val.replace('s', '')) if isinstance(x_val, str) else float(x_val)
            y = float(y_val.replace('s', '')) if isinstance(y_val, str) else float(y_val)
            deg = float(deg_val.replace('s', '')) if isinstance(deg_val, str) else float(deg_val)
            trees.append(Tree(x, y, deg))

    # Group by n
    solutions = {}
    tree_idx = 0
    for n in range(1, 201):
        if tree_idx + n > len(trees):
            break
        solutions[n] = trees[tree_idx:tree_idx + n]
        tree_idx += n

    return solutions


def load_optimized_json(json_path: str) -> Dict[int, List[Tree]]:
    """Load optimized solutions from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    solutions = {}
    for n_str, info in data.items():
        n = int(n_str)
        trees = [Tree(x, y, a) for x, y, a in info['trees']]
        solutions[n] = trees

    return solutions


def merge_best_solutions(
    rust_solutions: Dict[int, List[Tree]],
    python_solutions: Dict[int, List[Tree]],
    max_n: int = 200
) -> Tuple[Dict[int, List[Tree]], Dict[int, str]]:
    """Merge solutions, taking best for each n."""
    best_solutions = {}
    sources = {}

    for n in range(1, max_n + 1):
        candidates = []

        if n in rust_solutions:
            rust_trees = rust_solutions[n]
            if not has_overlap(rust_trees):
                rust_side = compute_side(rust_trees)
                candidates.append(('rust', rust_side, rust_trees))

        if n in python_solutions:
            py_trees = python_solutions[n]
            # Use strict validation for Python solutions
            if not has_overlap_strict(py_trees):
                py_side = compute_side(py_trees)
                candidates.append(('python', py_side, py_trees))
            else:
                print(f"  Warning: Python solution for n={n} has overlaps, skipping")

        if candidates:
            # Take best
            candidates.sort(key=lambda x: x[1])
            source, side, trees = candidates[0]
            best_solutions[n] = trees
            sources[n] = source

    return best_solutions, sources


def write_submission_csv(solutions: Dict[int, List[Tree]], output_path: str):
    """Write solutions to submission CSV format.

    Format: id is {n:03d}_{tree_idx}, values prefixed with 's'
    Example: 001_0,s0.0,s0.0,s90.0
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])

        for n in range(1, max(solutions.keys()) + 1):
            if n not in solutions:
                print(f"Warning: missing n={n}")
                continue

            for tree_idx, tree in enumerate(solutions[n]):
                row_id = f"{n:03d}_{tree_idx}"
                writer.writerow([row_id, f"s{tree.x}", f"s{tree.y}", f"s{tree.angle}"])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rust-csv', type=str, required=True, help='Rust submission CSV')
    parser.add_argument('--python-json', type=str, help='Python optimized JSON')
    parser.add_argument('--output', type=str, required=True, help='Output CSV')

    args = parser.parse_args()

    print("Loading Rust solutions...")
    rust_solutions = load_submission_csv(args.rust_csv)
    print(f"  Loaded {len(rust_solutions)} n values")

    python_solutions = {}
    if args.python_json:
        print("Loading Python solutions...")
        python_solutions = load_optimized_json(args.python_json)
        print(f"  Loaded {len(python_solutions)} n values")

    print("\nMerging solutions...")
    best_solutions, sources = merge_best_solutions(rust_solutions, python_solutions)

    # Calculate scores
    print("\nResults:")
    print(f"{'n':>4} | {'Side':>10} | {'Score':>10} | {'Source':>8}")
    print("-" * 45)

    total_score = 0.0
    rust_count = 0
    python_count = 0

    for n in range(1, max(best_solutions.keys()) + 1):
        if n not in best_solutions:
            continue

        side = compute_side(best_solutions[n])
        score = side**2 / n
        total_score += score
        source = sources[n]

        if source == 'rust':
            rust_count += 1
        else:
            python_count += 1

        if n <= 15 or n % 50 == 0:
            print(f"{n:>4} | {side:>10.4f} | {score:>10.4f} | {source:>8}")

    print("-" * 45)
    print(f"Total score: {total_score:.4f}")
    print(f"Rust best: {rust_count}, Python best: {python_count}")

    # Write output
    write_submission_csv(best_solutions, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
