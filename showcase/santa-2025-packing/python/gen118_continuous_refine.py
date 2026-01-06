#!/usr/bin/env python3
"""
Gen118: Post-SA Continuous Angle Refinement

Strategy: Take the Rust solution (discrete angles) and fine-tune each tree's
angle by small continuous amounts (±10°) to minimize bounding box.

This is different from:
1. CMA-ES for small n (which optimizes x, y, angle globally)
2. Continuous angles during SA (which hurts convergence)

This approach:
1. Keeps discrete angle structure from SA (stable)
2. Applies local continuous refinement to each tree (preserves structure)
3. Uses gradient-free line search for angle optimization
"""

import math
import csv
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)

@dataclass
class Tree:
    x: float
    y: float
    angle: float  # degrees

    def get_vertices(self) -> List[Tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [(vx * cos_a - vy * sin_a + self.x,
                 vx * sin_a + vy * cos_a + self.y) for vx, vy in TREE_VERTICES]

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_side_length(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    if not trees:
        return 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for tree in trees:
        for vx, vy in tree.get_vertices():
            min_x = min(min_x, vx)
            min_y = min(min_y, vy)
            max_x = max(max_x, vx)
            max_y = max(max_y, vy)

    return max(max_x - min_x, max_y - min_y)


def ccw(A, B, C):
    """Counter-clockwise test."""
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect_strict(A, B, C, D):
    """Check if segment AB intersects segment CD (strict)."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon_strict(point, polygon):
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


def polygons_overlap_strict(poly1, poly2):
    """Check if two polygons overlap (strict segment intersection check)."""
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect_strict(poly1[i], poly1[(i+1) % n1],
                                         poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon_strict(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon_strict(v, poly1):
            return True

    return False


def has_overlap_strict(trees: List[Tree], idx: int) -> bool:
    """Check if tree at idx overlaps with any other tree (strict check)."""
    verts_idx = trees[idx].get_vertices()
    for i, other in enumerate(trees):
        if i != idx:
            if polygons_overlap_strict(verts_idx, other.get_vertices()):
                return True
    return False


def has_any_overlap_strict(trees: List[Tree]) -> bool:
    """Check if any trees overlap (strict segment intersection check with Shapely pre-filter)."""
    n = len(trees)
    if n <= 1:
        return False

    # Pre-compute all polygons and vertices
    polygons = [t.get_polygon() for t in trees]
    vertices_list = [t.get_vertices() for t in trees]

    # Use STRtree for spatial indexing
    tree = STRtree(polygons)

    for i, poly_i in enumerate(polygons):
        # Only check nearby polygons
        candidates = tree.query(poly_i)
        for j in candidates:
            if i < j:
                # Shapely fast filter: skip if no intersection at all
                if not polygons[i].intersects(polygons[j]):
                    continue
                # Strict check for candidates
                if polygons_overlap_strict(vertices_list[i], vertices_list[j]):
                    return True
    return False


def has_any_overlap_shapely(trees: List[Tree]) -> bool:
    """Quick Shapely check - use for fast filtering, then confirm with strict."""
    polygons = [t.get_polygon() for t in trees]
    tree = STRtree(polygons)

    for i, poly in enumerate(polygons):
        candidates = tree.query(poly)
        for j in candidates:
            if i < j:
                if polygons[i].intersects(polygons[j]) and polygons[i].intersection(polygons[j]).area > 1e-10:
                    return True
    return False


def optimize_tree_angle(trees: List[Tree], idx: int, max_delta: float = 10.0) -> Optional[float]:
    """
    Find the optimal angle for tree[idx] within ±max_delta degrees.
    Uses discrete search with fine steps for efficiency.

    Returns the improvement in side length (negative = better), or None if no valid improvement found.
    """
    original_tree = Tree(trees[idx].x, trees[idx].y, trees[idx].angle)
    original_side = compute_side_length(trees)

    best_angle = original_tree.angle
    best_side = original_side

    # Pre-compute other polygons (Shapely for fast check)
    other_polygons = [trees[i].get_polygon() for i in range(len(trees)) if i != idx]

    def evaluate(angle):
        test_tree = Tree(original_tree.x, original_tree.y, angle)
        test_poly = test_tree.get_polygon()

        # Fast Shapely check first
        for other_poly in other_polygons:
            if test_poly.intersects(other_poly):
                inter = test_poly.intersection(other_poly)
                # If Shapely finds significant intersection, reject
                if inter.area > 1e-12:
                    return float('inf')

        # Compute side length if Shapely says valid
        trees[idx] = test_tree
        side = compute_side_length(trees)
        return side

    # Coarse search: 2 degree steps
    for delta in range(-int(max_delta), int(max_delta) + 1, 2):
        angle = original_tree.angle + delta
        side = evaluate(angle)
        if side < best_side:
            best_side = side
            best_angle = angle

    # Fine search around best: 0.5 degree steps
    fine_start = best_angle - 2
    fine_end = best_angle + 2
    for delta_10 in range(int((fine_end - fine_start) * 2)):
        angle = fine_start + delta_10 * 0.5
        side = evaluate(angle)
        if side < best_side:
            best_side = side
            best_angle = angle

    # Restore original
    trees[idx] = original_tree

    if best_side < original_side - 1e-10:
        trees[idx] = Tree(original_tree.x, original_tree.y, best_angle)
        return original_side - best_side
    else:
        return None


def refine_group(trees: List[Tree], passes: int = 3) -> Tuple[float, List[Tree]]:
    """
    Apply continuous angle refinement to all trees in a group.

    Returns: (improvement, refined_trees)
    """
    original_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    original_side = compute_side_length(trees)

    for p in range(passes):
        any_improvement = False

        # Process trees from outermost to innermost (boundary trees first)
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for t in trees:
            for vx, vy in t.get_vertices():
                min_x, min_y = min(min_x, vx), min(min_y, vy)
                max_x, max_y = max(max_x, vx), max(max_y, vy)

        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

        # Sort by distance from center (outermost first)
        indices = list(range(len(trees)))
        indices.sort(key=lambda i: -((trees[i].x - cx)**2 + (trees[i].y - cy)**2))

        for idx in indices:
            improvement = optimize_tree_angle(trees, idx, max_delta=10.0)
            if improvement is not None and improvement > 0:
                any_improvement = True

        if not any_improvement:
            break

    final_side = compute_side_length(trees)

    # Strict validation at end: check for any overlaps
    if has_any_overlap_strict(trees):
        # Revert to original if overlaps found
        return 0.0, original_trees

    return original_side - final_side, trees


def load_submission(csv_path: str) -> dict:
    """Load submission and return groups dictionary."""
    groups = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'][1:])  # Remove 's' prefix
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            groups[n].append(Tree(x, y, deg))

    return groups


def save_submission(groups: dict, csv_path: str):
    """Save groups to submission CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])

        for n in range(1, 201):
            for idx, tree in enumerate(groups[n]):
                row_id = f'{n:03d}_{idx}'
                writer.writerow([row_id, f's{tree.x}', f's{tree.y}', f's{tree.angle}'])


def main():
    parser = argparse.ArgumentParser(description='Gen118: Continuous angle refinement')
    parser.add_argument('--input', default='submission_best.csv', help='Input submission CSV')
    parser.add_argument('--output', default='submission_gen118.csv', help='Output submission CSV')
    parser.add_argument('--n-start', type=int, default=11, help='Start n value')
    parser.add_argument('--n-end', type=int, default=200, help='End n value')
    parser.add_argument('--passes', type=int, default=3, help='Refinement passes per group')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    groups = load_submission(args.input)

    # Compute original score
    original_score = sum(compute_side_length(groups[n])**2 / n for n in range(1, 201))
    print(f"Original score: {original_score:.4f}")

    total_improvement = 0
    improved_count = 0

    for n in range(args.n_start, args.n_end + 1):
        trees = groups[n]
        original_side = compute_side_length(trees)

        improvement, refined = refine_group(trees, passes=args.passes)

        if improvement > 1e-6:
            groups[n] = refined
            new_side = compute_side_length(refined)
            score_delta = (original_side**2 - new_side**2) / n
            total_improvement += score_delta
            improved_count += 1
            print(f"  n={n}: {original_side:.6f} -> {new_side:.6f} (Δside={improvement:.6f}, Δscore={score_delta:.6f})")

    # Compute final score
    final_score = sum(compute_side_length(groups[n])**2 / n for n in range(1, 201))

    print(f"\nSummary:")
    print(f"  Groups improved: {improved_count}")
    print(f"  Score: {original_score:.4f} -> {final_score:.4f} (Δ={original_score - final_score:.4f})")

    if final_score < original_score - 1e-6:
        # Validate no overlaps
        print("\nValidating solution...")
        valid = True
        for n in range(1, 201):
            if has_any_overlap_strict(groups[n]):
                print(f"  ERROR: Overlaps in n={n}!")
                valid = False

        if valid:
            save_submission(groups, args.output)
            print(f"\nSaved improved solution to {args.output}")
        else:
            print("\nNot saving due to overlaps!")
    else:
        print("\nNo improvements found.")


if __name__ == '__main__':
    main()
