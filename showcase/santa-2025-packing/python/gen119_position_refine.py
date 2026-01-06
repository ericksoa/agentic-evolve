#!/usr/bin/env python3
"""
Gen119: Combined Position + Angle Refinement

Strategy: Take the existing solution and refine each tree's position AND angle
together to minimize bounding box. This extends Gen118 (angle-only) with
position perturbation.

For each tree, search:
- Position: ±0.05 in x and y (coarse 0.02 steps, then fine 0.005 steps)
- Angle: ±10° (coarse 2° steps, then fine 0.5° steps)

Optimization strategy:
1. Boundary trees first (they define the bounding box edge)
2. Greedy acceptance: accept any improvement immediately
3. Multiple passes until no improvement
4. Strict validation (segment-intersection)
"""

import math
import csv
import sys
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


def compute_bounding_box(trees: List[Tree]) -> Tuple[float, float, float, float]:
    """Compute bounding box (min_x, min_y, max_x, max_y)."""
    if not trees:
        return 0.0, 0.0, 0.0, 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for tree in trees:
        for vx, vy in tree.get_vertices():
            min_x = min(min_x, vx)
            min_y = min(min_y, vy)
            max_x = max(max_x, vx)
            max_y = max(max_y, vy)

    return min_x, min_y, max_x, max_y


def compute_side_length(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    min_x, min_y, max_x, max_y = compute_bounding_box(trees)
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


def tree_overlaps_others_fast(test_tree: Tree, other_polygons: List[Polygon]) -> bool:
    """Fast Shapely check if tree overlaps any other polygon."""
    test_poly = test_tree.get_polygon()
    for other_poly in other_polygons:
        if test_poly.intersects(other_poly):
            inter = test_poly.intersection(other_poly)
            if inter.area > 1e-12:
                return True
    return False


def tree_overlaps_others_strict(test_tree: Tree, trees: List[Tree], idx: int) -> bool:
    """Strict segment-intersection check."""
    test_verts = test_tree.get_vertices()
    for i, other in enumerate(trees):
        if i != idx:
            if polygons_overlap_strict(test_verts, other.get_vertices()):
                return True
    return False


def get_boundary_tree_indices(trees: List[Tree]) -> List[int]:
    """Get indices of trees that define the bounding box edges."""
    if not trees:
        return []

    min_x, min_y, max_x, max_y = compute_bounding_box(trees)
    boundary_indices = set()
    eps = 1e-9

    for i, tree in enumerate(trees):
        for vx, vy in tree.get_vertices():
            if abs(vx - min_x) < eps or abs(vx - max_x) < eps or \
               abs(vy - min_y) < eps or abs(vy - max_y) < eps:
                boundary_indices.add(i)
                break

    return list(boundary_indices)


def optimize_tree_position_angle(trees: List[Tree], idx: int,
                                  pos_delta: float = 0.05,
                                  angle_delta: float = 10.0,
                                  coarse_pos_step: float = 0.02,
                                  fine_pos_step: float = 0.005,
                                  coarse_angle_step: float = 2.0,
                                  fine_angle_step: float = 0.5) -> Optional[float]:
    """
    Find the optimal position and angle for tree[idx].
    Uses coarse-to-fine search for efficiency.

    Returns the improvement in side length, or None if no valid improvement found.
    """
    original_tree = Tree(trees[idx].x, trees[idx].y, trees[idx].angle)
    original_side = compute_side_length(trees)

    best_x = original_tree.x
    best_y = original_tree.y
    best_angle = original_tree.angle
    best_side = original_side

    # Pre-compute other polygons for fast overlap check
    other_polygons = [trees[i].get_polygon() for i in range(len(trees)) if i != idx]

    def evaluate(x, y, angle):
        """Evaluate a candidate position/angle. Returns side length or inf if invalid."""
        test_tree = Tree(x, y, angle)

        # Fast Shapely check first
        if tree_overlaps_others_fast(test_tree, other_polygons):
            return float('inf')

        # Compute side length if no overlap
        trees[idx] = test_tree
        side = compute_side_length(trees)
        trees[idx] = original_tree  # Restore
        return side

    # Phase 1: Coarse search
    coarse_pos_range = int(pos_delta / coarse_pos_step)
    coarse_angle_range = int(angle_delta / coarse_angle_step)

    for dx in range(-coarse_pos_range, coarse_pos_range + 1):
        for dy in range(-coarse_pos_range, coarse_pos_range + 1):
            for da in range(-coarse_angle_range, coarse_angle_range + 1):
                x = original_tree.x + dx * coarse_pos_step
                y = original_tree.y + dy * coarse_pos_step
                angle = original_tree.angle + da * coarse_angle_step

                side = evaluate(x, y, angle)
                if side < best_side:
                    best_side = side
                    best_x, best_y, best_angle = x, y, angle

    # Phase 2: Fine search around best
    if best_side < original_side - 1e-10:
        fine_pos_range = int(coarse_pos_step / fine_pos_step)
        fine_angle_range = int(coarse_angle_step / fine_angle_step)

        center_x, center_y, center_angle = best_x, best_y, best_angle

        for dx in range(-fine_pos_range, fine_pos_range + 1):
            for dy in range(-fine_pos_range, fine_pos_range + 1):
                for da in range(-fine_angle_range, fine_angle_range + 1):
                    x = center_x + dx * fine_pos_step
                    y = center_y + dy * fine_pos_step
                    angle = center_angle + da * fine_angle_step

                    side = evaluate(x, y, angle)
                    if side < best_side:
                        best_side = side
                        best_x, best_y, best_angle = x, y, angle

    # Apply best if improved
    if best_side < original_side - 1e-10:
        trees[idx] = Tree(best_x, best_y, best_angle)
        return original_side - best_side
    else:
        return None


def refine_group(trees: List[Tree], passes: int = 3, verbose: bool = False) -> Tuple[float, List[Tree]]:
    """
    Apply combined position + angle refinement to all trees in a group.

    Returns: (improvement, refined_trees)
    """
    original_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    original_side = compute_side_length(trees)

    for p in range(passes):
        any_improvement = False

        # Get boundary trees first
        boundary_indices = get_boundary_tree_indices(trees)
        other_indices = [i for i in range(len(trees)) if i not in boundary_indices]

        # Process boundary trees first, then others
        indices = boundary_indices + other_indices

        for idx in indices:
            improvement = optimize_tree_position_angle(trees, idx)
            if improvement is not None and improvement > 0:
                any_improvement = True
                if verbose:
                    print(f"    Tree {idx}: improved by {improvement:.6f}")

        if not any_improvement:
            break

    final_side = compute_side_length(trees)

    # Strict validation at end
    if has_any_overlap_strict(trees):
        if verbose:
            print("    WARNING: Overlaps found after refinement, reverting")
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


def optimize_tree_position_angle_strict(trees: List[Tree], idx: int,
                                         pos_delta: float = 0.03,
                                         angle_delta: float = 5.0,
                                         pos_step: float = 0.015,
                                         angle_step: float = 2.5) -> Optional[float]:
    """
    Simpler search with strict validation - for large n.
    """
    original_tree = Tree(trees[idx].x, trees[idx].y, trees[idx].angle)
    original_side = compute_side_length(trees)

    best_x = original_tree.x
    best_y = original_tree.y
    best_angle = original_tree.angle
    best_side = original_side

    # Pre-compute other polygons for fast check
    other_polygons = [trees[i].get_polygon() for i in range(len(trees)) if i != idx]

    def evaluate(x, y, angle):
        test_tree = Tree(x, y, angle)

        # Fast Shapely check
        test_poly = test_tree.get_polygon()
        for other_poly in other_polygons:
            if test_poly.intersects(other_poly):
                inter = test_poly.intersection(other_poly)
                if inter.area > 1e-12:
                    return float('inf')

        # Compute side length
        trees[idx] = test_tree
        side = compute_side_length(trees)
        trees[idx] = original_tree
        return side

    pos_range = int(pos_delta / pos_step)
    angle_range = int(angle_delta / angle_step)

    for dx in range(-pos_range, pos_range + 1):
        for dy in range(-pos_range, pos_range + 1):
            for da in range(-angle_range, angle_range + 1):
                x = original_tree.x + dx * pos_step
                y = original_tree.y + dy * pos_step
                angle = original_tree.angle + da * angle_step

                side = evaluate(x, y, angle)
                if side < best_side:
                    best_side = side
                    best_x, best_y, best_angle = x, y, angle

    # Apply best if improved - but verify with strict check
    if best_side < original_side - 1e-10:
        candidate = Tree(best_x, best_y, best_angle)
        trees[idx] = candidate

        # Strict validation of this specific tree
        if tree_overlaps_others_strict(candidate, trees, idx):
            trees[idx] = original_tree
            return None

        return original_side - best_side
    else:
        return None


def refine_group_fast(trees: List[Tree], passes: int = 2) -> Tuple[float, List[Tree]]:
    """
    Faster version for large n - coarser search grid with strict validation.
    """
    original_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    original_side = compute_side_length(trees)

    for p in range(passes):
        any_improvement = False
        boundary_indices = get_boundary_tree_indices(trees)
        other_indices = [i for i in range(len(trees)) if i not in boundary_indices]
        indices = boundary_indices + other_indices

        for idx in indices:
            # Use strict validation version
            improvement = optimize_tree_position_angle_strict(
                trees, idx,
                pos_delta=0.03,
                angle_delta=5.0,
                pos_step=0.015,
                angle_step=2.5
            )
            if improvement is not None and improvement > 0:
                any_improvement = True

        if not any_improvement:
            break

    final_side = compute_side_length(trees)

    # Final strict validation
    if has_any_overlap_strict(trees):
        return 0.0, original_trees

    return original_side - final_side, trees


def main():
    parser = argparse.ArgumentParser(description='Gen119: Combined position + angle refinement')
    parser.add_argument('--input', default='submission_best.csv', help='Input submission CSV')
    parser.add_argument('--output', default='submission_gen119.csv', help='Output submission CSV')
    parser.add_argument('--n-start', type=int, default=2, help='Start n value')
    parser.add_argument('--n-end', type=int, default=200, help='End n value')
    parser.add_argument('--passes', type=int, default=3, help='Refinement passes per group')
    parser.add_argument('--pos-delta', type=float, default=0.05, help='Position search range')
    parser.add_argument('--angle-delta', type=float, default=10.0, help='Angle search range')
    parser.add_argument('--fast-threshold', type=int, default=100, help='Use fast mode for n above this')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
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

        print(f"Processing n={n} ({n} trees)...", end='', flush=True)

        # Use fast mode for large n
        if n > args.fast_threshold:
            improvement, refined = refine_group_fast(trees, passes=2)
        else:
            improvement, refined = refine_group(trees, passes=args.passes, verbose=args.verbose)

        if improvement > 1e-6:
            groups[n] = refined
            new_side = compute_side_length(refined)
            score_delta = (original_side**2 - new_side**2) / n
            total_improvement += score_delta
            improved_count += 1
            print(f" IMPROVED: {original_side:.6f} -> {new_side:.6f} (Δscore={score_delta:.6f})")
        else:
            print(" no improvement")

    # Compute final score
    final_score = sum(compute_side_length(groups[n])**2 / n for n in range(1, 201))

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Groups improved: {improved_count}")
    print(f"  Score: {original_score:.4f} -> {final_score:.4f} (Δ={original_score - final_score:.4f})")

    if final_score < original_score - 1e-6:
        # Full validation
        print("\nValidating solution...")
        valid = True
        for n in range(1, 201):
            if has_any_overlap_strict(groups[n]):
                print(f"  ERROR: Overlaps in n={n}!")
                valid = False

        if valid:
            print("  All groups valid!")
            save_submission(groups, args.output)
            print(f"\nSaved improved solution to {args.output}")
        else:
            print("\nNot saving due to overlaps!")
    else:
        print("\nNo improvements found.")


if __name__ == '__main__':
    main()
