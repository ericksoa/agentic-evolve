#!/usr/bin/env python3.12
"""
Constraint Programming solver for Santa 2025 tree packing.
Uses OR-Tools CP-SAT for small n values.
"""

import math
from typing import List, Tuple, Optional
import numpy as np

# Tree vertices (centered at origin)
TREE_VERTICES = [
    (0.0, 0.8),      # Tip
    (0.125, 0.5),    # top_w/2
    (0.0625, 0.5),   # top_w/4
    (0.2, 0.25),     # mid_w/2
    (0.1, 0.25),     # mid_w/4
    (0.35, 0.0),     # base_w/2
    (0.075, 0.0),    # trunk_w/2
    (0.075, -0.2),   # trunk_w/2, trunk_bottom
    (-0.075, -0.2),  # -trunk_w/2, trunk_bottom
    (-0.075, 0.0),   # -trunk_w/2
    (-0.35, 0.0),    # -base_w/2
    (-0.1, 0.25),    # -mid_w/4
    (-0.2, 0.25),    # -mid_w/2
    (-0.0625, 0.5),  # -top_w/4
    (-0.125, 0.5),   # -top_w/2
]

def rotate_point(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    """Rotate point around origin by angle_deg degrees."""
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

def get_rotated_vertices(angle_deg: float) -> List[Tuple[float, float]]:
    """Get tree vertices rotated by angle_deg."""
    return [rotate_point(x, y, angle_deg) for x, y in TREE_VERTICES]

def get_bounding_box(vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box (min_x, min_y, max_x, max_y) of vertices."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))

def polygons_overlap_sat(poly1: List[Tuple[float, float]],
                          poly2: List[Tuple[float, float]],
                          margin: float = 1e-6) -> bool:
    """Check if two polygons overlap using Separating Axis Theorem."""
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Normal to edge
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-10:
                axes.append((normal[0]/length, normal[1]/length))
        return axes

    def project(poly, axis):
        dots = [v[0] * axis[0] + v[1] * axis[1] for v in poly]
        return min(dots), max(dots)

    for axis in get_axes(poly1) + get_axes(poly2):
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        # Check for gap (with margin)
        if max1 < min2 - margin or max2 < min1 - margin:
            return False  # Separating axis found, no overlap
    return True  # No separating axis, polygons overlap

def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def check_trees_overlap(x1: float, y1: float, rot1: int,
                        x2: float, y2: float, rot2: int,
                        margin: float = 1e-5) -> bool:
    """Check if two trees overlap."""
    angle1 = rot1 * 45.0
    angle2 = rot2 * 45.0

    verts1 = [(vx + x1, vy + y1) for vx, vy in get_rotated_vertices(angle1)]
    verts2 = [(vx + x2, vy + y2) for vx, vy in get_rotated_vertices(angle2)]

    return polygons_overlap_sat(verts1, verts2, margin)

def compute_side_length(trees: List[Tuple[float, float, int]]) -> float:
    """Compute the side length needed to contain all trees."""
    all_vertices = []
    for x, y, rot in trees:
        angle = rot * 45.0
        for vx, vy in get_rotated_vertices(angle):
            all_vertices.append((vx + x, vy + y))

    if not all_vertices:
        return 0.0

    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)

    return max(max_x - min_x, max_y - min_y)

def brute_force_small_n(n: int, grid_resolution: float = 0.05) -> Tuple[float, List[Tuple[float, float, int]]]:
    """
    Brute force search for small n (n=1,2).
    Returns (side_length, [(x, y, rotation), ...])
    """
    if n == 1:
        # For n=1, find the optimal rotation (45 degrees gives smallest bbox)
        best_side = float('inf')
        best_config = None
        for rot in range(8):
            angle = rot * 45.0
            verts = get_rotated_vertices(angle)
            min_x, min_y, max_x, max_y = get_bounding_box(verts)
            side = max(max_x - min_x, max_y - min_y)
            if side < best_side:
                best_side = side
                # Center the tree
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                best_config = [(-cx + side/2, -cy + side/2, rot)]
        return best_side, best_config

    if n == 2:
        # Try all rotation combinations and positions
        best_side = float('inf')
        best_config = None

        # Precompute rotated vertices for each rotation
        rotated = [get_rotated_vertices(r * 45.0) for r in range(8)]

        # Try all rotation pairs
        for rot1 in range(8):
            for rot2 in range(8):
                # Get bounding boxes
                bbox1 = get_bounding_box(rotated[rot1])
                bbox2 = get_bounding_box(rotated[rot2])

                # Try placing trees in various configurations
                # Side by side horizontally
                for offset_y in np.arange(-1.0, 1.0, grid_resolution):
                    # Place tree1 at origin, tree2 to the right
                    x1, y1 = 0.0, 0.0

                    # Binary search for minimum x offset
                    lo, hi = 0.5, 2.0
                    while hi - lo > 0.001:
                        mid = (lo + hi) / 2
                        x2, y2 = mid, offset_y
                        if check_trees_overlap(x1, y1, rot1, x2, y2, rot2):
                            lo = mid
                        else:
                            hi = mid

                    x2, y2 = hi, offset_y
                    if not check_trees_overlap(x1, y1, rot1, x2, y2, rot2):
                        side = compute_side_length([(x1, y1, rot1), (x2, y2, rot2)])
                        if side < best_side:
                            best_side = side
                            best_config = [(x1, y1, rot1), (x2, y2, rot2)]

                # Try vertically
                for offset_x in np.arange(-1.0, 1.0, grid_resolution):
                    x1, y1 = 0.0, 0.0

                    lo, hi = 0.5, 2.0
                    while hi - lo > 0.001:
                        mid = (lo + hi) / 2
                        x2, y2 = offset_x, mid
                        if check_trees_overlap(x1, y1, rot1, x2, y2, rot2):
                            lo = mid
                        else:
                            hi = mid

                    x2, y2 = offset_x, hi
                    if not check_trees_overlap(x1, y1, rot1, x2, y2, rot2):
                        side = compute_side_length([(x1, y1, rot1), (x2, y2, rot2)])
                        if side < best_side:
                            best_side = side
                            best_config = [(x1, y1, rot1), (x2, y2, rot2)]

        return best_side, best_config

    return float('inf'), None

def evolved_greedy_pack(n: int, seed: int = 42) -> Tuple[float, List[Tuple[float, float, int]]]:
    """
    Simple greedy packing (Python reimplementation for comparison).
    """
    import random
    random.seed(seed)

    trees = []
    for _ in range(n):
        best_placement = None
        best_side = float('inf')

        if not trees:
            # First tree: use optimal rotation
            side, config = brute_force_small_n(1)
            return side if n == 1 else None, config if n == 1 else None

        # Try to place near existing trees
        for attempt in range(100):
            # Random position near existing trees
            ref_tree = random.choice(trees)
            x = ref_tree[0] + random.uniform(-1.5, 1.5)
            y = ref_tree[1] + random.uniform(-1.5, 1.5)
            rot = random.randint(0, 7)

            # Check overlap with existing trees
            overlaps = False
            for tx, ty, tr in trees:
                if check_trees_overlap(x, y, rot, tx, ty, tr):
                    overlaps = True
                    break

            if not overlaps:
                test_trees = trees + [(x, y, rot)]
                side = compute_side_length(test_trees)
                if side < best_side:
                    best_side = side
                    best_placement = (x, y, rot)

        if best_placement:
            trees.append(best_placement)
        else:
            # Fallback: place far away
            x = 10.0 + len(trees)
            trees.append((x, 0.0, 0))

    return compute_side_length(trees), trees

def main():
    print("Testing small n values:")
    print()

    # Test n=1
    side1, config1 = brute_force_small_n(1)
    print(f"n=1: side_length = {side1:.6f}")
    if config1:
        x, y, rot = config1[0]
        print(f"     rotation = {rot * 45}°")
    print()

    # Test n=2
    side2, config2 = brute_force_small_n(2)
    print(f"n=2: side_length = {side2:.6f}")
    if config2:
        for i, (x, y, rot) in enumerate(config2):
            print(f"     tree {i+1}: pos=({x:.4f}, {y:.4f}), rot={rot * 45}°")
    print()

    # For reference, tree at 45° has bounding box of ~0.813 x 0.813
    print("Reference:")
    print("  Tree at 0°:  bbox width=0.70, height=1.00")
    print("  Tree at 45°: bbox width≈0.813, height≈0.813")

if __name__ == "__main__":
    main()
