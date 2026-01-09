#!/usr/bin/env python3
"""
Gen121 Starter Solution for Santa 2025 Packing.

This is a baseline solution that can be evolved by the evolve SDK.
It implements a simple greedy packing algorithm with simulated annealing refinement.

The pack(n) function returns a list of (x, y, angle) tuples for n trees.
"""

import random
import math
from typing import List, Tuple

# Tree shape vertices (15 points defining the Christmas tree polygon)
TREE_VERTICES = [
    (0.0, 0.8),    # tip
    (-0.15, 0.5),  # tier 1 left
    (-0.05, 0.5),  # tier 1 inner left
    (-0.25, 0.2),  # tier 2 left
    (-0.1, 0.2),   # tier 2 inner left
    (-0.35, -0.1), # tier 3 left
    (-0.1, -0.1),  # trunk top left
    (-0.1, -0.2),  # trunk bottom left
    (0.1, -0.2),   # trunk bottom right
    (0.1, -0.1),   # trunk top right
    (0.35, -0.1),  # tier 3 right
    (0.1, 0.2),    # tier 2 inner right
    (0.25, 0.2),   # tier 2 right
    (0.05, 0.5),   # tier 1 inner right
    (0.15, 0.5),   # tier 1 right
]

# Tree dimensions
TREE_HEIGHT = 1.0  # From -0.2 to 0.8
TREE_WIDTH = 0.7   # From -0.35 to 0.35


def transform_tree(x: float, y: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Transform tree vertices by position and rotation."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    transformed = []
    for vx, vy in TREE_VERTICES:
        rx = vx * cos_a - vy * sin_a + x
        ry = vx * sin_a + vy * cos_a + y
        transformed.append((rx, ry))

    return transformed


def get_bounding_box(trees: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box of all trees."""
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for x, y, angle in trees:
        for px, py in transform_tree(x, y, angle):
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)

    return min_x, min_y, max_x, max_y


def get_side_length(trees: List[Tuple[float, float, float]]) -> float:
    """Get the side length of bounding square."""
    min_x, min_y, max_x, max_y = get_bounding_box(trees)
    return max(max_x - min_x, max_y - min_y)


def polygons_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> bool:
    """Check if two polygons overlap using separating axis theorem."""
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1, p2 = poly[i], poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Normal to edge
            axes.append((-edge[1], edge[0]))
        return axes

    def project(poly, axis):
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    def overlap_1d(min1, max1, min2, max2):
        return max1 >= min2 and max2 >= min1

    # Check all axes from both polygons
    for poly in [poly1, poly2]:
        for axis in get_axes(poly):
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if not overlap_1d(min1, max1, min2, max2):
                return False  # Found separating axis

    return True  # No separating axis found


def has_overlap(trees: List[Tuple[float, float, float]], new_tree: Tuple[float, float, float]) -> bool:
    """Check if new_tree overlaps with any existing tree."""
    new_poly = transform_tree(*new_tree)

    for tree in trees:
        existing_poly = transform_tree(*tree)
        if polygons_overlap(new_poly, existing_poly):
            return True

    return False


def greedy_pack(n: int) -> List[Tuple[float, float, float]]:
    """
    Greedy packing: place trees one by one in a spiral pattern.

    This is the baseline - can be improved by:
    - Better placement strategies
    - Smarter angle selection
    - Post-processing optimization
    """
    if n == 0:
        return []

    if n == 1:
        # Single tree at origin, find best angle
        return [(0.0, 0.0, 45.0)]  # 45 degrees often works well

    trees = []
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 discrete angles

    # Estimate grid size based on n
    # Tree area ~0.25, so n trees need ~0.25*n area
    # With ~50% efficiency, need side ~sqrt(0.5*n)
    estimated_side = math.sqrt(n) * 0.8

    # Spiral placement
    for i in range(n):
        best_pos = None
        best_side = float('inf')

        # Try multiple positions in a grid
        step = estimated_side / max(int(math.sqrt(n)), 3)

        for attempt in range(100):
            if attempt < 50:
                # Spiral positions
                t = attempt * 0.3
                r = 0.3 * t
                x = r * math.cos(t)
                y = r * math.sin(t)
            else:
                # Random positions as fallback
                x = random.uniform(-estimated_side, estimated_side)
                y = random.uniform(-estimated_side, estimated_side)

            # Try all angles
            for angle in angles:
                candidate = (x, y, angle)

                if not trees or not has_overlap(trees, candidate):
                    test_trees = trees + [candidate]
                    side = get_side_length(test_trees)

                    if side < best_side:
                        best_side = side
                        best_pos = candidate

        if best_pos:
            trees.append(best_pos)
        else:
            # Emergency fallback - place far away
            trees.append((estimated_side + i, 0, 0))

    return trees


def simulated_annealing(trees: List[Tuple[float, float, float]], iterations: int = 1000) -> List[Tuple[float, float, float]]:
    """
    Simulated annealing refinement.

    Tries to reduce bounding box by making small moves.
    """
    if len(trees) <= 1:
        return trees

    trees = list(trees)  # Make a copy
    current_side = get_side_length(trees)

    T = 0.5  # Initial temperature
    cooling = 0.995

    for _ in range(iterations):
        # Pick a random tree
        idx = random.randint(0, len(trees) - 1)
        x, y, angle = trees[idx]

        # Make a small move
        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        dangle = random.choice([0, 0, 0, 45, -45])  # Usually keep angle, sometimes rotate

        new_tree = (x + dx, y + dy, (angle + dangle) % 360)

        # Check if valid
        other_trees = trees[:idx] + trees[idx+1:]
        if has_overlap(other_trees, new_tree):
            continue

        # Evaluate
        test_trees = other_trees + [new_tree]
        new_side = get_side_length(test_trees)

        # Accept or reject
        delta = new_side - current_side
        if delta < 0 or random.random() < math.exp(-delta / T):
            trees = test_trees
            current_side = new_side

        T *= cooling

    return trees


def pack(n: int) -> List[Tuple[float, float, float]]:
    """
    Main packing function - called by the evaluator.

    Returns list of (x, y, angle) for each of n trees.
    """
    if n == 0:
        return []

    # Greedy initial placement
    trees = greedy_pack(n)

    # SA refinement
    trees = simulated_annealing(trees, iterations=min(n * 100, 5000))

    return trees


if __name__ == "__main__":
    # Test the packing
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    trees = pack(n)

    side = get_side_length(trees)
    score = side ** 2 / n

    print(f"n={n}: side={side:.4f}, score={score:.4f}")
    for i, (x, y, angle) in enumerate(trees):
        print(f"  Tree {i}: ({x:.3f}, {y:.3f}, {angle}°)")
