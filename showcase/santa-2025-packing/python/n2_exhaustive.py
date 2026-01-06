#!/usr/bin/env python3
"""
Exhaustive search for optimal n=2 packing.

Strategy: Fine-grained search over angles and relative positions.
For n=2, we can fix tree 0 at origin and search over tree 1's relative position.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE = Polygon(TREE_VERTICES)


def get_tree_poly(x: float, y: float, angle: float) -> Polygon:
    rotated = affinity.rotate(TREE, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_side_from_coords(coords: List[Tuple[float, float]]) -> float:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap(poly1: Polygon, poly2: Polygon, tol: float = 1e-9) -> bool:
    if not poly1.intersects(poly2):
        return False
    inter = poly1.intersection(poly2)
    return inter.area > tol


def evaluate_config(x0, y0, a0, x1, y1, a1) -> Optional[Tuple[float, Tuple]]:
    """Evaluate a single configuration. Returns (side, config) or None if invalid."""
    poly0 = get_tree_poly(x0, y0, a0)
    poly1 = get_tree_poly(x1, y1, a1)

    if has_overlap(poly0, poly1):
        return None

    coords = list(poly0.exterior.coords) + list(poly1.exterior.coords)
    side = compute_side_from_coords(coords)

    return (side, (x0, y0, a0, x1, y1, a1))


def find_touching_positions(angle0: float, angle1: float, n_dirs: int = 36) -> List[Tuple[float, float]]:
    """
    Find positions where tree1 just touches tree0 from various directions.
    Returns list of (x1, y1) positions.
    """
    poly0 = get_tree_poly(0, 0, angle0)
    positions = []

    # Try approaching from various directions
    for i in range(n_dirs):
        dir_angle = 2 * math.pi * i / n_dirs
        dx = math.cos(dir_angle)
        dy = math.sin(dir_angle)

        # Start far away and move closer
        dist = 2.0
        step = 0.05

        while dist > 0.01:
            x1 = dist * dx
            y1 = dist * dy
            poly1 = get_tree_poly(x1, y1, angle1)

            if has_overlap(poly0, poly1):
                # Back up a bit and use finer steps
                dist += step
                if step > 0.001:
                    step /= 2
                else:
                    break
            else:
                # Record this position and try closer
                positions.append((x1, y1))
                dist -= step

    return positions


def grid_search(angle0: float, angle1: float, best_so_far: float = float('inf')) -> Tuple[float, Tuple]:
    """Grid search for a specific angle pair."""
    best_side = best_so_far
    best_config = None

    # Get touching positions
    touch_positions = find_touching_positions(angle0, angle1, n_dirs=72)

    # Refine around touching positions
    for base_x, base_y in touch_positions:
        for dx in np.linspace(-0.05, 0.05, 11):
            for dy in np.linspace(-0.05, 0.05, 11):
                x1 = base_x + dx
                y1 = base_y + dy
                result = evaluate_config(0, 0, angle0, x1, y1, angle1)
                if result and result[0] < best_side:
                    best_side = result[0]
                    best_config = result[1]

    return best_side, best_config


def main():
    print("=== Exhaustive N=2 Search ===\n")

    # Current best
    current = [(0, 0, 159.55), (0.44, -0.45, 113.49)]
    poly0 = get_tree_poly(*current[0])
    poly1 = get_tree_poly(*current[1])
    coords = list(poly0.exterior.coords) + list(poly1.exterior.coords)
    current_side = compute_side_from_coords(coords)
    print(f"Current best: side={current_side:.6f}, score={current_side**2/2:.6f}")

    best_side = current_side
    best_config = None

    # Coarse angle search
    print("\nPhase 1: Coarse angle search (10° steps)...")
    angles = list(range(0, 360, 10))

    for a0 in [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90]:  # Due to symmetry, 0-90 is enough
        for a1 in angles:
            side, config = grid_search(a0, a1, best_side)
            if config and side < best_side:
                best_side = side
                best_config = config
                print(f"  New best: angles=({a0}°, {a1}°), side={side:.6f}")

    if best_config:
        print(f"\nBest from coarse search: side={best_side:.6f}")
        x0, y0, a0, x1, y1, a1 = best_config

        # Fine-tune with local search
        print("\nPhase 2: Local refinement...")
        for _ in range(3):  # Multiple refinement rounds
            improved = False
            for da0 in np.linspace(-5, 5, 21):
                for da1 in np.linspace(-5, 5, 21):
                    for dx in np.linspace(-0.02, 0.02, 9):
                        for dy in np.linspace(-0.02, 0.02, 9):
                            result = evaluate_config(x0, y0, a0 + da0, x1 + dx, y1 + dy, a1 + da1)
                            if result and result[0] < best_side - 0.0001:
                                best_side = result[0]
                                best_config = result[1]
                                x0, y0, a0, x1, y1, a1 = best_config
                                improved = True
            if improved:
                print(f"  Refined: side={best_side:.6f}")
            else:
                break

    print(f"\n=== Final Result ===")
    if best_config and best_side < current_side:
        x0, y0, a0, x1, y1, a1 = best_config
        print(f"IMPROVED!")
        print(f"  Side: {current_side:.6f} -> {best_side:.6f}")
        print(f"  Score: {current_side**2/2:.6f} -> {best_side**2/2:.6f}")
        print(f"  Delta: {(current_side**2 - best_side**2)/2:.6f}")
        print(f"\n  Tree 0: x={x0:.6f}, y={y0:.6f}, angle={a0:.6f}")
        print(f"  Tree 1: x={x1:.6f}, y={y1:.6f}, angle={a1:.6f}")

        # Center the solution
        poly0 = get_tree_poly(x0, y0, a0)
        poly1 = get_tree_poly(x1, y1, a1)
        coords = list(poly0.exterior.coords) + list(poly1.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2

        print(f"\n  Centered solution:")
        print(f"  Tree 0: x={x0-cx:.6f}, y={y0-cy:.6f}, angle={a0:.6f}")
        print(f"  Tree 1: x={x1-cx:.6f}, y={y1-cy:.6f}, angle={a1:.6f}")
    else:
        print(f"No improvement found. Current side={current_side:.6f} is optimal.")


if __name__ == '__main__':
    main()
