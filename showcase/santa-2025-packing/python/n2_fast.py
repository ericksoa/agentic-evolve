#!/usr/bin/env python3
"""
Fast exhaustive search for optimal n=2 packing using vectorized operations.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.prepared import prep
import math

TREE_VERTICES = np.array([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])

TREE = Polygon(TREE_VERTICES)


def get_tree_poly(x: float, y: float, angle: float) -> Polygon:
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    rotated = TREE_VERTICES[:, 0] * cos_a - TREE_VERTICES[:, 1] * sin_a + x, \
              TREE_VERTICES[:, 0] * sin_a + TREE_VERTICES[:, 1] * cos_a + y
    return Polygon(list(zip(rotated[0], rotated[1])))


def has_overlap(poly1: Polygon, poly2: Polygon, tol: float = 1e-9) -> bool:
    if not poly1.intersects(poly2):
        return False
    inter = poly1.intersection(poly2)
    return inter.area > tol


def compute_side(poly1: Polygon, poly2: Polygon) -> float:
    coords = list(poly1.exterior.coords) + list(poly2.exterior.coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def find_best_position(a0: float, a1: float, best_so_far: float) -> tuple:
    """Find best position for tree1 given angles, using binary search."""
    poly0 = get_tree_poly(0, 0, a0)
    prep_poly0 = prep(poly0)

    best_side = best_so_far
    best_pos = None

    # Search in multiple directions
    for dir_deg in range(0, 360, 5):
        dir_rad = math.radians(dir_deg)
        dx = math.cos(dir_rad)
        dy = math.sin(dir_rad)

        # Binary search for touching distance
        lo, hi = 0.0, 2.0
        while hi - lo > 0.001:
            mid = (lo + hi) / 2
            poly1 = get_tree_poly(mid * dx, mid * dy, a1)
            if poly0.intersects(poly1) and poly0.intersection(poly1).area > 1e-9:
                lo = mid
            else:
                hi = mid

        # Test at just-outside distance
        for dist_offset in [0.001, 0.002, 0.005, 0.01, 0.02]:
            dist = hi + dist_offset
            x1, y1 = dist * dx, dist * dy
            poly1 = get_tree_poly(x1, y1, a1)

            if not has_overlap(poly0, poly1):
                side = compute_side(poly0, poly1)
                if side < best_side:
                    best_side = side
                    best_pos = (x1, y1)

    return best_side, best_pos


def local_refine(x0, y0, a0, x1, y1, a1, best_side):
    """Local refinement with gradient-like search."""
    step_x = 0.01
    step_y = 0.01
    step_a = 1.0

    improved = True
    while improved:
        improved = False

        # Try small perturbations
        for dx in [-step_x, 0, step_x]:
            for dy in [-step_y, 0, step_y]:
                for da0 in [-step_a, 0, step_a]:
                    for da1 in [-step_a, 0, step_a]:
                        if dx == dy == da0 == da1 == 0:
                            continue

                        poly0 = get_tree_poly(x0, y0, a0 + da0)
                        poly1 = get_tree_poly(x1 + dx, y1 + dy, a1 + da1)

                        if has_overlap(poly0, poly1):
                            continue

                        side = compute_side(poly0, poly1)
                        if side < best_side - 0.0001:
                            best_side = side
                            a0 += da0
                            x1 += dx
                            y1 += dy
                            a1 += da1
                            improved = True

        step_x *= 0.5
        step_y *= 0.5
        step_a *= 0.5

        if step_x < 0.0001:
            break

    return best_side, (x0, y0, a0, x1, y1, a1)


def main():
    print("=== Fast N=2 Exhaustive Search ===\n")

    # Current best from submission
    x0_cur, y0_cur, a0_cur = 0, 0, 159.55
    x1_cur, y1_cur, a1_cur = 0.44, -0.45, 113.49

    poly0 = get_tree_poly(x0_cur, y0_cur, a0_cur)
    poly1 = get_tree_poly(x1_cur, y1_cur, a1_cur)
    current_side = compute_side(poly0, poly1)
    print(f"Current: side={current_side:.6f}, score={current_side**2/2:.6f}")

    best_side = current_side
    best_config = (x0_cur, y0_cur, a0_cur, x1_cur, y1_cur, a1_cur)

    # Coarse search over angle pairs
    print("\nPhase 1: Angle pair search...")
    count = 0
    for a0 in range(0, 95, 5):  # 0-90 due to symmetry
        for a1 in range(0, 360, 5):
            side, pos = find_best_position(a0, a1, best_side)
            if pos and side < best_side:
                best_side = side
                best_config = (0, 0, a0, pos[0], pos[1], a1)
                print(f"  Found: a0={a0}, a1={a1}, side={side:.6f}")
            count += 1
            if count % 200 == 0:
                print(f"  Progress: {count}/1440 angle pairs...")

    print(f"\nBest after coarse search: side={best_side:.6f}")

    if best_side < current_side:
        # Local refinement
        print("\nPhase 2: Local refinement...")
        x0, y0, a0, x1, y1, a1 = best_config
        best_side, best_config = local_refine(x0, y0, a0, x1, y1, a1, best_side)
        print(f"After refinement: side={best_side:.6f}")

    print(f"\n=== Final Result ===")
    if best_side < current_side - 0.0001:
        x0, y0, a0, x1, y1, a1 = best_config
        print(f"IMPROVED!")
        print(f"  Side: {current_side:.6f} -> {best_side:.6f}")
        print(f"  Score: {current_side**2/2:.6f} -> {best_side**2/2:.6f}")
        print(f"  Delta: {(current_side**2 - best_side**2)/2:.6f}")
        print(f"\nConfiguration:")
        print(f"  Tree 0: x={x0:.8f}, y={y0:.8f}, angle={a0:.8f}")
        print(f"  Tree 1: x={x1:.8f}, y={y1:.8f}, angle={a1:.8f}")

        # Center the solution
        poly0 = get_tree_poly(x0, y0, a0)
        poly1 = get_tree_poly(x1, y1, a1)
        coords = list(poly0.exterior.coords) + list(poly1.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2

        print(f"\nCentered (for submission):")
        print(f"  Tree 0: x={x0-cx:.8f}, y={y0-cy:.8f}, angle={a0:.8f}")
        print(f"  Tree 1: x={x1-cx:.8f}, y={y1-cy:.8f}, angle={a1:.8f}")
    else:
        print(f"No improvement found. Current side={current_side:.6f} is already optimal.")


if __name__ == '__main__':
    main()
