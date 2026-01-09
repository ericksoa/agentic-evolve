#!/usr/bin/env python3
"""
Gen123: Analyze tree interlocking patterns

The Christmas tree shape has concave regions (the tiered branches).
This script analyzes whether trees can interlock more tightly than
standard side-by-side packing.

Key idea: The tip of one tree could potentially fit into the concave
step of another tree's branches, allowing tighter packing.
"""

import math
import json
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon, LineString
from shapely import affinity
from shapely.strtree import STRtree

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8),      # 0: Tip
    (0.125, 0.5),    # 1: Right top tier outer
    (0.0625, 0.5),   # 2: Right top tier inner
    (0.2, 0.25),     # 3: Right mid tier outer
    (0.1, 0.25),     # 4: Right mid tier inner
    (0.35, 0.0),     # 5: Right base outer
    (0.075, 0.0),    # 6: Right trunk
    (0.075, -0.2),   # 7: Right trunk bottom
    (-0.075, -0.2),  # 8: Left trunk bottom
    (-0.075, 0.0),   # 9: Left trunk
    (-0.35, 0.0),    # 10: Left base outer
    (-0.1, 0.25),    # 11: Left mid tier inner
    (-0.2, 0.25),    # 12: Left mid tier outer
    (-0.0625, 0.5),  # 13: Left top tier inner
    (-0.125, 0.5),   # 14: Left top tier outer
]

TREE_POLYGON = Polygon(TREE_VERTICES)


@dataclass
class PlacedTree:
    x: float
    y: float
    angle: float  # degrees

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)

    def get_vertices(self) -> List[Tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [(vx * cos_a - vy * sin_a + self.x,
                 vx * sin_a + vy * cos_a + self.y) for vx, vy in TREE_VERTICES]


def compute_side_length(trees: List[PlacedTree]) -> float:
    """Compute bounding square side length."""
    if not trees:
        return 0.0

    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trees_overlap(t1: PlacedTree, t2: PlacedTree, margin: float = 1e-9) -> bool:
    """Check if two trees overlap."""
    p1 = t1.get_polygon()
    p2 = t2.get_polygon()

    if not p1.intersects(p2):
        return False

    intersection = p1.intersection(p2)
    return intersection.area > margin


def analyze_tree_shape():
    """Analyze the tree shape for interlocking opportunities."""
    print("Tree Shape Analysis")
    print("=" * 60)

    # Basic dimensions
    xs = [v[0] for v in TREE_VERTICES]
    ys = [v[1] for v in TREE_VERTICES]

    print(f"Width: {max(xs) - min(xs):.4f} (x: {min(xs):.4f} to {max(xs):.4f})")
    print(f"Height: {max(ys) - min(ys):.4f} (y: {min(ys):.4f} to {max(ys):.4f})")
    print()

    # Identify concave regions (steps between tiers)
    print("Concave regions (potential nesting points):")

    # Right side steps
    print("  Right side:")
    print(f"    Top step: y=0.5, indent from x=0.125 to x=0.0625 (depth: 0.0625)")
    print(f"    Mid step: y=0.25, indent from x=0.2 to x=0.1 (depth: 0.1)")

    # Left side steps
    print("  Left side:")
    print(f"    Top step: y=0.5, indent from x=-0.125 to x=-0.0625 (depth: 0.0625)")
    print(f"    Mid step: y=0.25, indent from x=-0.2 to x=-0.1 (depth: 0.1)")
    print()

    # Tip dimensions
    print(f"Tip: ({TREE_VERTICES[0]}) - can potentially nest into concave regions")
    print(f"Tip width at y=0.5: {0.125 * 2:.4f}")
    print()


def find_interlocking_pairs():
    """Find angle pairs where trees can interlock."""
    print("Searching for interlocking configurations...")
    print("=" * 60)

    results = []

    # Try many angle combinations with fine precision
    angle_steps = 72  # 5 degree increments

    for angle1 in range(0, 360, 5):
        for angle2 in range(0, 360, 5):
            # Place first tree at origin
            t1 = PlacedTree(0, 0, angle1)

            # For each angle pair, find minimum distance placement
            best_side = float('inf')
            best_t2 = None

            # Try different offset directions
            for offset_angle in range(0, 360, 15):
                offset_rad = math.radians(offset_angle)
                dx, dy = math.cos(offset_rad), math.sin(offset_rad)

                # Binary search for minimum valid distance
                lo, hi = 0.3, 2.0
                while hi - lo > 0.001:
                    mid = (lo + hi) / 2
                    t2 = PlacedTree(dx * mid, dy * mid, angle2)
                    if trees_overlap(t1, t2):
                        lo = mid
                    else:
                        hi = mid

                # Test at slightly above minimum
                for delta in [0.001, 0.002, 0.005]:
                    dist = hi + delta
                    t2 = PlacedTree(dx * dist, dy * dist, angle2)
                    if not trees_overlap(t1, t2):
                        side = compute_side_length([t1, t2])
                        if side < best_side:
                            best_side = side
                            best_t2 = t2

            if best_t2 and best_side < 1.0:  # Good results only
                results.append({
                    'angle1': angle1,
                    'angle2': angle2,
                    'side': best_side,
                    'tree2_x': best_t2.x,
                    'tree2_y': best_t2.y
                })

    # Sort by side length
    results.sort(key=lambda r: r['side'])

    print(f"\nTop 20 interlocking configurations (n=2):")
    print("-" * 60)
    for i, r in enumerate(results[:20]):
        print(f"{i+1:2}. angles=({r['angle1']:3}, {r['angle2']:3}), side={r['side']:.6f}")

    return results


def optimize_n2_intensive():
    """Intensive optimization for n=2."""
    print("\nIntensive n=2 optimization...")
    print("=" * 60)

    best_side = float('inf')
    best_config = None

    # Place first tree at origin with various rotations
    for angle1 in range(0, 360, 15):
        t1 = PlacedTree(0, 0, angle1)

        # Fine grid search for second tree
        for angle2 in range(0, 360, 15):
            # Search over a grid of positions
            for x in np.arange(-1.0, 1.5, 0.02):
                for y in np.arange(-1.0, 1.5, 0.02):
                    t2 = PlacedTree(x, y, angle2)

                    if not trees_overlap(t1, t2):
                        side = compute_side_length([t1, t2])
                        if side < best_side:
                            best_side = side
                            best_config = {
                                'tree1': {'x': 0, 'y': 0, 'angle': angle1},
                                'tree2': {'x': x, 'y': y, 'angle': angle2},
                                'side': side
                            }

    if best_config:
        print(f"Best n=2 found: side = {best_side:.6f}")
        print(f"  Tree 1: x=0, y=0, angle={best_config['tree1']['angle']}")
        print(f"  Tree 2: x={best_config['tree2']['x']:.4f}, y={best_config['tree2']['y']:.4f}, angle={best_config['tree2']['angle']}")
        print(f"  Score contribution: {best_side**2 / 2:.6f}")

        # Compare with current best
        print(f"\n  Current best n=2: side=0.9496, score=0.4509")
        improvement = (0.9496**2 - best_side**2) / 2
        print(f"  Potential improvement: {improvement:.6f}")

    return best_config


def optimize_n2_continuous():
    """Fine continuous optimization for n=2 starting from best discrete."""
    print("\nContinuous n=2 optimization...")
    print("=" * 60)

    from scipy.optimize import minimize

    def objective(params):
        """Objective: side length with penalty for overlap."""
        x2, y2, a1, a2 = params
        t1 = PlacedTree(0, 0, a1 * 360)
        t2 = PlacedTree(x2, y2, a2 * 360)

        side = compute_side_length([t1, t2])

        # Heavy penalty for overlap
        if trees_overlap(t1, t2):
            return side + 10.0

        return side

    best_side = float('inf')
    best_params = None

    # Try multiple starting points
    for a1_init in [0, 45, 90, 135, 180]:
        for a2_init in [0, 45, 90, 135, 180]:
            for x_init in [0.4, 0.6, 0.8]:
                for y_init in [-0.3, 0, 0.3]:
                    x0 = [x_init, y_init, a1_init / 360, a2_init / 360]

                    try:
                        result = minimize(
                            objective, x0,
                            method='Nelder-Mead',
                            options={'maxiter': 1000}
                        )

                        if result.fun < best_side:
                            best_side = result.fun
                            best_params = result.x
                    except:
                        pass

    if best_params and best_side < 1.0:
        x2, y2, a1, a2 = best_params
        print(f"Best continuous n=2: side = {best_side:.6f}")
        print(f"  Tree 1: x=0, y=0, angle={a1 * 360:.2f}")
        print(f"  Tree 2: x={x2:.4f}, y={y2:.4f}, angle={a2 * 360:.2f}")

        # Verify no overlap
        t1 = PlacedTree(0, 0, a1 * 360)
        t2 = PlacedTree(x2, y2, a2 * 360)
        print(f"  Overlaps: {trees_overlap(t1, t2)}")

    return best_params, best_side


def main():
    # Analyze tree shape
    analyze_tree_shape()

    # Find interlocking configurations
    results = find_interlocking_pairs()

    # Intensive n=2 optimization
    config = optimize_n2_intensive()

    # Continuous refinement
    try:
        params, side = optimize_n2_continuous()
    except ImportError:
        print("scipy not available for continuous optimization")


if __name__ == '__main__':
    main()
