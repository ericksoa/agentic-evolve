#!/usr/bin/env python3
"""
Fast n=2 optimization using multi-resolution search.
"""

import math
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)


@dataclass
class PlacedTree:
    x: float
    y: float
    angle: float

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)

    def get_vertices(self) -> List[Tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [(vx * cos_a - vy * sin_a + self.x,
                 vx * sin_a + vy * cos_a + self.y) for vx, vy in TREE_VERTICES]


def compute_side_length(trees: List[PlacedTree]) -> float:
    if not trees:
        return 0.0
    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trees_overlap(t1: PlacedTree, t2: PlacedTree, margin: float = 1e-9) -> bool:
    p1 = t1.get_polygon()
    p2 = t2.get_polygon()
    if not p1.intersects(p2):
        return False
    return p1.intersection(p2).area > margin


def search_n2():
    """Multi-resolution search for best n=2 configuration."""
    print("Multi-resolution n=2 search")
    print("=" * 60)

    best_side = float('inf')
    best_config = None

    # Coarse search: 30 degree angles, 0.1 position step
    print("Phase 1: Coarse search...")
    candidates = []

    for a1 in range(0, 360, 30):
        for a2 in range(0, 360, 30):
            t1 = PlacedTree(0, 0, a1)

            for x in np.arange(-0.5, 1.2, 0.1):
                for y in np.arange(-0.8, 0.8, 0.1):
                    t2 = PlacedTree(x, y, a2)
                    if not trees_overlap(t1, t2):
                        side = compute_side_length([t1, t2])
                        if side < best_side:
                            best_side = side
                            best_config = (0, 0, a1, x, y, a2)
                        if side < 1.1:
                            candidates.append((a1, a2, x, y, side))

    print(f"  Found {len(candidates)} promising candidates")
    print(f"  Best coarse: side = {best_side:.6f}")

    # Medium search around best candidates
    print("Phase 2: Medium refinement...")
    candidates.sort(key=lambda c: c[4])
    top_candidates = candidates[:100]

    for a1, a2, x_base, y_base, _ in top_candidates:
        for da1 in [-15, 0, 15]:
            for da2 in [-15, 0, 15]:
                new_a1 = (a1 + da1) % 360
                new_a2 = (a2 + da2) % 360
                t1 = PlacedTree(0, 0, new_a1)

                for dx in np.arange(-0.05, 0.06, 0.02):
                    for dy in np.arange(-0.05, 0.06, 0.02):
                        t2 = PlacedTree(x_base + dx, y_base + dy, new_a2)
                        if not trees_overlap(t1, t2):
                            side = compute_side_length([t1, t2])
                            if side < best_side:
                                best_side = side
                                best_config = (0, 0, new_a1, x_base + dx, y_base + dy, new_a2)

    print(f"  Best medium: side = {best_side:.6f}")

    # Fine search around the best
    print("Phase 3: Fine refinement...")
    if best_config:
        _, _, a1, x_base, y_base, a2 = best_config

        for da1 in range(-10, 11, 2):
            for da2 in range(-10, 11, 2):
                new_a1 = a1 + da1
                new_a2 = a2 + da2
                t1 = PlacedTree(0, 0, new_a1)

                for dx in np.arange(-0.02, 0.025, 0.005):
                    for dy in np.arange(-0.02, 0.025, 0.005):
                        t2 = PlacedTree(x_base + dx, y_base + dy, new_a2)
                        if not trees_overlap(t1, t2):
                            side = compute_side_length([t1, t2])
                            if side < best_side:
                                best_side = side
                                best_config = (0, 0, new_a1, x_base + dx, y_base + dy, new_a2)

    print(f"  Best fine: side = {best_side:.6f}")

    # Very fine search
    print("Phase 4: Very fine refinement...")
    if best_config:
        _, _, a1, x_base, y_base, a2 = best_config

        for da1 in np.arange(-2, 2.5, 0.5):
            for da2 in np.arange(-2, 2.5, 0.5):
                new_a1 = a1 + da1
                new_a2 = a2 + da2
                t1 = PlacedTree(0, 0, new_a1)

                for dx in np.arange(-0.005, 0.0055, 0.001):
                    for dy in np.arange(-0.005, 0.0055, 0.001):
                        t2 = PlacedTree(x_base + dx, y_base + dy, new_a2)
                        if not trees_overlap(t1, t2):
                            side = compute_side_length([t1, t2])
                            if side < best_side:
                                best_side = side
                                best_config = (0, 0, new_a1, x_base + dx, y_base + dy, new_a2)

    print(f"\n  Best final: side = {best_side:.6f}")

    # Report results
    print("\n" + "=" * 60)
    print("Results:")
    print("-" * 60)

    if best_config:
        x1, y1, a1, x2, y2, a2 = best_config
        print(f"Best n=2 configuration:")
        print(f"  Tree 1: x={x1:.4f}, y={y1:.4f}, angle={a1:.2f}")
        print(f"  Tree 2: x={x2:.4f}, y={y2:.4f}, angle={a2:.2f}")
        print(f"  Side length: {best_side:.6f}")
        print(f"  Score contribution: {best_side**2 / 2:.6f}")

        # Compare with current best
        print(f"\nCurrent best n=2:")
        print(f"  Side length: 0.9496")
        print(f"  Score contribution: 0.4509")

        improvement = (0.9496**2 - best_side**2) / 2
        if improvement > 0:
            print(f"\n*** IMPROVEMENT: {improvement:.6f} ***")
        else:
            print(f"\n  No improvement (difference: {improvement:.6f})")

        # Verify no overlap
        t1 = PlacedTree(x1, y1, a1)
        t2 = PlacedTree(x2, y2, a2)
        print(f"\nValidation: overlap={trees_overlap(t1, t2)}")

    return best_config, best_side


if __name__ == '__main__':
    config, side = search_n2()
