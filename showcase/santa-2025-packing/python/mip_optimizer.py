#!/usr/bin/env python3
"""
Mixed Integer Programming optimizer for Santa 2025 Tree Packing.

Uses OR-Tools CP-SAT solver with discretized positions.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import json
import time

# Try to import OR-Tools
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("Warning: OR-Tools not installed. Using scipy fallback.")

# Tree polygon vertices (from Rust lib.rs)
TREE_VERTICES = np.array([
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
])

ROTATION_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def rotate_polygon(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate polygon vertices by given angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return vertices @ rotation_matrix.T


def polygon_bounds(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) bounding box."""
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    return min_x, min_y, max_x, max_y


def cross_product_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a1, a2, b1, b2) -> bool:
    """Check if segment (a1,a2) intersects segment (b1,b2)."""
    eps = 1e-9
    d1 = cross_product_2d(b1, b2, a1)
    d2 = cross_product_2d(b1, b2, a2)
    d3 = cross_product_2d(a1, a2, b1)
    d4 = cross_product_2d(a1, a2, b2)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if point is inside polygon using winding number."""
    winding = 0
    n = len(polygon)
    px, py = point

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 <= py:
            if y2 > py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross > 1e-10:
                    winding += 1
        else:
            if y2 <= py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross < -1e-10:
                    winding -= 1

    return winding != 0


def polygons_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Check if two polygons overlap."""
    b1 = polygon_bounds(poly1)
    b2 = polygon_bounds(poly2)
    eps = 1e-9

    if b1[2] + eps < b2[0] or b2[2] + eps < b1[0] or \
       b1[3] + eps < b2[1] or b2[3] + eps < b1[1]:
        return False

    n1, n2 = len(poly1), len(poly2)

    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return True

    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True

    return False


class ConflictGraph:
    """
    Precompute which grid positions conflict for each rotation pair.

    This creates a graph where nodes are (grid_x, grid_y, rotation) tuples
    and edges connect conflicting placements.
    """

    def __init__(self, grid_resolution: float, grid_range: float):
        self.resolution = grid_resolution
        self.grid_range = grid_range

        # Create grid points
        self.grid_points = []
        steps = int(2 * grid_range / grid_resolution) + 1
        for i in range(steps):
            for j in range(steps):
                x = -grid_range + i * grid_resolution
                y = -grid_range + j * grid_resolution
                self.grid_points.append((x, y))

        self.n_points = len(self.grid_points)
        print(f"Grid: {self.n_points} points, resolution {grid_resolution}")

        # Precompute rotated polygons
        self.rotated_polys = []
        for r in range(8):
            self.rotated_polys.append(rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[r]))

        # Precompute conflicts
        self._precompute_conflicts()

    def _trees_conflict(self, x1: float, y1: float, r1: int,
                       x2: float, y2: float, r2: int) -> bool:
        """Check if two tree placements conflict."""
        poly1 = self.rotated_polys[r1] + np.array([x1, y1])
        poly2 = self.rotated_polys[r2] + np.array([x2, y2])
        return polygons_overlap(poly1, poly2)

    def _precompute_conflicts(self):
        """
        Precompute which (position, rotation) pairs conflict.

        For efficiency, we compute conflicts relative to origin.
        If tree at (0,0,r1) conflicts with tree at (dx,dy,r2),
        then tree at (x,y,r1) conflicts with tree at (x+dx,y+dy,r2).
        """
        self.relative_conflicts = {}  # (r1, r2) -> set of (dx, dy) that conflict

        for r1 in range(8):
            for r2 in range(8):
                conflicts = set()
                # Check positions in a local range
                local_range = 2.0  # Trees can't be further than this and overlap
                local_steps = int(2 * local_range / self.resolution) + 1

                for i in range(local_steps):
                    for j in range(local_steps):
                        dx = -local_range + i * self.resolution
                        dy = -local_range + j * self.resolution

                        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                            conflicts.add((0, 0))
                            continue

                        if self._trees_conflict(0, 0, r1, dx, dy, r2):
                            # Quantize to grid
                            qx = round(dx / self.resolution) * self.resolution
                            qy = round(dy / self.resolution) * self.resolution
                            conflicts.add((qx, qy))

                self.relative_conflicts[(r1, r2)] = conflicts

        print(f"Precomputed {len(self.relative_conflicts)} conflict sets")

    def get_conflicts(self, r1: int, r2: int) -> Set[Tuple[float, float]]:
        """Get relative positions that conflict."""
        return self.relative_conflicts[(r1, r2)]


@dataclass
class PlacedTree:
    x: float
    y: float
    rotation: int  # 0-7

    def angle_deg(self) -> float:
        return ROTATION_ANGLES[self.rotation]


def solve_mip_cpsat(n: int, grid_resolution: float = 0.1,
                    grid_range: float = 3.0,
                    time_limit_sec: float = 60.0) -> List[PlacedTree]:
    """
    Solve tree packing using CP-SAT MIP formulation.

    Variables:
    - placement[i, p, r] = 1 if tree i is at grid point p with rotation r

    Constraints:
    - Each tree has exactly one placement
    - No two trees can have conflicting placements
    """
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not installed")

    model = cp_model.CpModel()

    # Build conflict graph
    conflicts = ConflictGraph(grid_resolution, grid_range)
    n_points = conflicts.n_points
    grid_points = conflicts.grid_points

    print(f"Creating MIP model for n={n}, {n_points} grid points")

    # Variables: placement[tree, point, rotation]
    placement = {}
    for i in range(n):
        for p in range(n_points):
            for r in range(8):
                placement[i, p, r] = model.NewBoolVar(f'place_{i}_{p}_{r}')

    # Constraint: each tree has exactly one placement
    for i in range(n):
        model.Add(sum(placement[i, p, r]
                     for p in range(n_points)
                     for r in range(8)) == 1)

    # Constraint: no conflicts between trees
    for i in range(n):
        for j in range(i + 1, n):
            for p1 in range(n_points):
                x1, y1 = grid_points[p1]
                for r1 in range(8):
                    for r2 in range(8):
                        conflict_offsets = conflicts.get_conflicts(r1, r2)
                        for (dx, dy) in conflict_offsets:
                            x2, y2 = x1 + dx, y1 + dy
                            # Find grid point closest to (x2, y2)
                            p2 = None
                            for p in range(n_points):
                                if abs(grid_points[p][0] - x2) < grid_resolution/2 and \
                                   abs(grid_points[p][1] - y2) < grid_resolution/2:
                                    p2 = p
                                    break

                            if p2 is not None:
                                # Can't have both placements
                                model.Add(placement[i, p1, r1] + placement[j, p2, r2] <= 1)

    # Objective: minimize bounding box
    # We need to track min/max x and y
    SCALE = 1000  # Scale to integers
    min_x = model.NewIntVar(-int(grid_range * SCALE), int(grid_range * SCALE), 'min_x')
    max_x = model.NewIntVar(-int(grid_range * SCALE), int(grid_range * SCALE), 'max_x')
    min_y = model.NewIntVar(-int(grid_range * SCALE), int(grid_range * SCALE), 'min_y')
    max_y = model.NewIntVar(-int(grid_range * SCALE), int(grid_range * SCALE), 'max_y')

    # For each tree, track its bounds contribution
    for i in range(n):
        for p in range(n_points):
            for r in range(8):
                x, y = grid_points[p]
                poly = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[r]) + np.array([x, y])
                bounds = polygon_bounds(poly)

                # If this placement is active, it affects bounds
                px_min = int(bounds[0] * SCALE)
                py_min = int(bounds[1] * SCALE)
                px_max = int(bounds[2] * SCALE)
                py_max = int(bounds[3] * SCALE)

                # When placement[i,p,r] = 1, these bounds apply
                model.Add(min_x <= px_min).OnlyEnforceIf(placement[i, p, r])
                model.Add(max_x >= px_max).OnlyEnforceIf(placement[i, p, r])
                model.Add(min_y <= py_min).OnlyEnforceIf(placement[i, p, r])
                model.Add(max_y >= py_max).OnlyEnforceIf(placement[i, p, r])

    # Minimize max of width and height
    width = model.NewIntVar(0, int(2 * grid_range * SCALE), 'width')
    height = model.NewIntVar(0, int(2 * grid_range * SCALE), 'height')
    model.Add(width == max_x - min_x)
    model.Add(height == max_y - min_y)

    side = model.NewIntVar(0, int(2 * grid_range * SCALE), 'side')
    model.AddMaxEquality(side, [width, height])

    model.Minimize(side)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = True

    print(f"Solving with time limit {time_limit_sec}s...")
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"Solution found! Side = {solver.Value(side) / SCALE:.4f}")

        trees = []
        for i in range(n):
            for p in range(n_points):
                for r in range(8):
                    if solver.Value(placement[i, p, r]):
                        x, y = grid_points[p]
                        trees.append(PlacedTree(x, y, r))
                        break
                else:
                    continue
                break

        return trees
    else:
        print(f"No solution found, status = {status}")
        return []


def solve_simpler_mip(n: int, time_limit_sec: float = 60.0) -> List[PlacedTree]:
    """
    Simpler MIP formulation using continuous positions with Big-M constraints.
    """
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not installed")

    # For this problem, use simulated annealing as fallback
    # Full MIP with non-overlap constraints is complex

    from nfp_optimizer import greedy_nfp_placement, NFPCache, local_search, compute_packing_side

    nfp_cache = NFPCache()
    trees = greedy_nfp_placement(n, nfp_cache)
    trees = local_search(trees, iterations=1000)

    return [PlacedTree(t.x, t.y, t.angle_idx) for t in trees]


def verify_solution(trees: List[PlacedTree]) -> bool:
    """Verify no trees overlap."""
    for i, t1 in enumerate(trees):
        poly1 = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[t1.rotation]) + np.array([t1.x, t1.y])
        for j, t2 in enumerate(trees[i+1:], i+1):
            poly2 = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[t2.rotation]) + np.array([t2.x, t2.y])
            if polygons_overlap(poly1, poly2):
                print(f"Overlap between tree {i} and {j}")
                return False
    return True


def compute_side(trees: List[PlacedTree]) -> float:
    """Compute bounding box side length."""
    if not trees:
        return 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for t in trees:
        poly = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[t.rotation]) + np.array([t.x, t.y])
        bounds = polygon_bounds(poly)
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    return max(max_x - min_x, max_y - min_y)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--resolution', type=float, default=0.1)
    parser.add_argument('--time-limit', type=float, default=60.0)
    args = parser.parse_args()

    print(f"Santa 2025 Tree Packing - MIP Optimizer")
    print(f"Solving for n={args.n}")

    if HAS_ORTOOLS and args.n <= 8:
        # Use exact solver for small n
        trees = solve_mip_cpsat(args.n, args.resolution, time_limit_sec=args.time_limit)
    else:
        # Use heuristic
        trees = solve_simpler_mip(args.n, args.time_limit)

    if trees:
        print(f"\nSolution with {len(trees)} trees:")
        for i, t in enumerate(trees):
            print(f"  Tree {i}: ({t.x:.4f}, {t.y:.4f}) rot={t.rotation * 45}°")

        side = compute_side(trees)
        print(f"\nBounding box side: {side:.4f}")
        print(f"Valid: {verify_solution(trees)}")
