#!/usr/bin/env python3
"""
ILP Solver for Small n (n <= 10) using OR-Tools CP-SAT

This approach discretizes the problem and uses constraint programming to find
optimal or near-optimal solutions for small n where the search space is tractable.

For n=1 to 10, optimal packing is critical because the score contribution is s²/n,
so small n values dominate the total score.
"""

import json
import math
import sys
from typing import List, Tuple, Optional
from ortools.sat.python import cp_model

# Tree vertices (same as Rust lib.rs)
TREE_VERTICES = [
    (0.0, 0.8),      # Tip
    (0.125, 0.5),    # Right - Top Tier
    (0.0625, 0.5),
    (0.2, 0.25),     # Right - Middle Tier
    (0.1, 0.25),
    (0.35, 0.0),     # Right - Bottom Tier
    (0.075, 0.0),    # Right Trunk
    (0.075, -0.2),
    (-0.075, -0.2),  # Left Trunk
    (-0.075, 0.0),
    (-0.35, 0.0),    # Left - Bottom Tier
    (-0.1, 0.25),    # Left - Middle Tier
    (-0.2, 0.25),
    (-0.0625, 0.5),  # Left - Top Tier
    (-0.125, 0.5),
]


def rotate_point(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    """Rotate a point around origin by angle in radians."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def get_tree_vertices(x: float, y: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Get tree vertices at given position and rotation."""
    angle_rad = angle_deg * math.pi / 180.0
    return [(rotate_point(vx, vy, angle_rad)[0] + x,
             rotate_point(vx, vy, angle_rad)[1] + y)
            for vx, vy in TREE_VERTICES]


def get_tree_bounds(x: float, y: float, angle_deg: float) -> Tuple[float, float, float, float]:
    """Get bounding box of tree at given position and rotation."""
    vertices = get_tree_vertices(x, y, angle_deg)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> float:
    """Cross product for orientation test."""
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A: Tuple[float, float], B: Tuple[float, float],
                       C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Check if line segment AB intersects with CD."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases - treat as not intersecting for our purposes
    return False


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting."""
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


def polygons_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> bool:
    """Check if two polygons overlap (not just touch)."""
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i+1) % n1],
                                  poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex of one is inside the other
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True

    return False


def trees_overlap(x1: float, y1: float, a1: float,
                  x2: float, y2: float, a2: float) -> bool:
    """Check if two trees overlap at given positions and rotations."""
    v1 = get_tree_vertices(x1, y1, a1)
    v2 = get_tree_vertices(x2, y2, a2)
    return polygons_overlap(v1, v2)


def compute_packing_bounds(n: int) -> Tuple[float, float]:
    """Estimate reasonable search bounds for n trees."""
    # A single tree fits in ~0.7 x 1.0 box
    # Very rough estimate: sqrt(n) * 0.8 for each dimension
    size = max(1.5, math.sqrt(n) * 0.9)
    return (-size, size)


def solve_ilp(n: int,
              grid_resolution: int = 50,
              angle_steps: int = 8,
              time_limit: float = 60.0,
              verbose: bool = True) -> Optional[List[Tuple[float, float, float]]]:
    """
    Solve the packing problem for n trees using CP-SAT.

    Args:
        n: Number of trees
        grid_resolution: Grid discretization (per dimension)
        angle_steps: Number of rotation angles to try (8 = every 45°)
        time_limit: Max solve time in seconds
        verbose: Print progress

    Returns:
        List of (x, y, angle_deg) for each tree, or None if no solution
    """
    if n < 1:
        return []
    if n == 1:
        # Optimal for n=1 is a single tree centered
        return [(0.0, 0.0, 0.0)]

    model = cp_model.CpModel()

    # Compute bounds
    min_coord, max_coord = compute_packing_bounds(n)
    coord_range = max_coord - min_coord

    # Discretize: convert continuous coords to integer grid
    # x_int in [0, grid_resolution] maps to [min_coord, max_coord]

    # Variables for each tree
    x = [model.NewIntVar(0, grid_resolution, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, grid_resolution, f'y_{i}') for i in range(n)]
    r = [model.NewIntVar(0, angle_steps - 1, f'r_{i}') for i in range(n)]

    # Side length (what we're minimizing)
    # We need to track the bounding box
    min_x = model.NewIntVar(0, grid_resolution, 'min_x')
    max_x = model.NewIntVar(0, grid_resolution, 'max_x')
    min_y = model.NewIntVar(0, grid_resolution, 'min_y')
    max_y = model.NewIntVar(0, grid_resolution, 'max_y')

    # The bounding box of each tree depends on its rotation
    # For simplicity, use a fixed buffer for tree extent
    tree_half_extent = int(0.6 / coord_range * grid_resolution) + 1  # ~0.5 radius

    model.AddMinEquality(min_x, [x[i] for i in range(n)])
    model.AddMaxEquality(max_x, [x[i] for i in range(n)])
    model.AddMinEquality(min_y, [y[i] for i in range(n)])
    model.AddMaxEquality(max_y, [y[i] for i in range(n)])

    # Width and height
    width = model.NewIntVar(0, grid_resolution + 2 * tree_half_extent, 'width')
    height = model.NewIntVar(0, grid_resolution + 2 * tree_half_extent, 'height')

    model.Add(width == max_x - min_x + 2 * tree_half_extent)
    model.Add(height == max_y - min_y + 2 * tree_half_extent)

    # Side = max(width, height)
    side = model.NewIntVar(0, grid_resolution + 2 * tree_half_extent, 'side')
    model.AddMaxEquality(side, [width, height])

    # Non-overlap constraints
    # This is the tricky part - we need to encode polygon overlap checking
    # For CP-SAT, we use big-M style constraints or precomputed forbidden pairs

    # Approach: For each pair (i,j) and each rotation pair (ri, rj),
    # precompute minimum distance required, then add constraints

    # For efficiency, use bounding box overlap as a necessary condition
    # If bbox don't overlap, trees don't overlap

    if verbose:
        print(f"  Adding non-overlap constraints for {n*(n-1)//2} pairs...")

    for i in range(n):
        for j in range(i + 1, n):
            # Bounding box non-overlap:
            # Either x[i] >= x[j] + buffer OR x[j] >= x[i] + buffer
            # OR y[i] >= y[j] + buffer OR y[j] >= y[i] + buffer

            min_dist = int(0.7 / coord_range * grid_resolution)  # Approximate tree width

            b1 = model.NewBoolVar(f'sep_xi>xj_{i}_{j}')
            b2 = model.NewBoolVar(f'sep_xj>xi_{i}_{j}')
            b3 = model.NewBoolVar(f'sep_yi>yj_{i}_{j}')
            b4 = model.NewBoolVar(f'sep_yj>yi_{i}_{j}')

            model.Add(x[i] >= x[j] + min_dist).OnlyEnforceIf(b1)
            model.Add(x[j] >= x[i] + min_dist).OnlyEnforceIf(b2)
            model.Add(y[i] >= y[j] + min_dist).OnlyEnforceIf(b3)
            model.Add(y[j] >= y[i] + min_dist).OnlyEnforceIf(b4)

            # At least one must be true
            model.AddBoolOr([b1, b2, b3, b4])

    # Objective: minimize side length
    model.Minimize(side)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    if verbose:
        print(f"  Solving with {time_limit}s timeout...")

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        result = []
        angles = [i * 360.0 / angle_steps for i in range(angle_steps)]

        for i in range(n):
            xi = solver.Value(x[i])
            yi = solver.Value(y[i])
            ri = solver.Value(r[i])

            # Convert back to continuous
            real_x = min_coord + (xi / grid_resolution) * coord_range
            real_y = min_coord + (yi / grid_resolution) * coord_range
            real_angle = angles[ri]

            result.append((real_x, real_y, real_angle))

        if verbose:
            side_val = solver.Value(side)
            real_side = (side_val / grid_resolution) * coord_range
            print(f"  Found solution: side={real_side:.4f}")
            print(f"  Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")

        return result
    else:
        if verbose:
            print(f"  No solution found (status: {solver.StatusName(status)})")
        return None


def validate_solution(solution: List[Tuple[float, float, float]]) -> Tuple[bool, float]:
    """Validate solution has no overlaps and compute side length."""
    n = len(solution)

    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = solution[i]
            x2, y2, a2 = solution[j]
            if trees_overlap(x1, y1, a1, x2, y2, a2):
                return False, float('inf')

    # Compute bounding box
    all_vertices = []
    for x, y, a in solution:
        all_vertices.extend(get_tree_vertices(x, y, a))

    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    side = max(width, height)

    return True, side


def local_refine(solution: List[Tuple[float, float, float]],
                 max_iters: int = 1000) -> List[Tuple[float, float, float]]:
    """
    Simple local search refinement to improve the discretized ILP solution.
    """
    import random

    best = list(solution)
    _, best_side = validate_solution(best)

    for _ in range(max_iters):
        # Pick random tree
        idx = random.randint(0, len(best) - 1)
        x, y, a = best[idx]

        # Try small perturbation
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        da = random.choice([0, 15, -15, 30, -30, 45, -45])

        candidate = list(best)
        candidate[idx] = (x + dx, y + dy, (a + da) % 360)

        valid, side = validate_solution(candidate)
        if valid and side < best_side:
            best = candidate
            best_side = side

    return best


def main():
    """Main entry point for ILP solver."""
    import argparse

    parser = argparse.ArgumentParser(description='ILP solver for small n')
    parser.add_argument('n', type=int, help='Number of trees')
    parser.add_argument('--grid', type=int, default=50, help='Grid resolution')
    parser.add_argument('--angles', type=int, default=8, help='Number of angles')
    parser.add_argument('--timeout', type=float, default=60.0, help='Time limit')
    parser.add_argument('--refine', action='store_true', help='Apply local refinement')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    print(f"Solving for n={args.n}...")

    solution = solve_ilp(args.n,
                         grid_resolution=args.grid,
                         angle_steps=args.angles,
                         time_limit=args.timeout)

    if solution is None:
        print("No solution found!")
        sys.exit(1)

    valid, side = validate_solution(solution)
    print(f"Initial solution: valid={valid}, side={side:.4f}")

    if args.refine:
        print("Applying local refinement...")
        solution = local_refine(solution)
        valid, side = validate_solution(solution)
        print(f"Refined solution: valid={valid}, side={side:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'n': args.n,
                'side': side,
                'trees': [{'x': x, 'y': y, 'angle': a} for x, y, a in solution]
            }, f, indent=2)
        print(f"Saved to {args.output}")

    # Print in format compatible with submission
    print("\nSolution:")
    for i, (x, y, a) in enumerate(solution):
        print(f"  Tree {i}: x={x:.4f}, y={y:.4f}, angle={a:.1f}")


if __name__ == '__main__':
    main()
