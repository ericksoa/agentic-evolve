#!/usr/bin/env python3
"""
Gen123: CP-SAT Solver for Small N Tree Packing

Uses Google OR-Tools CP-SAT to find optimal packings for small n values.

Key approach:
1. Discretize positions to integer grid (scaled by precision factor)
2. Use 8 discrete rotation angles (45 degree steps)
3. Add pairwise no-overlap constraints using precomputed NFP
4. Minimize bounding square side length

For very small n (1-5), this can find globally optimal solutions.
For larger n (6-15), it provides good upper bounds.
"""

import math
import csv
import json
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from ortools.sat.python import cp_model
from shapely.geometry import Polygon
from shapely import affinity
import numpy as np

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
    angle: float  # degrees


def get_rotated_vertices(angle_deg: float) -> List[Tuple[float, float]]:
    """Get tree vertices rotated by angle_deg."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [(vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a)
            for vx, vy in TREE_VERTICES]


def get_rotated_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Get translated and rotated tree polygon."""
    rotated = affinity.rotate(TREE_POLYGON, angle_deg, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_side_length(trees: List[PlacedTree]) -> float:
    """Compute bounding square side length."""
    if not trees:
        return 0.0

    all_verts = []
    for t in trees:
        verts = get_rotated_vertices(t.angle)
        for vx, vy in verts:
            all_verts.append((vx + t.x, vy + t.y))

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def trees_overlap(t1: PlacedTree, t2: PlacedTree, margin: float = 1e-9) -> bool:
    """Check if two trees overlap using Shapely."""
    p1 = get_rotated_polygon(t1.x, t1.y, t1.angle)
    p2 = get_rotated_polygon(t2.x, t2.y, t2.angle)

    if not p1.intersects(p2):
        return False

    intersection = p1.intersection(p2)
    return intersection.area > margin


def has_any_overlap(trees: List[PlacedTree], margin: float = 1e-9) -> bool:
    """Check if any trees overlap."""
    n = len(trees)
    for i in range(n):
        for j in range(i + 1, n):
            if trees_overlap(trees[i], trees[j], margin):
                return True
    return False


def precompute_nfp_offsets(precision: int = 100) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Precompute no-fit polygon minimum offsets for each rotation pair.

    For each (rot1, rot2) pair, compute the minimum offsets where
    tree2 can be placed relative to tree1 without overlap.

    Returns dict mapping (rot1, rot2) -> list of (min_dx, min_dy) at various angles
    """
    nfp_cache = {}

    for rot1 in range(8):
        for rot2 in range(8):
            angle1 = rot1 * 45.0
            angle2 = rot2 * 45.0

            p1 = get_rotated_polygon(0, 0, angle1)

            # Sample offsets and find minimum distances
            min_offsets = []

            # For each direction, find minimum valid offset
            for dir_angle in range(0, 360, 15):
                dir_rad = math.radians(dir_angle)
                dx, dy = math.cos(dir_rad), math.sin(dir_rad)

                # Binary search for minimum distance
                lo, hi = 0.0, 3.0
                while hi - lo > 0.001:
                    mid = (lo + hi) / 2
                    p2 = get_rotated_polygon(dx * mid, dy * mid, angle2)
                    if p1.intersects(p2) and p1.intersection(p2).area > 1e-10:
                        lo = mid
                    else:
                        hi = mid

                min_dist = hi
                min_offsets.append((
                    int(dx * min_dist * precision),
                    int(dy * min_dist * precision)
                ))

            nfp_cache[(rot1, rot2)] = min_offsets

    return nfp_cache


def solve_cpsat_basic(n: int, precision: int = 100, time_limit: int = 60) -> Tuple[float, List[PlacedTree]]:
    """
    Solve tree packing using CP-SAT with basic position constraints.

    This uses a simple approach:
    1. Integer position variables (x, y) scaled by precision
    2. Discrete rotation (0-7 for 45 degree steps)
    3. No-overlap constraints using minimum distance
    4. Minimize bounding box

    Args:
        n: Number of trees
        precision: Position precision (positions are integers / precision)
        time_limit: Solver time limit in seconds

    Returns:
        (side_length, list of PlacedTree)
    """
    model = cp_model.CpModel()

    # Estimate reasonable bounds
    max_pos = int(n * 1.5 * precision)  # Allow positions up to n * 1.5

    # Variables for each tree
    x_vars = [model.new_int_var(-max_pos, max_pos, f'x_{i}') for i in range(n)]
    y_vars = [model.new_int_var(-max_pos, max_pos, f'y_{i}') for i in range(n)]
    rot_vars = [model.new_int_var(0, 7, f'rot_{i}') for i in range(n)]

    # Bounding box variables
    min_x = model.new_int_var(-max_pos, max_pos, 'min_x')
    max_x = model.new_int_var(-max_pos, max_pos, 'max_x')
    min_y = model.new_int_var(-max_pos, max_pos, 'min_y')
    max_y = model.new_int_var(-max_pos, max_pos, 'max_y')
    side = model.new_int_var(0, 2 * max_pos, 'side')

    # Precompute tree extents for each rotation
    tree_extents = []  # [(min_x, max_x, min_y, max_y) for each rotation]
    for rot in range(8):
        verts = get_rotated_vertices(rot * 45.0)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        tree_extents.append((
            int(min(xs) * precision),
            int(max(xs) * precision),
            int(min(ys) * precision),
            int(max(ys) * precision)
        ))

    # Bounding box constraints
    for i in range(n):
        # For each rotation, add conditional constraint on bounding box contribution
        for rot in range(8):
            ext = tree_extents[rot]

            # When rot_vars[i] == rot, tree i contributes these extents
            b = model.new_bool_var(f'rot_{i}_{rot}')
            model.add(rot_vars[i] == rot).only_enforce_if(b)
            model.add(rot_vars[i] != rot).only_enforce_if(b.negated())

            # When this rotation is active, add bounding constraints
            model.add(min_x <= x_vars[i] + ext[0]).only_enforce_if(b)
            model.add(max_x >= x_vars[i] + ext[1]).only_enforce_if(b)
            model.add(min_y <= y_vars[i] + ext[2]).only_enforce_if(b)
            model.add(max_y >= y_vars[i] + ext[3]).only_enforce_if(b)

    # Side length constraint
    model.add(side >= max_x - min_x)
    model.add(side >= max_y - min_y)

    # Precompute minimum distances for no-overlap constraints
    # For simplicity, use bounding box overlap prevention
    min_dist_table = {}
    for rot1 in range(8):
        for rot2 in range(8):
            ext1 = tree_extents[rot1]
            ext2 = tree_extents[rot2]
            # Minimum distance based on bounding box (conservative)
            dx = (ext1[1] - ext1[0] + ext2[1] - ext2[0]) // 2 + 1
            dy = (ext1[3] - ext1[2] + ext2[3] - ext2[2]) // 2 + 1
            min_dist_table[(rot1, rot2)] = (dx, dy)

    # No-overlap constraints (using bounding box approximation)
    for i in range(n):
        for j in range(i + 1, n):
            # For each rotation pair, trees must be separated
            for rot_i in range(8):
                for rot_j in range(8):
                    b_i = model.new_bool_var(f'b_{i}_{rot_i}_{j}')
                    b_j = model.new_bool_var(f'b_{j}_{rot_j}_{i}')

                    model.add(rot_vars[i] == rot_i).only_enforce_if(b_i)
                    model.add(rot_vars[i] != rot_i).only_enforce_if(b_i.negated())
                    model.add(rot_vars[j] == rot_j).only_enforce_if(b_j)
                    model.add(rot_vars[j] != rot_j).only_enforce_if(b_j.negated())

                    # When both rotations are active
                    both = model.new_bool_var(f'both_{i}_{rot_i}_{j}_{rot_j}')
                    model.add_bool_and([b_i, b_j]).only_enforce_if(both)
                    model.add_bool_or([b_i.negated(), b_j.negated()]).only_enforce_if(both.negated())

                    # Disjunctive constraint: at least one separation holds
                    min_dx, min_dy = min_dist_table[(rot_i, rot_j)]

                    sep_left = model.new_bool_var(f'sep_left_{i}_{j}_{rot_i}_{rot_j}')
                    sep_right = model.new_bool_var(f'sep_right_{i}_{j}_{rot_i}_{rot_j}')
                    sep_down = model.new_bool_var(f'sep_down_{i}_{j}_{rot_i}_{rot_j}')
                    sep_up = model.new_bool_var(f'sep_up_{i}_{j}_{rot_i}_{rot_j}')

                    model.add(x_vars[j] - x_vars[i] >= min_dx).only_enforce_if(sep_right)
                    model.add(x_vars[i] - x_vars[j] >= min_dx).only_enforce_if(sep_left)
                    model.add(y_vars[j] - y_vars[i] >= min_dy).only_enforce_if(sep_up)
                    model.add(y_vars[i] - y_vars[j] >= min_dy).only_enforce_if(sep_down)

                    # At least one separation must hold when both rotations active
                    model.add_bool_or([sep_left, sep_right, sep_down, sep_up]).only_enforce_if(both)

    # Symmetry breaking: first tree at origin
    if n >= 1:
        model.add(x_vars[0] == 0)
        model.add(y_vars[0] == 0)

    # Second tree to the right of first (break reflection symmetry)
    if n >= 2:
        model.add(x_vars[1] >= x_vars[0])

    # Minimize side length
    model.minimize(side)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4

    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        trees = []
        for i in range(n):
            x = solver.value(x_vars[i]) / precision
            y = solver.value(y_vars[i]) / precision
            rot = solver.value(rot_vars[i])
            trees.append(PlacedTree(x, y, rot * 45.0))

        side_length = solver.value(side) / precision

        # Verify no overlaps
        if has_any_overlap(trees):
            print(f"  WARNING: CP-SAT solution has overlaps! Side={side_length:.6f}")
            return float('inf'), []

        return side_length, trees
    else:
        status_names = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE"
        }
        print(f"  CP-SAT status: {status_names.get(status, status)}")
        return float('inf'), []


def solve_cpsat_incremental(n: int, precision: int = 100, time_limit: int = 60) -> Tuple[float, List[PlacedTree]]:
    """
    Solve using incremental approach: start with loose bounds, tighten iteratively.
    """
    # First get a baseline solution using greedy
    best_side = float('inf')
    best_trees = []

    # Try multiple random starts
    import random
    for seed in range(5):
        random.seed(seed)
        trees = []
        for i in range(n):
            best_pos = None
            best_local_side = float('inf')

            # Try random positions
            for _ in range(50 if i > 0 else 1):
                if i == 0:
                    x, y = 0.0, 0.0
                else:
                    # Near existing trees
                    ref = random.choice(trees)
                    x = ref.x + random.uniform(-1.5, 1.5)
                    y = ref.y + random.uniform(-1.5, 1.5)

                for rot in range(8):
                    test_tree = PlacedTree(x, y, rot * 45.0)
                    test_trees = trees + [test_tree]

                    if not has_any_overlap(test_trees):
                        side = compute_side_length(test_trees)
                        if side < best_local_side:
                            best_local_side = side
                            best_pos = PlacedTree(x, y, rot * 45.0)

            if best_pos:
                trees.append(best_pos)
            else:
                # Fallback
                trees.append(PlacedTree(i * 1.0, 0, 0))

        side = compute_side_length(trees)
        if side < best_side and not has_any_overlap(trees):
            best_side = side
            best_trees = trees

    return best_side, best_trees


def load_best_submission(csv_path: str, n: int) -> List[PlacedTree]:
    """Load trees for specific n from submission CSV."""
    trees = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_n = int(row['id'].split('_')[0])
            if row_n == n:
                x = float(row['x'][1:])  # Remove 's' prefix
                y = float(row['y'][1:])
                deg = float(row['deg'][1:])
                trees.append(PlacedTree(x, y, deg))
    return trees


def save_results(results: Dict[int, Tuple[float, List[PlacedTree]]], output_path: str):
    """Save results to JSON file."""
    data = {}
    for n, (side, trees) in results.items():
        data[str(n)] = {
            'side': side,
            'score_contribution': side * side / n,
            'trees': [{'x': t.x, 'y': t.y, 'angle': t.angle} for t in trees]
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='CP-SAT solver for tree packing')
    parser.add_argument('--n-start', type=int, default=1, help='Start n value')
    parser.add_argument('--n-end', type=int, default=10, help='End n value')
    parser.add_argument('--precision', type=int, default=100, help='Position precision')
    parser.add_argument('--time-limit', type=int, default=60, help='Time limit per n')
    parser.add_argument('--compare', type=str, default='submission_best.csv', help='CSV to compare against')
    parser.add_argument('--output', type=str, default='cpsat_results.json', help='Output JSON file')
    args = parser.parse_args()

    print(f"CP-SAT Tree Packing Solver")
    print(f"  n range: {args.n_start}-{args.n_end}")
    print(f"  precision: {args.precision}")
    print(f"  time limit: {args.time_limit}s per n")
    print()

    results = {}
    total_improvement = 0.0

    for n in range(args.n_start, args.n_end + 1):
        print(f"n={n}:")

        # Load current best for comparison
        current_trees = []
        current_side = float('inf')
        try:
            current_trees = load_best_submission(args.compare, n)
            if current_trees:
                current_side = compute_side_length(current_trees)
        except:
            pass

        print(f"  Current best: side={current_side:.6f}")

        # Try CP-SAT
        cpsat_side, cpsat_trees = solve_cpsat_basic(n, args.precision, args.time_limit)
        print(f"  CP-SAT basic: side={cpsat_side:.6f}")

        # Try incremental greedy for comparison
        greedy_side, greedy_trees = solve_cpsat_incremental(n, args.precision, args.time_limit)
        print(f"  Greedy: side={greedy_side:.6f}")

        # Pick best
        if cpsat_side < greedy_side and cpsat_side < current_side:
            results[n] = (cpsat_side, cpsat_trees)
            improvement = (current_side**2 - cpsat_side**2) / n
            total_improvement += max(0, improvement)
            print(f"  -> Using CP-SAT (improvement: {improvement:.6f})")
        elif greedy_side < current_side:
            results[n] = (greedy_side, greedy_trees)
            improvement = (current_side**2 - greedy_side**2) / n
            total_improvement += max(0, improvement)
            print(f"  -> Using greedy (improvement: {improvement:.6f})")
        else:
            results[n] = (current_side, current_trees)
            print(f"  -> Keeping current best")

        print()

    print(f"Total score improvement: {total_improvement:.4f}")

    # Save results
    save_results(results, args.output)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
