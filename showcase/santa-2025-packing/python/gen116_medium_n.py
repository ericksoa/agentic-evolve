#!/usr/bin/env python3
"""
Gen116 Medium-N Optimizer - CMA-ES with large search budget for n=11-30.

Key features:
1. 10 restarts per n value
2. 20k evals per restart (200k total per n)
3. Large sigma (0.25) for exploration
4. High overlap penalty (5000)
5. Strict overlap validation (1e-9)
"""

import json
import sys
import time
import random
import math
import csv
import numpy as np
sys.path.insert(0, 'python')

from cmaes_optimizer import (
    Tree, compute_side_length, compute_overlap_penalty, TREE_POLYGON
)
from shapely.geometry import Polygon
from shapely import affinity
import cma


# Strict overlap tolerance
OVERLAP_TOLERANCE = 1e-9

# Tree vertices for manual polygon checks
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]


def ccw(A, B, C):
    """Counter-clockwise test for segment intersection."""
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    """Check if segment AB intersects segment CD (proper intersection)."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon(point, polygon):
    """Ray casting algorithm for point-in-polygon test."""
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


def get_tree_vertices(tree) -> list:
    """Get vertices for a tree as list of tuples."""
    angle_rad = math.radians(tree.angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return [
        (vx * cos_a - vy * sin_a + tree.x,
         vx * sin_a + vy * cos_a + tree.y)
        for vx, vy in TREE_VERTICES
    ]


def polygons_overlap_strict(poly1, poly2) -> bool:
    """Check if two polygons overlap using segment intersection (strict)."""
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i+1) % n1],
                                  poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True

    return False


def has_any_overlap_strict(trees: list, tolerance: float = OVERLAP_TOLERANCE) -> bool:
    """Strict overlap check: Shapely fast filter + segment intersection for edge cases."""
    n = len(trees)
    shapely_polys = [t.get_polygon() for t in trees]

    for i in range(n):
        for j in range(i + 1, n):
            # Fast Shapely filter - if no intersection at all, skip
            if not shapely_polys[i].intersects(shapely_polys[j]):
                continue

            # Check Shapely area first
            intersection = shapely_polys[i].intersection(shapely_polys[j])
            if intersection.area > tolerance:
                return True

            # For zero-area intersections, use strict segment check
            poly1 = get_tree_vertices(trees[i])
            poly2 = get_tree_vertices(trees[j])
            if polygons_overlap_strict(poly1, poly2):
                return True
    return False


def get_max_overlap(trees: list) -> float:
    """Get the maximum pairwise overlap area (for soft penalty)."""
    n = len(trees)
    polygons = [t.get_polygon() for t in trees]
    max_overlap = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                max_overlap = max(max_overlap, intersection.area)

    return max_overlap


def trees_to_params(trees: list) -> np.ndarray:
    """Convert trees to params with continuous angle."""
    params = []
    for t in trees:
        params.extend([t.x, t.y, t.angle / 360.0])
    return np.array(params)


def params_to_trees(params: np.ndarray) -> list:
    """Convert params to trees with continuous angle."""
    n = len(params) // 3
    trees = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        angle = (params[3*i + 2] % 1.0) * 360.0
        trees.append(Tree(x, y, angle))
    return trees


def objective_strict(params: np.ndarray, penalty_weight: float = 5000.0) -> float:
    """Strict objective function with very high overlap penalty."""
    trees = params_to_trees(params)
    side = compute_side_length(trees)
    overlap = compute_overlap_penalty(trees)

    # Exponential penalty for any overlap
    if overlap > 1e-10:
        overlap_penalty = penalty_weight * overlap + penalty_weight * (1 + overlap * 100)
    else:
        overlap_penalty = 0

    return side + overlap_penalty


def repair_solution(trees: list, max_iters: int = 5000) -> list:
    """Repair overlapping solution by perturbations."""
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_overlap = get_max_overlap(best_trees)

    if best_overlap < OVERLAP_TOLERANCE:
        return best_trees

    # Strategy 1: Small perturbations
    for iteration in range(max_iters):
        if best_overlap < OVERLAP_TOLERANCE:
            break

        idx = random.randint(0, len(best_trees) - 1)
        t = best_trees[idx]

        scale = 0.1 * (1.0 - iteration / max_iters)
        dx = random.uniform(-scale, scale)
        dy = random.uniform(-scale, scale)
        da = random.uniform(-15, 15)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx+1:]

        candidate_overlap = get_max_overlap(candidate)
        if candidate_overlap < best_overlap * 0.999:
            best_trees = candidate
            best_overlap = candidate_overlap

    return best_trees


def load_trees_from_submission(csv_path: str, n: int) -> list:
    """Load trees for specific n from submission CSV (Kaggle format)."""
    trees = []
    prefix = f"{n:03d}_"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['id'].startswith(prefix):
                # Remove 's' prefix from values
                x = float(row['x'][1:])
                y = float(row['y'][1:])
                angle = float(row['deg'][1:])
                trees.append(Tree(x, y, angle))

    if len(trees) != n:
        raise ValueError(f"Expected {n} trees for n={n}, got {len(trees)}")

    return trees


def optimize_single_n(n: int, initial_trees: list, max_evals: int = 20000,
                      sigma0: float = 0.25, penalty_weight: float = 5000.0,
                      verbose: bool = True) -> tuple:
    """Optimize single n value with strict validation."""
    dim = 3 * n
    initial_params = trees_to_params(initial_trees)
    initial_side = compute_side_length(initial_trees)
    initial_valid = not has_any_overlap_strict(initial_trees)

    if verbose:
        print(f"    Initial: side={initial_side:.4f}, valid={initial_valid}")

    options = {
        'maxfevals': max_evals,
        'verbose': -9,
        'popsize': max(16, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-9,
        'tolx': 1e-9,
    }

    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    best_valid_side = initial_side if initial_valid else float('inf')
    best_valid_trees = initial_trees if initial_valid else None

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_strict(x, penalty_weight) for x in solutions]
        es.tell(solutions, fitnesses)

        # Check for strictly valid solutions
        for x, f in zip(solutions, fitnesses):
            trees = params_to_trees(x)
            if not has_any_overlap_strict(trees):
                side = compute_side_length(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = trees

        generation += 1
        if verbose and generation % 100 == 0:
            print(f"      Gen {generation}: best valid = {best_valid_side:.4f}")

    # Try repair on CMA-ES best
    cmaes_best = params_to_trees(es.result.xbest)
    cmaes_side = compute_side_length(cmaes_best)

    if has_any_overlap_strict(cmaes_best):
        repaired = repair_solution(cmaes_best)
        if not has_any_overlap_strict(repaired):
            repaired_side = compute_side_length(repaired)
            if repaired_side < best_valid_side:
                best_valid_side = repaired_side
                best_valid_trees = repaired
    else:
        if cmaes_side < best_valid_side:
            best_valid_side = cmaes_side
            best_valid_trees = cmaes_best

    is_valid = best_valid_trees is not None and not has_any_overlap_strict(best_valid_trees)

    if verbose:
        print(f"    Final: side={best_valid_side:.4f}, valid={is_valid}")

    return best_valid_side, best_valid_trees, is_valid


def run_multi_restart(n: int, initial_trees: list, restarts: int = 10,
                      max_evals: int = 20000, sigma0: float = 0.25) -> tuple:
    """Run multiple restarts with different initial configurations."""
    initial_side = compute_side_length(initial_trees)
    initial_valid = not has_any_overlap_strict(initial_trees)

    best_side = initial_side if initial_valid else float('inf')
    best_trees = initial_trees if initial_valid else None

    print(f"\n{'='*60}")
    print(f"Optimizing n={n} with {restarts} restarts, {max_evals} evals each")
    print(f"Initial side: {initial_side:.6f}, s^2/n: {initial_side**2/n:.6f}")

    for restart in range(restarts):
        print(f"\n  Restart {restart + 1}/{restarts}")

        # Perturb initial solution for restart > 0
        if restart == 0:
            start_trees = initial_trees
        else:
            start_trees = []
            for t in initial_trees:
                dx = random.uniform(-0.15, 0.15)
                dy = random.uniform(-0.15, 0.15)
                da = random.uniform(-45, 45)
                start_trees.append(Tree(t.x + dx, t.y + dy, (t.angle + da) % 360))

        # Vary sigma and penalty across restarts
        current_sigma = sigma0 * (0.8 + 0.4 * random.random())
        current_penalty = 5000.0 * (1 + 0.5 * restart)

        side, trees, valid = optimize_single_n(
            n, start_trees,
            max_evals=max_evals,
            sigma0=current_sigma,
            penalty_weight=current_penalty,
            verbose=True
        )

        if valid and side < best_side:
            best_side = side
            best_trees = trees
            improvement_pct = (initial_side - side) / initial_side * 100
            print(f"  >>> NEW BEST: {side:.6f} ({improvement_pct:.2f}% improvement)")

    return best_side, best_trees


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=list(range(11, 16)),
                       help='n values to optimize (default: 11-15)')
    parser.add_argument('--evals', type=int, default=20000,
                       help='Max evaluations per restart')
    parser.add_argument('--restarts', type=int, default=10,
                       help='Number of restarts')
    parser.add_argument('--sigma', type=float, default=0.25,
                       help='Initial step size')
    parser.add_argument('--input', type=str, default='submission_best.csv',
                       help='Input submission CSV')
    parser.add_argument('--output', type=str, default='python/gen116_medium_n_results.json',
                       help='Output file')
    args = parser.parse_args()

    print("Gen116: Medium-N CMA-ES Optimization")
    print(f"n values: {args.n}")
    print(f"Max evals per restart: {args.evals}")
    print(f"Restarts: {args.restarts}")
    print(f"Sigma: {args.sigma}")
    print(f"Total evals per n: {args.evals * args.restarts:,}")
    print(f"Overlap tolerance: {OVERLAP_TOLERANCE}")

    results = {}
    improvements = []

    for n in args.n:
        # Load from submission
        try:
            initial_trees = load_trees_from_submission(args.input, n)
        except Exception as e:
            print(f"Error loading n={n}: {e}")
            continue

        current_side = compute_side_length(initial_trees)

        start = time.time()
        best_side, best_trees = run_multi_restart(
            n, initial_trees,
            restarts=args.restarts,
            max_evals=args.evals,
            sigma0=args.sigma
        )
        elapsed = time.time() - start

        # Final validation
        if best_trees is None or has_any_overlap_strict(best_trees):
            print(f"\n  WARNING: No valid solution found! Keeping original.")
            best_side = current_side
            best_trees = initial_trees

        results[str(n)] = {
            'side': best_side,
            'trees': [[t.x, t.y, t.angle] for t in best_trees],
            'original_side': current_side
        }

        if best_side < current_side:
            improvement = (current_side - best_side) / current_side * 100
            score_delta = (current_side**2 - best_side**2) / n
            improvements.append({
                'n': n,
                'old_side': current_side,
                'new_side': best_side,
                'improvement_pct': improvement,
                'score_delta': score_delta
            })
            print(f"\n  IMPROVED: {current_side:.6f} -> {best_side:.6f} ({improvement:.2f}%)")
            print(f"  Score delta: {score_delta:.6f}")
        else:
            print(f"\n  No improvement for n={n}")

        print(f"  Elapsed: {elapsed:.1f}s")

        # Save incrementally
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if improvements:
        total_score_delta = sum(i['score_delta'] for i in improvements)
        print(f"Improvements found: {len(improvements)}")
        for imp in improvements:
            print(f"  n={imp['n']}: {imp['old_side']:.4f} -> {imp['new_side']:.4f} "
                  f"({imp['improvement_pct']:.2f}%, score delta: {imp['score_delta']:.4f})")
        print(f"\nTotal score improvement: {total_score_delta:.4f}")
    else:
        print("No improvements found.")


if __name__ == '__main__':
    main()
