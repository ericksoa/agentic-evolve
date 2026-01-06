#!/usr/bin/env python3
"""
Gen115 Runner - CMA-ES with STRICT overlap handling.

Key improvements over Gen114:
1. 10x higher overlap penalty (2000 vs 200)
2. Stricter overlap tolerance (1e-9)
3. Multiple random restarts
4. Aggressive repair with validation
5. Continuous angle optimization
"""

import json
import sys
import time
import random
import math
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


def has_any_overlap_strict(trees: list, tolerance: float = OVERLAP_TOLERANCE) -> bool:
    """Strict overlap check with configurable tolerance."""
    n = len(trees)
    polygons = [t.get_polygon() for t in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > tolerance:
                    return True
    return False


def get_max_overlap(trees: list) -> float:
    """Get the maximum pairwise overlap area."""
    n = len(trees)
    polygons = [t.get_polygon() for t in trees]
    max_overlap = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                max_overlap = max(max_overlap, intersection.area)

    return max_overlap


def trees_to_params_continuous(trees: list) -> np.ndarray:
    """Convert trees to params with continuous angle."""
    params = []
    for t in trees:
        params.extend([t.x, t.y, t.angle / 360.0])
    return np.array(params)


def params_to_trees_continuous(params: np.ndarray) -> list:
    """Convert params to trees with continuous angle."""
    n = len(params) // 3
    trees = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        angle = (params[3*i + 2] % 1.0) * 360.0
        trees.append(Tree(x, y, angle))
    return trees


def objective_strict(params: np.ndarray, penalty_weight: float = 2000.0) -> float:
    """Strict objective function with very high overlap penalty."""
    trees = params_to_trees_continuous(params)
    side = compute_side_length(trees)
    overlap = compute_overlap_penalty(trees)

    # Exponential penalty for any overlap
    if overlap > 1e-10:
        overlap_penalty = penalty_weight * overlap + penalty_weight * (1 + overlap * 100)
    else:
        overlap_penalty = 0

    return side + overlap_penalty


def repair_solution_aggressive(trees: list, max_iters: int = 5000) -> list:
    """
    Aggressive repair with multiple strategies.
    """
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_overlap = get_max_overlap(best_trees)

    if best_overlap < OVERLAP_TOLERANCE:
        return best_trees

    print(f"    Repair: starting overlap = {best_overlap:.2e}")

    # Strategy 1: Small perturbations
    for iteration in range(max_iters):
        if best_overlap < OVERLAP_TOLERANCE:
            print(f"    Repair: fixed after {iteration} iterations")
            break

        idx = random.randint(0, len(best_trees) - 1)
        t = best_trees[idx]

        # Try progressively smaller perturbations
        scale = 0.1 * (1.0 - iteration / max_iters)
        dx = random.uniform(-scale, scale)
        dy = random.uniform(-scale, scale)
        da = random.uniform(-15, 15)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx+1:]

        candidate_overlap = get_max_overlap(candidate)
        # Accept if overlap decreases OR if no worse and side is smaller
        if candidate_overlap < best_overlap * 0.999:
            best_trees = candidate
            best_overlap = candidate_overlap

    # Strategy 2: Push apart overlapping trees
    if best_overlap >= OVERLAP_TOLERANCE:
        print(f"    Repair: trying push-apart strategy")
        polygons = [t.get_polygon() for t in best_trees]
        n = len(best_trees)

        for _ in range(1000):
            # Find overlapping pair
            for i in range(n):
                for j in range(i + 1, n):
                    if polygons[i].intersects(polygons[j]):
                        intersection = polygons[i].intersection(polygons[j])
                        if intersection.area > OVERLAP_TOLERANCE:
                            # Push trees apart
                            ci = polygons[i].centroid
                            cj = polygons[j].centroid
                            dx = cj.x - ci.x
                            dy = cj.y - ci.y
                            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
                            dx /= dist
                            dy /= dist

                            # Move each tree slightly apart
                            move = 0.01
                            ti = best_trees[i]
                            tj = best_trees[j]
                            best_trees[i] = Tree(ti.x - dx*move, ti.y - dy*move, ti.angle)
                            best_trees[j] = Tree(tj.x + dx*move, tj.y + dy*move, tj.angle)
                            polygons[i] = best_trees[i].get_polygon()
                            polygons[j] = best_trees[j].get_polygon()

            best_overlap = get_max_overlap(best_trees)
            if best_overlap < OVERLAP_TOLERANCE:
                print(f"    Repair: push-apart succeeded")
                break

    final_overlap = get_max_overlap(best_trees)
    print(f"    Repair: final overlap = {final_overlap:.2e}")
    return best_trees


def optimize_single_n(n: int, initial_trees: list, max_evals: int = 5000,
                      sigma0: float = 0.15, penalty_weight: float = 2000.0,
                      verbose: bool = True) -> tuple:
    """
    Optimize single n value with strict validation.

    Returns: (side, trees, is_valid)
    """
    dim = 3 * n
    initial_params = trees_to_params_continuous(initial_trees)
    initial_side = compute_side_length(initial_trees)
    initial_valid = not has_any_overlap_strict(initial_trees)

    if verbose:
        print(f"  Initial: side={initial_side:.4f}, valid={initial_valid}")

    # CMA-ES options
    options = {
        'maxfevals': max_evals,
        'verbose': -9,
        'popsize': max(12, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-9,
        'tolx': 1e-9,
    }

    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    # Track best VALID solution
    best_valid_side = initial_side if initial_valid else float('inf')
    best_valid_trees = initial_trees if initial_valid else None

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_strict(x, penalty_weight) for x in solutions]
        es.tell(solutions, fitnesses)

        # Check for strictly valid solutions
        for x, f in zip(solutions, fitnesses):
            trees = params_to_trees_continuous(x)
            if not has_any_overlap_strict(trees):
                side = compute_side_length(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = trees

        generation += 1
        if verbose and generation % 50 == 0:
            print(f"    Gen {generation}: best valid = {best_valid_side:.4f}")

    # Get CMA-ES best and try to repair
    cmaes_best = params_to_trees_continuous(es.result.xbest)
    cmaes_side = compute_side_length(cmaes_best)

    if verbose:
        max_olap = get_max_overlap(cmaes_best)
        print(f"  CMA-ES raw: side={cmaes_side:.4f}, max_overlap={max_olap:.2e}")

    # Aggressive repair if needed
    if has_any_overlap_strict(cmaes_best):
        repaired = repair_solution_aggressive(cmaes_best)
        if not has_any_overlap_strict(repaired):
            repaired_side = compute_side_length(repaired)
            if repaired_side < best_valid_side:
                best_valid_side = repaired_side
                best_valid_trees = repaired
                if verbose:
                    print(f"  Repair successful: {repaired_side:.4f}")
    else:
        if cmaes_side < best_valid_side:
            best_valid_side = cmaes_side
            best_valid_trees = cmaes_best

    is_valid = best_valid_trees is not None and not has_any_overlap_strict(best_valid_trees)

    if verbose:
        print(f"  Final: side={best_valid_side:.4f}, valid={is_valid}")

    return best_valid_side, best_valid_trees, is_valid


def run_multi_restart(n: int, initial_trees: list, restarts: int = 3,
                      max_evals: int = 3000, sigma0: float = 0.15) -> tuple:
    """
    Run multiple restarts with different initial configurations.
    """
    initial_side = compute_side_length(initial_trees)
    initial_valid = not has_any_overlap_strict(initial_trees)

    best_side = initial_side if initial_valid else float('inf')
    best_trees = initial_trees if initial_valid else None

    print(f"\n{'='*60}")
    print(f"Optimizing n={n} with {restarts} restarts")
    print(f"Initial side: {initial_side:.6f}")

    for restart in range(restarts):
        print(f"\n  Restart {restart + 1}/{restarts}")

        # Perturb initial solution for restart > 0
        if restart == 0:
            start_trees = initial_trees
        else:
            start_trees = []
            for t in initial_trees:
                dx = random.uniform(-0.1, 0.1)
                dy = random.uniform(-0.1, 0.1)
                da = random.uniform(-30, 30)
                start_trees.append(Tree(t.x + dx, t.y + dy, (t.angle + da) % 360))

        side, trees, valid = optimize_single_n(
            n, start_trees,
            max_evals=max_evals,
            sigma0=sigma0 * (1 + 0.2 * restart),  # Vary sigma
            penalty_weight=2000.0 * (1 + restart),  # Increase penalty
            verbose=True
        )

        if valid and side < best_side:
            best_side = side
            best_trees = trees
            print(f"  NEW BEST: {side:.6f}")

    return best_side, best_trees


def load_current_best():
    """Load current best solutions."""
    with open('python/optimized_small_n.json') as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[7],
                       help='n values to optimize')
    parser.add_argument('--evals', type=int, default=5000,
                       help='Max evaluations per restart')
    parser.add_argument('--restarts', type=int, default=3,
                       help='Number of restarts')
    parser.add_argument('--sigma', type=float, default=0.15,
                       help='Initial step size')
    parser.add_argument('--output', type=str, default='python/gen115_optimized.json',
                       help='Output file')
    args = parser.parse_args()

    print("Gen115: Strict CMA-ES Optimization")
    print(f"n values: {args.n}")
    print(f"Max evals per restart: {args.evals}")
    print(f"Restarts: {args.restarts}")
    print(f"Sigma: {args.sigma}")
    print(f"Overlap tolerance: {OVERLAP_TOLERANCE}")

    current_best = load_current_best()
    results = {}
    improvements = []

    for n in args.n:
        n_str = str(n)
        if n_str not in current_best:
            print(f"No current solution for n={n}, skipping")
            continue

        current = current_best[n_str]
        current_side = current['side']
        trees_data = current['trees']

        initial_trees = [Tree(t[0], t[1], t[2]) for t in trees_data]

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

        results[n_str] = {
            'side': best_side,
            'trees': [[t.x, t.y, t.angle] for t in best_trees]
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

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

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
