#!/usr/bin/env python3
"""
Simulated Annealing optimizer for small n values.

More efficient than exhaustive grid search.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple
import json
import sys

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE = Polygon(TREE_VERTICES)


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_poly(self) -> Polygon:
        rotated = affinity.rotate(TREE, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_side(trees: List[Tree]) -> float:
    all_coords = []
    for t in trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap_strict(trees: List[Tree], tol: float = 1e-9) -> bool:
    """Strict overlap check."""
    polys = [t.get_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def random_valid_config(n: int, max_tries: int = 10000) -> List[Tree]:
    """Generate a random valid (non-overlapping) configuration."""
    for _ in range(max_tries):
        trees = []
        bound = 0.3 * n
        for _ in range(n):
            x = np.random.uniform(-bound, bound)
            y = np.random.uniform(-bound, bound)
            a = np.random.uniform(0, 360)
            trees.append(Tree(x, y, a))
        if not has_overlap_strict(trees):
            return trees
    return None


def simulated_annealing(n: int, iterations: int = 100000,
                         initial_config: List[Tree] = None,
                         verbose: bool = True) -> Tuple[float, List[Tree]]:
    """Run simulated annealing to minimize the bounding side."""

    # Initialize
    if initial_config is None:
        trees = random_valid_config(n)
        if trees is None:
            print(f"  Could not find valid initial config for n={n}")
            return float('inf'), None
    else:
        trees = [Tree(t.x, t.y, t.angle) for t in initial_config]

    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    current_side = compute_side(trees)
    best_side = current_side

    # SA parameters
    T0 = 0.5  # Initial temperature
    Tf = 0.001  # Final temperature
    alpha = (Tf / T0) ** (1 / iterations)

    T = T0
    accepted = 0
    improved = 0

    for i in range(iterations):
        # Pick a random tree to perturb
        idx = np.random.randint(0, n)
        t = trees[idx]

        # Perturbation scales with temperature
        scale = max(0.01, T * 2)
        dx = np.random.normal(0, scale * 0.1)
        dy = np.random.normal(0, scale * 0.1)
        da = np.random.normal(0, scale * 10)

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = trees[:idx] + [new_tree] + trees[idx + 1:]

        if has_overlap_strict(candidate):
            T *= alpha
            continue

        new_side = compute_side(candidate)
        delta = new_side - current_side

        # Accept or reject
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            trees = candidate
            current_side = new_side
            accepted += 1

            if new_side < best_side:
                best_side = new_side
                best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
                improved += 1

        T *= alpha

        if verbose and (i + 1) % 20000 == 0:
            print(f"    {i+1}/{iterations}: best={best_side:.4f}, curr={current_side:.4f}, T={T:.4f}")

    if verbose:
        print(f"  Accepted: {accepted}/{iterations}, Improved: {improved}")

    return best_side, best_trees


def multi_start_sa(n: int, restarts: int = 10, iterations_per: int = 50000,
                    verbose: bool = True) -> Tuple[float, List[Tree]]:
    """Run SA multiple times from different starting points."""

    best_side = float('inf')
    best_trees = None

    for r in range(restarts):
        if verbose:
            print(f"  Restart {r+1}/{restarts}...")

        side, trees = simulated_annealing(n, iterations_per, verbose=False)

        if trees and side < best_side:
            best_side = side
            best_trees = trees
            if verbose:
                print(f"    -> New best: {best_side:.4f}")

    return best_side, best_trees


def solve_with_base(n: int, base_trees: List[Tree], iterations: int = 100000,
                    verbose: bool = True) -> Tuple[float, List[Tree]]:
    """Add one tree to existing solution, then refine all."""

    if verbose:
        print(f"  Starting from {len(base_trees)} trees, adding 1...")

    # Get bounds of existing packing
    all_coords = []
    for t in base_trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    best_side = float('inf')
    best_trees = None

    # Grid search for new tree position
    margin = 0.8
    angles = np.arange(0, 360, 15)

    count = 0
    for dx in np.arange(min_x - margin, max_x + margin, 0.04):
        for dy in np.arange(min_y - margin, max_y + margin, 0.04):
            for angle in angles:
                new_tree = Tree(dx, dy, angle)
                candidate = base_trees + [new_tree]

                if has_overlap_strict(candidate):
                    continue

                side = compute_side(candidate)
                if side < best_side:
                    best_side = side
                    best_trees = [Tree(t.x, t.y, t.angle) for t in candidate]
                    count += 1

    if best_trees:
        if verbose:
            print(f"  Found {count} valid configs, best: {best_side:.4f}")

        # Refine all trees together
        best_side, best_trees = simulated_annealing(n, iterations, best_trees, verbose)

    return best_side, best_trees


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[4, 7, 8], help='n values to optimize')
    parser.add_argument('--restarts', type=int, default=5, help='Number of restarts')
    parser.add_argument('--iterations', type=int, default=50000, help='Iterations per restart')
    parser.add_argument('--output', type=str, default='python/sa_optimized.json', help='Output JSON')
    parser.add_argument('--existing-json', type=str, help='Use existing solutions as base')

    args = parser.parse_args()

    # Load existing solutions if provided
    existing = {}
    if args.existing_json:
        with open(args.existing_json) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing solutions")

    results = {}

    for n in args.n:
        print(f"\n=== Optimizing n={n} ===")

        # Try multi-start SA
        side1, trees1 = multi_start_sa(n, args.restarts, args.iterations, verbose=True)

        # Also try extending from n-1 if available
        side2 = float('inf')
        trees2 = None
        n_prev = str(n - 1)
        if n_prev in existing:
            base_trees = [Tree(t[0], t[1], t[2]) for t in existing[n_prev]['trees']]
            side2, trees2 = solve_with_base(n, base_trees, args.iterations, verbose=True)

        # Take the best
        if side1 < side2:
            best_side, best_trees = side1, trees1
            method = "multi-start SA"
        else:
            best_side, best_trees = side2, trees2
            method = "extend from n-1"

        if best_trees:
            results[str(n)] = {
                'side': best_side,
                'trees': [(t.x, t.y, t.angle) for t in best_trees],
                'method': method
            }
            print(f"\nn={n}: best side={best_side:.4f} ({method})")
        else:
            print(f"\nn={n}: FAILED to find valid solution")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
