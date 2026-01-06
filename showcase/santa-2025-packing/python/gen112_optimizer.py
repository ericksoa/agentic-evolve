#!/usr/bin/env python3
"""
Gen112 SA Optimizer with pattern-based initialization.

Key improvements:
1. Pattern-based initialization (circular with 90° angle offsets)
2. Stricter overlap tolerance (1e-9)
3. Longer runs with restarts
4. Adaptive temperature schedules
"""

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import sys

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE = Polygon(TREE_VERTICES)
TREE_BOUNDS = TREE.bounds  # (minx, miny, maxx, maxy)


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_poly(self) -> Polygon:
        rotated = affinity.rotate(TREE, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_side(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    all_coords = []
    for t in trees:
        all_coords.extend(list(t.get_poly().exterior.coords))
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap_strict(trees: List[Tree], tol: float = 1e-9) -> bool:
    """Strict overlap check with very small tolerance."""
    polys = [t.get_poly() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if inter.area > tol:
                    return True
    return False


def circular_init(n: int, base_radius: float = 0.5, base_angle: float = 0) -> List[Tree]:
    """
    Initialize trees in a circular pattern with 90° offset rotations.

    This mimics the successful n=4 pattern.
    """
    trees = []
    for i in range(n):
        # Position in a circle
        theta = 2 * np.pi * i / n
        r = base_radius * (0.3 + 0.1 * n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Rotation: offset by 90° increments, like n=4 pattern
        # Pattern: 60°, 60°, 150°, 240° (90° jumps)
        angle_offset = (i % 4) * 90
        angle = (base_angle + angle_offset) % 360

        trees.append(Tree(x, y, angle))

    return trees


def grid_init(n: int, cols: int = None, angle_pattern: List[float] = None) -> List[Tree]:
    """Initialize trees in a grid pattern with alternating angles."""
    if cols is None:
        cols = max(2, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))

    if angle_pattern is None:
        angle_pattern = [0, 90, 180, 270]

    spacing = 0.6  # Tree width approx
    trees = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = c * spacing
            y = r * spacing
            angle = angle_pattern[idx % len(angle_pattern)]
            trees.append(Tree(x, y, angle))
            idx += 1

    return trees


def random_valid_config(n: int, max_tries: int = 5000) -> Optional[List[Tree]]:
    """Generate a random valid (non-overlapping) configuration."""
    for _ in range(max_tries):
        trees = []
        bound = 0.3 * n
        for _ in range(n):
            x = np.random.uniform(-bound, bound)
            y = np.random.uniform(-bound, bound)
            a = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            a += np.random.uniform(-10, 10)  # Small perturbation
            trees.append(Tree(x, y, a % 360))
        if not has_overlap_strict(trees):
            return trees
    return None


def simulated_annealing(
    trees: List[Tree],
    iterations: int = 100000,
    T0: float = 0.5,
    Tf: float = 0.0001,
    verbose: bool = False
) -> Tuple[float, List[Tree]]:
    """Run simulated annealing to minimize the bounding side."""
    n = len(trees)
    trees = [Tree(t.x, t.y, t.angle) for t in trees]

    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    current_side = compute_side(trees)
    best_side = current_side

    alpha = (Tf / T0) ** (1 / iterations)
    T = T0
    accepted = 0

    for i in range(iterations):
        idx = np.random.randint(0, n)
        t = trees[idx]

        # Perturbation scales with temperature
        scale = max(0.005, T * 2)
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

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            trees = candidate
            current_side = new_side
            accepted += 1

            if new_side < best_side:
                best_side = new_side
                best_trees = [Tree(t.x, t.y, t.angle) for t in trees]

        T *= alpha

        if verbose and (i + 1) % 50000 == 0:
            print(f"    {i+1}/{iterations}: best={best_side:.4f}, T={T:.6f}")

    return best_side, best_trees


def optimize_n(
    n: int,
    current_best: Optional[Tuple[float, List[Tuple]]] = None,
    restarts: int = 10,
    iterations_per: int = 100000,
    verbose: bool = True
) -> Tuple[float, List[Tree]]:
    """
    Optimize a single n value with multiple strategies.
    """
    if verbose:
        print(f"\n=== Optimizing n={n} ===")

    best_side = float('inf')
    best_trees = None

    # If we have a current best, start by refining it
    if current_best:
        current_side, current_trees_data = current_best
        current_trees = [Tree(t[0], t[1], t[2]) for t in current_trees_data]
        if verbose:
            print(f"  Refining current best ({current_side:.4f})...")
        side, trees = simulated_annealing(current_trees, iterations_per * 2, verbose=False)
        if trees and side < best_side:
            best_side = side
            best_trees = trees
            if verbose:
                print(f"    -> Refined: {side:.4f}")

    # Try pattern-based initializations
    for base_angle in [0, 45, 60, 90]:
        init_trees = circular_init(n, base_angle=base_angle)
        if not has_overlap_strict(init_trees):
            if verbose:
                print(f"  Circular init (base={base_angle}°)...")
            side, trees = simulated_annealing(init_trees, iterations_per, verbose=False)
            if trees and side < best_side:
                best_side = side
                best_trees = trees
                if verbose:
                    print(f"    -> Found: {side:.4f}")

    # Try grid initializations
    for angle_pattern in [[0, 90, 180, 270], [45, 135, 225, 315], [60, 150, 240, 330]]:
        init_trees = grid_init(n, angle_pattern=angle_pattern)
        if not has_overlap_strict(init_trees):
            if verbose:
                print(f"  Grid init (angles={angle_pattern[:2]}...)...")
            side, trees = simulated_annealing(init_trees, iterations_per, verbose=False)
            if trees and side < best_side:
                best_side = side
                best_trees = trees
                if verbose:
                    print(f"    -> Found: {side:.4f}")

    # Random multi-start
    for r in range(restarts):
        init = random_valid_config(n)
        if init is None:
            continue
        side, trees = simulated_annealing(init, iterations_per, verbose=False)
        if trees and side < best_side:
            best_side = side
            best_trees = trees
            if verbose:
                print(f"  Random restart {r+1}: {side:.4f} *new best*")

    if verbose:
        print(f"  Best for n={n}: {best_side:.4f}")

    return best_side, best_trees


def load_current_submission(csv_path: str) -> dict:
    """Load current submission and extract solutions per n."""
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        groups = {}
        for row in reader:
            id_parts = row['id'].split('_')
            n = int(id_parts[0])
            x = float(row['x'].lstrip('s'))
            y = float(row['y'].lstrip('s'))
            deg = float(row['deg'].lstrip('s'))
            if n not in groups:
                groups[n] = []
            groups[n].append((x, y, deg))

    results = {}
    for n, trees_data in groups.items():
        trees = [Tree(x, y, a) for x, y, a in trees_data]
        side = compute_side(trees)
        results[n] = (side, trees_data)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[5, 6, 7, 8], help='n values to optimize')
    parser.add_argument('--restarts', type=int, default=10, help='Random restarts')
    parser.add_argument('--iterations', type=int, default=100000, help='Iterations per run')
    parser.add_argument('--submission', type=str, default='submission_best.csv', help='Current submission')
    parser.add_argument('--output', type=str, default='python/gen112_optimized.json', help='Output JSON')

    args = parser.parse_args()

    # Load current solutions
    print("Loading current submission...")
    current = load_current_submission(args.submission)
    print(f"Loaded {len(current)} solutions")

    results = {}
    improvements = []

    for n in args.n:
        current_best = current.get(n)
        if current_best:
            print(f"\nCurrent n={n}: side={current_best[0]:.4f}, score={current_best[0]**2/n:.4f}")

        side, trees = optimize_n(
            n,
            current_best=current_best,
            restarts=args.restarts,
            iterations_per=args.iterations,
            verbose=True
        )

        if trees:
            old_side = current_best[0] if current_best else float('inf')
            improvement = old_side - side
            score_delta = (old_side**2 - side**2) / n

            results[str(n)] = {
                'side': side,
                'trees': [(t.x, t.y, t.angle) for t in trees]
            }

            if improvement > 0.0001:
                improvements.append((n, improvement, score_delta))
                print(f"  IMPROVED: {old_side:.4f} -> {side:.4f} (delta={improvement:.4f}, score_delta={score_delta:.4f})")
            else:
                print(f"  No improvement (current is better)")
        else:
            print(f"  Failed to find valid solution")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    if improvements:
        print("\n=== Improvements ===")
        total_score_delta = 0
        for n, imp, sd in improvements:
            print(f"  n={n}: -{imp:.4f} side ({sd:.4f} score)")
            total_score_delta += sd
        print(f"  Total score improvement: {total_score_delta:.4f}")


if __name__ == '__main__':
    main()
