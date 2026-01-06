#!/usr/bin/env python3
"""
Gen117 Pattern-Based CMA-ES Optimizer

Key innovations:
1. Multiple pattern initializations (radial, hexagonal, diagonal, spiral)
2. CMA-ES global search from each pattern
3. Strict segment-intersection validation (matches Kaggle checker)
4. Focus on medium n (20-50) where patterns may help escape local optima

Usage:
    python3 python/gen117_patterns.py --n 20 21 22 23 24 25 --evals 10000
"""

import math
import numpy as np
import cma
import csv
import json
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]


@dataclass
class Tree:
    x: float
    y: float
    angle: float

    def get_vertices(self) -> List[Tuple[float, float]]:
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return [
            (vx * cos_a - vy * sin_a + self.x,
             vx * sin_a + vy * cos_a + self.y)
            for vx, vy in TREE_VERTICES
        ]


def compute_side(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    all_verts = []
    for t in trees:
        all_verts.extend(t.get_vertices())
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


# ============ STRICT VALIDATION (matches Kaggle) ============

def ccw(A, B, C):
    """Counter-clockwise test."""
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    """Check if segment AB intersects segment CD (proper intersection only)."""
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


def polygons_overlap_strict(poly1, poly2) -> bool:
    """
    Strict overlap check using segment intersection.
    This matches Kaggle's validator exactly.
    """
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i + 1) % n1],
                                  poly2[j], poly2[(j + 1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True

    return False


def has_any_overlap_strict(trees: List[Tree]) -> bool:
    """Check if any trees overlap using strict validation."""
    polys = [t.get_vertices() for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_strict(polys[i], polys[j]):
                return True
    return False


# ============ PATTERN INITIALIZATIONS ============

def radial_pattern(n: int, radius_scale: float = 0.5) -> List[Tree]:
    """
    Radial/circular pattern: trees arranged in concentric circles.
    Good for symmetric configurations.
    """
    trees = []
    # Estimate radius based on n
    base_radius = radius_scale * math.sqrt(n) * 0.6

    # Distribute in rings
    placed = 0
    ring = 0
    while placed < n:
        if ring == 0:
            # Center tree
            trees.append(Tree(0, 0, 0))
            placed += 1
        else:
            # Trees in this ring
            ring_radius = base_radius * ring / max(1, math.sqrt(n) / 2)
            trees_in_ring = min(6 * ring, n - placed)  # hexagonal packing
            for i in range(trees_in_ring):
                theta = 2 * math.pi * i / trees_in_ring + (ring % 2) * math.pi / trees_in_ring
                x = ring_radius * math.cos(theta)
                y = ring_radius * math.sin(theta)
                # Angle follows radial direction + 90 degree offset pattern
                angle = (math.degrees(theta) + (i % 4) * 90) % 360
                trees.append(Tree(x, y, angle))
                placed += 1
                if placed >= n:
                    break
        ring += 1

    return trees[:n]


def hexagonal_pattern(n: int, spacing: float = 0.65) -> List[Tree]:
    """
    Hexagonal close-packing pattern.
    Optimal for dense packing of circles, may work for trees.
    """
    trees = []
    # Hexagonal grid
    cols = int(math.ceil(math.sqrt(n * 4 / 3)))
    rows = int(math.ceil(n / cols))

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            # Hexagonal offset
            x_offset = (row % 2) * spacing / 2
            x = col * spacing + x_offset
            y = row * spacing * 0.866  # sqrt(3)/2 for hex packing
            # Alternating angles for non-overlap
            angle = ((row + col) % 4) * 90
            trees.append(Tree(x, y, angle))
            idx += 1

    return trees[:n]


def diagonal_pattern(n: int, spacing: float = 0.7) -> List[Tree]:
    """
    Diagonal grid pattern with alternating angles.
    Trees arranged along diagonal lines.
    """
    trees = []
    # Arrange along diagonals
    side = int(math.ceil(math.sqrt(n)))

    idx = 0
    for diag in range(2 * side - 1):
        for i in range(max(0, diag - side + 1), min(diag + 1, side)):
            if idx >= n:
                break
            j = diag - i
            x = i * spacing
            y = j * spacing
            angle = ((i + j) % 4) * 90 + 45  # offset by 45
            trees.append(Tree(x, y, angle))
            idx += 1

    return trees[:n]


def spiral_pattern(n: int, turns: float = 2.5) -> List[Tree]:
    """
    Fibonacci/golden spiral pattern.
    Trees spread outward in a spiral.
    """
    trees = []
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    for i in range(n):
        # Golden angle spacing
        theta = 2 * math.pi * i / phi**2
        # Radius grows with sqrt for even distribution
        r = 0.3 * math.sqrt(i + 1)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        # Angle follows spiral direction
        angle = (math.degrees(theta) + (i % 4) * 90) % 360
        trees.append(Tree(x, y, angle))

    return trees


def grid_90_pattern(n: int, spacing: float = 0.6) -> List[Tree]:
    """
    Square grid with 90-degree angle rotations (like n=4 solution).
    """
    trees = []
    side = int(math.ceil(math.sqrt(n)))

    idx = 0
    for row in range(side):
        for col in range(side):
            if idx >= n:
                break
            x = col * spacing
            y = row * spacing
            # 90-degree rotation pattern: 0, 90, 180, 270
            angle_idx = (row * 2 + col) % 4  # Creates checkerboard of angles
            angle = angle_idx * 90
            trees.append(Tree(x, y, angle))
            idx += 1

    return trees[:n]


# ============ CMA-ES OPTIMIZATION ============

def trees_to_params(trees: List[Tree]) -> np.ndarray:
    """Convert trees to flat parameter array [x0, y0, a0, x1, y1, a1, ...]"""
    params = []
    for t in trees:
        params.extend([t.x, t.y, t.angle / 360.0])  # Normalize angle to [0,1)
    return np.array(params)


def params_to_trees(params: np.ndarray) -> List[Tree]:
    """Convert flat parameter array back to trees."""
    n = len(params) // 3
    trees = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        angle = (params[3 * i + 2] % 1.0) * 360.0
        trees.append(Tree(x, y, angle))
    return trees


def compute_overlap_penalty_fast(trees: List[Tree]) -> float:
    """Fast overlap penalty using bounding box pre-filter."""
    from shapely.geometry import Polygon
    from shapely import affinity

    TREE_POLY = Polygon(TREE_VERTICES)

    polys = []
    for t in trees:
        rotated = affinity.rotate(TREE_POLY, t.angle, origin=(0, 0))
        translated = affinity.translate(rotated, t.x, t.y)
        polys.append(translated)

    total_overlap = 0.0
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                total_overlap += inter.area

    return total_overlap


def objective_function(params: np.ndarray, penalty_weight: float = 100.0) -> float:
    """CMA-ES objective: minimize side + penalty * overlap."""
    trees = params_to_trees(params)
    side = compute_side(trees)
    overlap = compute_overlap_penalty_fast(trees)
    return side + penalty_weight * overlap


def optimize_with_pattern(
    n: int,
    pattern_init: List[Tree],
    pattern_name: str,
    max_evals: int = 10000,
    sigma0: float = 0.15,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]]]:
    """
    Run CMA-ES from a pattern initialization.
    Returns (best_valid_side, best_valid_trees) or (inf, None) if no valid solution.
    """
    initial_params = trees_to_params(pattern_init)
    dim = len(initial_params)

    options = {
        'maxfevals': max_evals,
        'verbose': -9,
        'popsize': max(8, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-9,
        'tolx': 1e-9,
    }

    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    best_valid_side = float('inf')
    best_valid_trees = None

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)

        # Check each solution for validity
        for x in solutions:
            trees = params_to_trees(x)
            # Use strict validation
            if not has_any_overlap_strict(trees):
                side = compute_side(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = [Tree(t.x, t.y, t.angle) for t in trees]

        generation += 1
        if verbose and generation % 25 == 0:
            print(f"    {pattern_name} gen {generation}: best_valid={best_valid_side:.4f}")

    return best_valid_side, best_valid_trees


def optimize_n(
    n: int,
    current_best: Optional[Tuple[float, List[Tuple]]] = None,
    max_evals_per_pattern: int = 10000,
    verbose: bool = True
) -> Tuple[float, Optional[List[Tree]], str]:
    """
    Optimize n using multiple pattern initializations.
    Returns (best_side, best_trees, winning_pattern).
    """
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Optimizing n={n}")
        print(f"{'=' * 50}")

    best_side = float('inf')
    best_trees = None
    best_pattern = "none"

    # Start with current best if available
    if current_best:
        current_side, current_trees_data = current_best
        best_side = current_side
        best_trees = [Tree(t[0], t[1], t[2]) for t in current_trees_data]
        best_pattern = "current"
        if verbose:
            print(f"Current best: {current_side:.4f}")

    # Define patterns to try (reduced set for speed)
    patterns = [
        ("radial", lambda: radial_pattern(n)),
        ("hexagonal", lambda: hexagonal_pattern(n)),
        ("spiral", lambda: spiral_pattern(n)),
        ("grid_90", lambda: grid_90_pattern(n)),
    ]

    # Also try refining current best if available
    if current_best:
        current_trees = [Tree(t[0], t[1], t[2]) for t in current_best[1]]
        patterns.insert(0, ("refine_current", lambda: current_trees))

    for pattern_name, pattern_fn in patterns:
        if verbose:
            print(f"\n  Trying {pattern_name}...")

        try:
            init_trees = pattern_fn()
        except Exception as e:
            if verbose:
                print(f"    Pattern init failed: {e}")
            continue

        # Check if initial pattern is valid (unlikely but worth checking)
        if not has_any_overlap_strict(init_trees):
            init_side = compute_side(init_trees)
            if verbose:
                print(f"    Initial: {init_side:.4f} (valid)")
        else:
            if verbose:
                print(f"    Initial: overlapping, will optimize")

        # Run CMA-ES from this pattern
        side, trees = optimize_with_pattern(
            n, init_trees, pattern_name,
            max_evals=max_evals_per_pattern,
            verbose=verbose
        )

        if trees and side < best_side:
            improvement = best_side - side
            if verbose:
                print(f"    *** NEW BEST: {side:.4f} (improved by {improvement:.4f})")
            best_side = side
            best_trees = trees
            best_pattern = pattern_name

    return best_side, best_trees, best_pattern


def load_current_submission(csv_path: str) -> dict:
    """Load current submission and extract solutions per n."""
    groups = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
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

    parser = argparse.ArgumentParser(description='Gen117 Pattern-Based CMA-ES Optimizer')
    parser.add_argument('--n', type=int, nargs='+', default=[20, 21, 22, 23, 24, 25],
                        help='n values to optimize')
    parser.add_argument('--evals', type=int, default=10000,
                        help='Max evaluations per pattern')
    parser.add_argument('--submission', type=str, default='submission_best.csv',
                        help='Current submission CSV')
    parser.add_argument('--output', type=str, default='python/gen117_optimized.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("Gen117 Pattern-Based CMA-ES Optimizer")
    print("=" * 50)
    print(f"Targets: n={args.n}")
    print(f"Evals per pattern: {args.evals}")

    # Load current solutions
    print(f"\nLoading {args.submission}...")
    current = load_current_submission(args.submission)
    print(f"Loaded {len(current)} solutions")

    results = {}
    improvements = []

    for n in args.n:
        current_best = current.get(n)
        if current_best:
            print(f"\nCurrent n={n}: side={current_best[0]:.4f}, score={current_best[0]**2/n:.4f}")

        side, trees, pattern = optimize_n(
            n,
            current_best=current_best,
            max_evals_per_pattern=args.evals,
            verbose=True
        )

        if trees:
            old_side = current_best[0] if current_best else float('inf')
            improvement = old_side - side
            score_delta = (old_side**2 - side**2) / n

            results[str(n)] = {
                'side': side,
                'pattern': pattern,
                'trees': [(t.x, t.y, t.angle) for t in trees]
            }

            if improvement > 0.0001:
                # Verify with strict validation one more time
                if not has_any_overlap_strict(trees):
                    improvements.append((n, improvement, score_delta, pattern))
                    print(f"\n  IMPROVED n={n}: {old_side:.4f} -> {side:.4f}")
                    print(f"    Pattern: {pattern}")
                    print(f"    Side delta: {improvement:.4f}")
                    print(f"    Score delta: {score_delta:.4f}")
                else:
                    print(f"\n  WARNING: n={n} solution has overlaps! Keeping original.")
                    results[str(n)] = None
            else:
                print(f"\n  No improvement for n={n} (current is best)")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to {args.output}")

    # Summary
    if improvements:
        print("\n" + "=" * 50)
        print("IMPROVEMENTS FOUND")
        print("=" * 50)
        total_score_delta = 0
        for n, imp, sd, pattern in improvements:
            print(f"  n={n}: -{imp:.4f} side ({sd:.4f} score) via {pattern}")
            total_score_delta += sd
        print(f"\n  Total score improvement: {total_score_delta:.4f}")
    else:
        print("\n" + "=" * 50)
        print("No improvements found")
        print("=" * 50)


if __name__ == '__main__':
    main()
