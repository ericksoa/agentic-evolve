#!/usr/bin/env python3
"""
CMA-ES Optimizer for Santa 2025 Packing Competition

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to globally
optimize tree positions starting from Rust's greedy solution.

Strategy:
1. Load existing solution (from CSV or generate with Rust)
2. Encode as 3n parameters (x, y, angle per tree)
3. Optimize with soft overlap penalty
4. Return best valid configuration
"""

import math
import numpy as np
import cma
import csv
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely import affinity

# Tree polygon vertices
TREE_VERTICES = [
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
]

TREE_POLYGON = Polygon(TREE_VERTICES)


@dataclass
class Tree:
    x: float
    y: float
    angle: float  # degrees

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)

    def get_vertices(self) -> np.ndarray:
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        verts = np.array(TREE_VERTICES)
        rotated = np.zeros_like(verts)
        rotated[:, 0] = verts[:, 0] * cos_a - verts[:, 1] * sin_a + self.x
        rotated[:, 1] = verts[:, 0] * sin_a + verts[:, 1] * cos_a + self.y
        return rotated


def trees_to_params(trees: List[Tree]) -> np.ndarray:
    """Convert list of trees to flat parameter array."""
    params = []
    for t in trees:
        params.extend([t.x, t.y, t.angle / 360.0])  # Normalize angle to [0,1)
    return np.array(params)


def params_to_trees(params: np.ndarray) -> List[Tree]:
    """Convert flat parameter array to list of trees."""
    n = len(params) // 3
    trees = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        angle = (params[3*i + 2] % 1.0) * 360.0  # Wrap to [0, 360)
        trees.append(Tree(x, y, angle))
    return trees


def compute_side_length(trees: List[Tree]) -> float:
    """Compute bounding box side length."""
    if not trees:
        return 0.0

    all_verts = np.vstack([t.get_vertices() for t in trees])
    minx, miny = all_verts.min(axis=0)
    maxx, maxy = all_verts.max(axis=0)
    return max(maxx - minx, maxy - miny)


def compute_overlap_penalty(trees: List[Tree]) -> float:
    """Compute total overlap area penalty."""
    n = len(trees)
    total_overlap = 0.0

    polygons = [t.get_polygon() for t in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                total_overlap += intersection.area

    return total_overlap


def has_any_overlap(trees: List[Tree], tolerance: float = 1e-8) -> bool:
    """Check if any trees overlap."""
    n = len(trees)
    polygons = [t.get_polygon() for t in trees]

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > tolerance:
                    return True
    return False


def objective_function(params: np.ndarray, penalty_weight: float = 100.0) -> float:
    """
    Objective function for CMA-ES.

    Minimizes side length with penalty for overlaps.
    """
    trees = params_to_trees(params)
    side = compute_side_length(trees)
    overlap = compute_overlap_penalty(trees)

    return side + penalty_weight * overlap


def repair_solution(trees: List[Tree], max_iters: int = 1000) -> List[Tree]:
    """
    Repair overlapping solution by small perturbations.
    """
    import random

    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_overlap = compute_overlap_penalty(best_trees)

    if best_overlap < 1e-10:
        return best_trees

    for _ in range(max_iters):
        if best_overlap < 1e-10:
            break

        # Pick random tree and perturb
        idx = random.randint(0, len(best_trees) - 1)
        t = best_trees[idx]

        # Try small perturbation
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        da = random.choice([0, 15, -15, 45, -45])

        new_tree = Tree(t.x + dx, t.y + dy, (t.angle + da) % 360)
        candidate = best_trees[:idx] + [new_tree] + best_trees[idx+1:]

        candidate_overlap = compute_overlap_penalty(candidate)
        if candidate_overlap < best_overlap:
            best_trees = candidate
            best_overlap = candidate_overlap

    return best_trees


def optimize_with_cmaes(
    initial_trees: List[Tree],
    max_evals: int = 5000,
    sigma0: float = 0.1,
    penalty_weight: float = 100.0,
    verbose: bool = True
) -> Tuple[float, List[Tree]]:
    """
    Optimize tree positions using CMA-ES.

    Args:
        initial_trees: Starting solution
        max_evals: Maximum function evaluations
        sigma0: Initial step size
        penalty_weight: Weight for overlap penalty
        verbose: Print progress

    Returns:
        (side_length, optimized_trees)
    """
    n = len(initial_trees)
    dim = 3 * n

    initial_params = trees_to_params(initial_trees)
    initial_side = compute_side_length(initial_trees)

    if verbose:
        print(f"Starting CMA-ES optimization for n={n}")
        print(f"  Initial side: {initial_side:.4f}")
        print(f"  Max evals: {max_evals}")

    # CMA-ES options
    options = {
        'maxfevals': max_evals,
        'verbose': -9,  # Suppress output
        'popsize': max(8, 4 + int(3 * np.log(dim))),
        'tolfun': 1e-8,
        'tolx': 1e-8,
    }

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, options)

    best_valid_side = initial_side
    best_valid_trees = initial_trees

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_function(x, penalty_weight) for x in solutions]
        es.tell(solutions, fitnesses)

        # Check for valid solutions
        for x, f in zip(solutions, fitnesses):
            trees = params_to_trees(x)
            if not has_any_overlap(trees):
                side = compute_side_length(trees)
                if side < best_valid_side:
                    best_valid_side = side
                    best_valid_trees = trees

        generation += 1
        if verbose and generation % 20 == 0:
            print(f"  Gen {generation}: best valid = {best_valid_side:.4f}")

    # Get best from CMA-ES
    cmaes_best = params_to_trees(es.result.xbest)
    cmaes_side = compute_side_length(cmaes_best)

    if verbose:
        print(f"  CMA-ES finished: raw best = {cmaes_side:.4f}")

    # If best has overlaps, try to repair
    if has_any_overlap(cmaes_best):
        if verbose:
            print("  Repairing overlaps...")
        repaired = repair_solution(cmaes_best)
        if not has_any_overlap(repaired):
            repaired_side = compute_side_length(repaired)
            if repaired_side < best_valid_side:
                best_valid_side = repaired_side
                best_valid_trees = repaired

    if verbose:
        print(f"  Final best valid: {best_valid_side:.4f}")
        improvement = (initial_side - best_valid_side) / initial_side * 100
        print(f"  Improvement: {improvement:.2f}%")

    return best_valid_side, best_valid_trees


def load_trees_from_csv(csv_path: str, n: int) -> List[Tree]:
    """
    Load trees for a specific n from submission CSV.

    The CSV has consecutive trees: 1 for n=1, 2 for n=2, etc.
    """
    all_trees = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            angle = float(row['angle'])
            all_trees.append(Tree(x, y, angle))

    # Calculate starting index for n
    start_idx = sum(range(1, n))  # 0 + 1 + 2 + ... + (n-1)
    if start_idx + n > len(all_trees):
        raise ValueError(f"Not enough trees in CSV for n={n}")

    return all_trees[start_idx:start_idx + n]


def save_trees_to_json(trees: List[Tree], output_path: str, n: int, side: float):
    """Save trees to JSON file."""
    import json
    data = {
        'n': n,
        'side': side,
        'trees': [{'x': t.x, 'y': t.y, 'angle': t.angle} for t in trees]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CMA-ES optimizer for tree packing')
    parser.add_argument('n', type=int, help='Number of trees')
    parser.add_argument('--csv', type=str, help='Input CSV with initial solution')
    parser.add_argument('--max-evals', type=int, default=5000, help='Max function evaluations')
    parser.add_argument('--sigma', type=float, default=0.1, help='Initial step size')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    # Load initial solution
    if args.csv:
        print(f"Loading initial solution from {args.csv}")
        initial_trees = load_trees_from_csv(args.csv, args.n)
    else:
        # Generate random initial solution
        print("Generating random initial solution")
        import random
        initial_trees = []
        for i in range(args.n):
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            initial_trees.append(Tree(x, y, angle))

    initial_side = compute_side_length(initial_trees)
    initial_valid = not has_any_overlap(initial_trees)
    print(f"Initial: side={initial_side:.4f}, valid={initial_valid}")

    # Optimize
    best_side, best_trees = optimize_with_cmaes(
        initial_trees,
        max_evals=args.max_evals,
        sigma0=args.sigma
    )

    # Report
    final_valid = not has_any_overlap(best_trees)
    print(f"\nFinal: side={best_side:.4f}, valid={final_valid}")
    print(f"Score contribution: {best_side**2 / args.n:.4f}")

    if args.output:
        save_trees_to_json(best_trees, args.output, args.n, best_side)
        print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
