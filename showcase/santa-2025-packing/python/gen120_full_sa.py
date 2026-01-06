#!/usr/bin/env python3
"""
Gen120: Full-Configuration Simulated Annealing

Key insight: Local refinement (Gen118-119) can only find local optima.
With 23% gap to top scores, we need global search.

This SA optimizer:
1. Considers ALL trees together (not one at a time)
2. Makes random perturbations to random trees
3. Uses Metropolis criterion to accept worse moves (escape local minima)
4. Gradually lowers temperature to converge

Move types:
- Position perturbation: random dx, dy (scale decreases with T)
- Angle perturbation: random da (scale decreases with T)
- Swap two trees (exchange positions/angles)
- Rotate all trees by same offset
"""

import math
import csv
import random
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict
from copy import deepcopy
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
import numpy as np

# Tree polygon vertices
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

TREE_POLYGON = Polygon(TREE_VERTICES)
TREE_AREA = TREE_POLYGON.area  # ~0.35

@dataclass
class Tree:
    x: float
    y: float
    angle: float  # degrees

    def get_vertices(self) -> List[Tuple[float, float]]:
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [(vx * cos_a - vy * sin_a + self.x,
                 vx * sin_a + vy * cos_a + self.y) for vx, vy in TREE_VERTICES]

    def get_polygon(self) -> Polygon:
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        return affinity.translate(rotated, self.x, self.y)


def compute_bounding_box(trees: List[Tree]) -> Tuple[float, float, float, float]:
    """Compute bounding box (min_x, min_y, max_x, max_y)."""
    if not trees:
        return 0.0, 0.0, 0.0, 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for tree in trees:
        for vx, vy in tree.get_vertices():
            min_x = min(min_x, vx)
            min_y = min(min_y, vy)
            max_x = max(max_x, vx)
            max_y = max(max_y, vy)

    return min_x, min_y, max_x, max_y


def compute_side_length(trees: List[Tree]) -> float:
    """Compute bounding square side length."""
    min_x, min_y, max_x, max_y = compute_bounding_box(trees)
    return max(max_x - min_x, max_y - min_y)


def ccw(A, B, C):
    """Counter-clockwise test."""
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect_strict(A, B, C, D):
    """Check if segment AB intersects segment CD (strict)."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon_strict(point, polygon):
    """Ray casting algorithm."""
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


def polygons_overlap_strict(poly1, poly2):
    """Check if two polygons overlap (strict segment intersection check)."""
    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        for j in range(n2):
            if segments_intersect_strict(poly1[i], poly1[(i+1) % n1],
                                         poly2[j], poly2[(j+1) % n2]):
                return True

    # Check if any vertex is inside the other polygon
    for v in poly1:
        if point_in_polygon_strict(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon_strict(v, poly1):
            return True

    return False


def has_any_overlap_fast(trees: List[Tree]) -> bool:
    """Fast Shapely-based overlap check."""
    n = len(trees)
    if n <= 1:
        return False

    polygons = [t.get_polygon() for t in trees]
    tree_idx = STRtree(polygons)

    for i, poly_i in enumerate(polygons):
        candidates = tree_idx.query(poly_i)
        for j in candidates:
            if i < j:
                if polygons[i].intersects(polygons[j]):
                    inter = polygons[i].intersection(polygons[j])
                    if inter.area > 1e-12:
                        return True
    return False


def has_any_overlap_strict(trees: List[Tree]) -> bool:
    """Check if any trees overlap (strict segment intersection check)."""
    n = len(trees)
    if n <= 1:
        return False

    polygons = [t.get_polygon() for t in trees]
    vertices_list = [t.get_vertices() for t in trees]
    tree_idx = STRtree(polygons)

    for i, poly_i in enumerate(polygons):
        candidates = tree_idx.query(poly_i)
        for j in candidates:
            if i < j:
                if not polygons[i].intersects(polygons[j]):
                    continue
                if polygons_overlap_strict(vertices_list[i], vertices_list[j]):
                    return True
    return False


class SAOptimizer:
    """Simulated Annealing optimizer for full configuration."""

    def __init__(self, trees: List[Tree],
                 T_start: float = 0.1,
                 T_end: float = 0.0001,
                 cooling_rate: float = 0.999,
                 max_iters: int = 100000,
                 seed: int = None):
        self.trees = [Tree(t.x, t.y, t.angle) for t in trees]
        self.n = len(trees)
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_rate = cooling_rate
        self.max_iters = max_iters

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Best state tracking
        self.best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
        self.best_side = compute_side_length(trees)

        # Current state
        self.current_side = self.best_side
        self.T = T_start

        # Statistics
        self.accepted = 0
        self.rejected = 0
        self.improved = 0

    def make_move(self, scale: float) -> Tuple[int, float, float, float]:
        """Make a random move. Returns (tree_idx, old_x, old_y, old_angle)."""
        idx = random.randint(0, self.n - 1)
        tree = self.trees[idx]
        old_x, old_y, old_angle = tree.x, tree.y, tree.angle

        # Choose move type
        move_type = random.random()

        if move_type < 0.5:
            # Position perturbation
            dx = random.gauss(0, scale * 0.3)
            dy = random.gauss(0, scale * 0.3)
            tree.x += dx
            tree.y += dy
        elif move_type < 0.8:
            # Angle perturbation
            da = random.gauss(0, scale * 30)  # degrees
            tree.angle = (tree.angle + da) % 360
        else:
            # Combined position + angle
            dx = random.gauss(0, scale * 0.2)
            dy = random.gauss(0, scale * 0.2)
            da = random.gauss(0, scale * 20)
            tree.x += dx
            tree.y += dy
            tree.angle = (tree.angle + da) % 360

        return idx, old_x, old_y, old_angle

    def undo_move(self, idx: int, old_x: float, old_y: float, old_angle: float):
        """Undo a move."""
        self.trees[idx].x = old_x
        self.trees[idx].y = old_y
        self.trees[idx].angle = old_angle

    def is_valid(self) -> bool:
        """Fast validity check."""
        return not has_any_overlap_fast(self.trees)

    def step(self) -> bool:
        """Execute one SA step. Returns True if accepted."""
        # Scale based on temperature (larger moves when hot)
        scale = self.T / self.T_start

        # Make move
        idx, old_x, old_y, old_angle = self.make_move(scale)

        # Check validity
        if not self.is_valid():
            self.undo_move(idx, old_x, old_y, old_angle)
            self.rejected += 1
            return False

        # Compute new energy
        new_side = compute_side_length(self.trees)
        delta = new_side - self.current_side

        # Metropolis criterion
        if delta < 0:
            # Improvement - always accept
            self.current_side = new_side
            self.accepted += 1
            self.improved += 1

            # Update best if needed
            if new_side < self.best_side:
                self.best_side = new_side
                self.best_trees = [Tree(t.x, t.y, t.angle) for t in self.trees]
            return True
        else:
            # Worse - accept with probability exp(-delta/T)
            p = math.exp(-delta / self.T) if self.T > 0 else 0
            if random.random() < p:
                self.current_side = new_side
                self.accepted += 1
                return True
            else:
                self.undo_move(idx, old_x, old_y, old_angle)
                self.rejected += 1
                return False

    def run(self, verbose: bool = False) -> List[Tree]:
        """Run SA optimization."""
        start_time = time.time()
        iter_count = 0

        while self.T > self.T_end and iter_count < self.max_iters:
            self.step()
            self.T *= self.cooling_rate
            iter_count += 1

            if verbose and iter_count % 10000 == 0:
                elapsed = time.time() - start_time
                accept_rate = self.accepted / max(1, self.accepted + self.rejected)
                print(f"  Iter {iter_count}: T={self.T:.6f}, side={self.current_side:.6f}, "
                      f"best={self.best_side:.6f}, accept={accept_rate:.2%}")

        # Final strict validation
        if has_any_overlap_strict(self.best_trees):
            if verbose:
                print("  WARNING: Best solution has overlaps, reverting...")
            return None

        return self.best_trees


def aggressive_sa(trees: List[Tree], restarts: int = 3,
                  iters_per_restart: int = 50000,
                  verbose: bool = False) -> Optional[List[Tree]]:
    """
    Run multiple SA restarts with aggressive exploration.
    """
    original_side = compute_side_length(trees)
    best_trees = [Tree(t.x, t.y, t.angle) for t in trees]
    best_side = original_side

    for r in range(restarts):
        if verbose:
            print(f"  Restart {r+1}/{restarts}")

        # Different temperature schedules for different restarts
        T_start = 0.2 * (0.7 ** r)  # Decreasing T_start
        cooling = 0.9997 + 0.0001 * r  # Slower cooling later

        # Start from either original or best found
        if r == 0 or random.random() < 0.5:
            start = [Tree(t.x, t.y, t.angle) for t in trees]
        else:
            start = [Tree(t.x, t.y, t.angle) for t in best_trees]

        optimizer = SAOptimizer(
            start,
            T_start=T_start,
            T_end=0.0001,
            cooling_rate=cooling,
            max_iters=iters_per_restart,
            seed=42 + r
        )

        result = optimizer.run(verbose=verbose)

        if result is not None:
            side = compute_side_length(result)
            if side < best_side:
                best_side = side
                best_trees = result
                if verbose:
                    print(f"    New best: {side:.6f}")

    if best_side < original_side - 1e-10:
        return best_trees
    return None


def load_submission(csv_path: str) -> dict:
    """Load submission and return groups dictionary."""
    groups = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'][1:])  # Remove 's' prefix
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            groups[n].append(Tree(x, y, deg))

    return groups


def save_submission(groups: dict, csv_path: str):
    """Save groups to submission CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])

        for n in range(1, 201):
            for idx, tree in enumerate(groups[n]):
                row_id = f'{n:03d}_{idx}'
                writer.writerow([row_id, f's{tree.x}', f's{tree.y}', f's{tree.angle}'])


def main():
    parser = argparse.ArgumentParser(description='Gen120: Full-configuration SA')
    parser.add_argument('--input', default='submission_best.csv', help='Input submission CSV')
    parser.add_argument('--output', default='submission_gen120.csv', help='Output submission CSV')
    parser.add_argument('--n-start', type=int, default=2, help='Start n value')
    parser.add_argument('--n-end', type=int, default=50, help='End n value')
    parser.add_argument('--restarts', type=int, default=3, help='SA restarts per group')
    parser.add_argument('--iters', type=int, default=50000, help='Iterations per restart')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    groups = load_submission(args.input)

    # Compute original score
    original_score = sum(compute_side_length(groups[n])**2 / n for n in range(1, 201))
    print(f"Original score: {original_score:.4f}")

    total_improvement = 0
    improved_count = 0

    for n in range(args.n_start, args.n_end + 1):
        trees = groups[n]
        original_side = compute_side_length(trees)

        print(f"\nProcessing n={n} ({n} trees)...")

        # Scale iterations with n (more trees = more iterations needed)
        iters = args.iters * max(1, n // 10)

        result = aggressive_sa(
            trees,
            restarts=args.restarts,
            iters_per_restart=iters,
            verbose=args.verbose
        )

        if result is not None:
            new_side = compute_side_length(result)
            if new_side < original_side - 1e-10:
                groups[n] = result
                score_delta = (original_side**2 - new_side**2) / n
                total_improvement += score_delta
                improved_count += 1
                print(f"  IMPROVED: {original_side:.6f} -> {new_side:.6f} (Δscore={score_delta:.6f})")
            else:
                print(f"  No improvement (SA found {new_side:.6f} vs original {original_side:.6f})")
        else:
            print(f"  No valid improvement found")

    # Compute final score
    final_score = sum(compute_side_length(groups[n])**2 / n for n in range(1, 201))

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Groups improved: {improved_count}")
    print(f"  Score: {original_score:.4f} -> {final_score:.4f} (Δ={original_score - final_score:.4f})")

    if final_score < original_score - 1e-6:
        # Full validation
        print("\nValidating solution...")
        valid = True
        for n in range(1, 201):
            if has_any_overlap_strict(groups[n]):
                print(f"  ERROR: Overlaps in n={n}!")
                valid = False

        if valid:
            print("  All groups valid!")
            save_submission(groups, args.output)
            print(f"\nSaved improved solution to {args.output}")
        else:
            print("\nNot saving due to overlaps!")
    else:
        print("\nNo improvements found.")


if __name__ == '__main__':
    main()
