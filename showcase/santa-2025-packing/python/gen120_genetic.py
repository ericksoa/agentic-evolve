#!/usr/bin/env python3
"""
Gen120: Genetic Algorithm for Tree Packing

Key difference from SA: Can combine good sub-solutions through crossover.

Chromosome: List of (x, y, angle) for n trees
Crossover: Exchange subsets of trees between parents
Mutation: Small position/angle perturbations
Selection: Tournament selection on bounding box size
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
import csv
import time

# Tree polygon
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
TREE_POLY = Polygon(TREE_VERTICES)


def get_tree_poly(x: float, y: float, angle: float) -> Polygon:
    rotated = affinity.rotate(TREE_POLY, angle, origin=(0, 0))
    return affinity.translate(rotated, x, y)


def compute_side(trees: List[Tuple[float, float, float]]) -> float:
    all_points = []
    for x, y, a in trees:
        poly = get_tree_poly(x, y, a)
        all_points.extend(poly.exterior.coords)
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_overlap(trees: List[Tuple[float, float, float]], tol: float = 1e-9) -> bool:
    n = len(trees)
    if n <= 1:
        return False

    polys = [get_tree_poly(x, y, a) for x, y, a in trees]
    tree_idx = STRtree(polys)

    for i, poly in enumerate(polys):
        candidates = tree_idx.query(poly)
        for j in candidates:
            if i < j:
                if polys[i].intersects(polys[j]):
                    inter = polys[i].intersection(polys[j])
                    if inter.area > tol:
                        return True
    return False


def fitness(trees: List[Tuple[float, float, float]]) -> float:
    """Fitness = negative side length (higher is better). Returns -inf if overlaps."""
    if has_overlap(trees):
        return float('-inf')
    return -compute_side(trees)


def random_individual(n: int, scale: float = 3.0) -> List[Tuple[float, float, float]]:
    """Generate a random individual."""
    trees = []
    for _ in range(n):
        x = random.uniform(-scale, scale)
        y = random.uniform(-scale, scale)
        a = random.uniform(0, 360)
        trees.append((x, y, a))
    return trees


def greedy_individual(n: int) -> List[Tuple[float, float, float]]:
    """Generate individual using greedy placement."""
    trees = []

    # First tree at origin
    trees.append((0, 0, random.uniform(0, 360)))

    for i in range(1, n):
        best_pos = None
        best_side = float('inf')

        # Try random positions and keep best valid one
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0.5, 3.0) * math.sqrt(i)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            a = random.uniform(0, 360)

            test = trees + [(x, y, a)]
            if not has_overlap(test):
                side = compute_side(test)
                if side < best_side:
                    best_side = side
                    best_pos = (x, y, a)

        if best_pos is None:
            # Fall back to random far position
            angle = random.uniform(0, 2 * math.pi)
            radius = 5 * math.sqrt(i)
            best_pos = (radius * math.cos(angle), radius * math.sin(angle), random.uniform(0, 360))

        trees.append(best_pos)

    return trees


def crossover(p1: List, p2: List) -> List:
    """Single-point crossover of tree positions."""
    n = len(p1)
    if n <= 2:
        return p1 if random.random() < 0.5 else p2

    # Single point crossover
    pt = random.randint(1, n - 1)
    child = p1[:pt] + p2[pt:]
    return child


def mutate(ind: List, mutation_rate: float = 0.1, scale: float = 0.3) -> List:
    """Mutate by perturbing random trees."""
    result = []
    for x, y, a in ind:
        if random.random() < mutation_rate:
            x += random.gauss(0, scale)
            y += random.gauss(0, scale)
            a = (a + random.gauss(0, 30)) % 360
        result.append((x, y, a))
    return result


def repair(ind: List, max_attempts: int = 100) -> Optional[List]:
    """Try to repair overlapping individual by shifting trees."""
    if not has_overlap(ind):
        return ind

    ind = list(ind)
    n = len(ind)

    for _ in range(max_attempts):
        # Find overlapping pair
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                p1 = get_tree_poly(*ind[i])
                p2 = get_tree_poly(*ind[j])
                if p1.intersects(p2) and p1.intersection(p2).area > 1e-9:
                    # Move tree j away from i
                    x1, y1, _ = ind[i]
                    x2, y2, a2 = ind[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    dx, dy = dx/dist, dy/dist

                    # Push j away
                    ind[j] = (x2 + dx * 0.1, y2 + dy * 0.1, a2)
                    overlap_found = True
                    break
            if overlap_found:
                break

        if not overlap_found or not has_overlap(ind):
            break

    return ind if not has_overlap(ind) else None


def tournament_select(pop: List, fitnesses: List[float], k: int = 3) -> List:
    """Tournament selection."""
    candidates = random.sample(range(len(pop)), k)
    best = max(candidates, key=lambda i: fitnesses[i])
    return pop[best]


def ga_optimize(n: int, pop_size: int = 50, generations: int = 100,
                mutation_rate: float = 0.2, elite_count: int = 5,
                seed_with: List = None, verbose: bool = False) -> Tuple[float, List]:
    """Run genetic algorithm."""

    # Initialize population
    population = []

    # Seed with provided solution
    if seed_with is not None:
        population.append(list(seed_with))

    # Add greedy individuals
    for _ in range(pop_size // 3):
        ind = greedy_individual(n)
        if not has_overlap(ind):
            population.append(ind)

    # Fill rest with random
    while len(population) < pop_size:
        ind = random_individual(n, scale=math.sqrt(n))
        if not has_overlap(ind):
            population.append(ind)

    # Compute initial fitnesses
    fitnesses = [fitness(ind) for ind in population]

    # Track best
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_ind = population[best_idx]
    best_fitness = fitnesses[best_idx]

    for gen in range(generations):
        # Create new population
        new_pop = []

        # Elitism: keep best individuals
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        for i in sorted_indices[:elite_count]:
            new_pop.append(population[i])

        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Select parents
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)

            # Crossover
            if random.random() < 0.7:
                child = crossover(p1, p2)
            else:
                child = p1 if random.random() < 0.5 else p2

            # Mutate
            child = mutate(child, mutation_rate=mutation_rate)

            # Repair if needed
            child = repair(child)
            if child is not None:
                new_pop.append(child)

        population = new_pop[:pop_size]
        fitnesses = [fitness(ind) for ind in population]

        # Update best
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_ind = population[gen_best_idx]
            if verbose:
                side = -best_fitness
                print(f"  Gen {gen}: new best side = {side:.6f}")

        if verbose and gen % 20 == 0:
            side = -fitnesses[gen_best_idx]
            print(f"  Gen {gen}: current best = {side:.6f}")

    return -best_fitness, best_ind


def load_submission(csv_path: str) -> dict:
    groups = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'][1:])
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            groups[n].append((x, y, deg))
    return groups


def save_submission(groups: dict, csv_path: str):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])
        for n in range(1, 201):
            for idx, (x, y, a) in enumerate(groups[n]):
                writer.writerow([f'{n:03d}_{idx}', f's{x}', f's{y}', f's{a}'])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=list(range(2, 21)))
    parser.add_argument('--input', default='submission_best.csv')
    parser.add_argument('--output', default='submission_ga.csv')
    parser.add_argument('--pop-size', type=int, default=50)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("Gen120: Genetic Algorithm for Tree Packing")
    print("=" * 50)

    groups = load_submission(args.input)
    original_score = sum(compute_side(groups[n])**2 / n for n in range(1, 201))
    print(f"Original score: {original_score:.4f}")

    improvements = []

    for n in args.n:
        current = groups[n]
        current_side = compute_side(current)

        print(f"\nn={n}: current side = {current_side:.6f}")

        # Run GA
        ga_side, ga_trees = ga_optimize(
            n,
            pop_size=args.pop_size,
            generations=args.generations,
            seed_with=current,
            verbose=args.verbose
        )

        if ga_side < current_side - 1e-6:
            score_delta = (current_side**2 - ga_side**2) / n
            improvements.append((n, current_side - ga_side, score_delta))
            groups[n] = ga_trees
            print(f"  IMPROVED: {current_side:.6f} -> {ga_side:.6f} (Δscore={score_delta:.6f})")
        else:
            print(f"  No improvement (GA best: {ga_side:.6f})")

    # Summary
    final_score = sum(compute_side(groups[n])**2 / n for n in range(1, 201))
    print(f"\n{'='*50}")
    print(f"Original: {original_score:.4f}")
    print(f"Final: {final_score:.4f}")
    print(f"Improvement: {original_score - final_score:.4f}")

    if improvements:
        print("\nImproved groups:")
        for n, imp, sd in improvements:
            print(f"  n={n}: side -{imp:.6f}, score -{sd:.6f}")

        save_submission(groups, args.output)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
