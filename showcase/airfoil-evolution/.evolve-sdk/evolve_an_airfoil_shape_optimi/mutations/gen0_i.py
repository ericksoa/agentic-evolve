#!/usr/bin/env python3
"""
Gen 0 Variant I: Hybrid Memetic Algorithm Initialization
Combines population-based evolutionary initialization with local search.
Optimized with vectorized population operations and parallel local search.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class VectorizedEvolution:
    """Vectorized evolutionary operations for CST coefficients."""

    def __init__(self, seed: int = 42, pop_size: int = 100):
        self.rng = np.random.RandomState(seed)
        self.pop_size = pop_size
        self.dimension = 10  # 5 upper + 5 lower CST coeffs

    def initialize_population_matrix(self) -> np.ndarray:
        """Initialize population as matrix for vectorized operations."""
        # Use different initialization strategies for diversity
        population = np.zeros((self.pop_size, self.dimension))

        # Strategy allocation
        strategies = [
            ('uniform', 0.3),
            ('normal', 0.3),
            ('known_good', 0.2),
            ('extreme', 0.2)
        ]

        start_idx = 0
        for strategy, fraction in strategies:
            end_idx = start_idx + int(fraction * self.pop_size)

            if strategy == 'uniform':
                # Uniform random in parameter bounds
                upper_part = self.rng.uniform(0.05, 0.25, (end_idx - start_idx, 5))
                lower_part = self.rng.uniform(-0.25, -0.05, (end_idx - start_idx, 5))
                population[start_idx:end_idx] = np.hstack([upper_part, lower_part])

            elif strategy == 'normal':
                # Normal distribution around good values
                upper_part = np.clip(self.rng.normal(0.15, 0.04, (end_idx - start_idx, 5)), 0.01, 0.3)
                lower_part = np.clip(self.rng.normal(-0.12, 0.03, (end_idx - start_idx, 5)), -0.3, -0.01)
                population[start_idx:end_idx] = np.hstack([upper_part, lower_part])

            elif strategy == 'known_good':
                # Perturbations around known good designs
                base_designs = [
                    [0.17, 0.16, 0.14, 0.12, 0.10, -0.17, -0.16, -0.14, -0.12, -0.10],  # Symmetric
                    [0.20, 0.18, 0.16, 0.13, 0.10, -0.14, -0.12, -0.10, -0.09, -0.08],  # Cambered
                    [0.13, 0.12, 0.11, 0.09, 0.07, -0.13, -0.12, -0.11, -0.09, -0.07],  # Thin
                ]

                for i in range(start_idx, end_idx):
                    base = self.rng.choice(len(base_designs))
                    noise = self.rng.normal(0, 0.02, self.dimension)
                    individual = np.array(base_designs[base]) + noise
                    # Clamp to bounds
                    individual[:5] = np.clip(individual[:5], 0.01, 0.3)
                    individual[5:] = np.clip(individual[5:], -0.3, -0.01)
                    population[i] = individual

            elif strategy == 'extreme':
                # Extreme cases for diversity
                for i in range(start_idx, end_idx):
                    if self.rng.random() < 0.33:
                        # Very thick
                        individual = np.array([0.22, 0.20, 0.18, 0.15, 0.12, -0.22, -0.20, -0.18, -0.15, -0.12])
                    elif self.rng.random() < 0.5:
                        # Very thin
                        individual = np.array([0.08, 0.07, 0.06, 0.05, 0.04, -0.08, -0.07, -0.06, -0.05, -0.04])
                    else:
                        # High camber
                        individual = np.array([0.20, 0.18, 0.15, 0.12, 0.10, -0.10, -0.08, -0.06, -0.05, -0.04])

                    noise = self.rng.normal(0, 0.01, self.dimension)
                    individual += noise
                    individual[:5] = np.clip(individual[:5], 0.01, 0.3)
                    individual[5:] = np.clip(individual[5:], -0.3, -0.01)
                    population[i] = individual

            start_idx = end_idx

        return population

    def vectorized_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                            crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized uniform crossover."""
        n_individuals = parent1.shape[0]
        crossover_mask = self.rng.random((n_individuals, self.dimension)) < crossover_rate

        child1 = np.where(crossover_mask, parent1, parent2)
        child2 = np.where(crossover_mask, parent2, parent1)

        return child1, child2

    def vectorized_mutation(self, population: np.ndarray, mutation_rate: float = 0.1,
                           mutation_strength: float = 0.02) -> np.ndarray:
        """Vectorized Gaussian mutation."""
        mutation_mask = self.rng.random(population.shape) < mutation_rate
        mutation_noise = self.rng.normal(0, mutation_strength, population.shape)

        mutated = population + mutation_mask * mutation_noise

        # Apply bounds
        mutated[:, :5] = np.clip(mutated[:, :5], 0.01, 0.3)  # Upper coeffs
        mutated[:, 5:] = np.clip(mutated[:, 5:], -0.3, -0.01)  # Lower coeffs

        return mutated

    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray,
                           tournament_size: int = 3) -> np.ndarray:
        """Vectorized tournament selection."""
        n_individuals = len(population)
        selected_indices = []

        for _ in range(n_individuals):
            # Random tournament
            tournament_indices = self.rng.choice(n_individuals, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)

        return population[selected_indices]

class LocalSearchOptimizer:
    """Fast local search for individual airfoil optimization."""

    def __init__(self):
        self.step_sizes = np.array([0.01] * 5 + [0.01] * 5)  # Adaptive step sizes

    def pattern_search(self, coeffs: np.ndarray, objective_func, max_iterations: int = 20) -> np.ndarray:
        """Simple pattern search local optimization."""
        current = coeffs.copy()
        current_fitness = objective_func(current)

        for iteration in range(max_iterations):
            improved = False

            # Try positive and negative steps for each dimension
            for dim in range(len(coeffs)):
                for direction in [1, -1]:
                    candidate = current.copy()
                    candidate[dim] += direction * self.step_sizes[dim]

                    # Apply bounds
                    if dim < 5:  # Upper coeffs
                        candidate[dim] = np.clip(candidate[dim], 0.01, 0.3)
                    else:  # Lower coeffs
                        candidate[dim] = np.clip(candidate[dim], -0.3, -0.01)

                    candidate_fitness = objective_func(candidate)

                    if candidate_fitness > current_fitness:
                        current = candidate
                        current_fitness = candidate_fitness
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                # Reduce step size
                self.step_sizes *= 0.8
                if np.max(self.step_sizes) < 1e-4:
                    break

        return current

class MemeticAlgorithmInitializer:
    """Memetic algorithm combining evolution with local search."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evolution = VectorizedEvolution(seed=seed, pop_size=60)
        self.local_search = LocalSearchOptimizer()
        self.eval_cache = {}

    def fitness_function(self, coeffs: np.ndarray) -> float:
        """Cached fitness evaluation."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.eval_cache:
            return self.eval_cache[coeffs_key]

        # Convert to airfoil and evaluate
        upper_coeffs = coeffs[:5].tolist()
        lower_coeffs = coeffs[5:].tolist()
        airfoil = Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

        req = DesignRequirements(reynolds=200000, objective="max_ld")
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)

        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        self.eval_cache[coeffs_key] = fitness
        return fitness

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate entire population."""
        fitness = np.array([self.fitness_function(individual) for individual in population])
        return fitness

    def run_memetic_algorithm(self, generations: int = 15) -> List[np.ndarray]:
        """Run memetic algorithm to generate optimized population."""
        print("Starting memetic algorithm initialization...")

        # Initialize population
        population = self.evolution.initialize_population_matrix()
        print(f"Initialized population: {population.shape}")

        # Evaluate initial population
        fitness = self.evaluate_population(population)
        print(f"Initial fitness - Best: {np.max(fitness):.4f}, Mean: {np.mean(fitness[fitness > 0]):.4f}")

        best_individuals = []

        for generation in range(generations):
            # Selection
            selected = self.evolution.tournament_selection(population, fitness)

            # Crossover (pairwise)
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1 = selected[i:i+1]
                    parent2 = selected[i+1:i+2]
                    child1, child2 = self.evolution.vectorized_crossover(parent1, parent2)
                    offspring.extend([child1[0], child2[0]])
                else:
                    offspring.append(selected[i])

            offspring = np.array(offspring)

            # Mutation
            offspring = self.evolution.vectorized_mutation(offspring)

            # Local search on best individuals (memetic component)
            offspring_fitness = self.evaluate_population(offspring)

            # Apply local search to top 10% of offspring
            n_elite = max(1, len(offspring) // 10)
            elite_indices = np.argsort(offspring_fitness)[-n_elite:]

            for idx in elite_indices:
                if offspring_fitness[idx] > 0:  # Only optimize valid designs
                    optimized = self.local_search.pattern_search(
                        offspring[idx], self.fitness_function, max_iterations=10
                    )
                    offspring[idx] = optimized
                    offspring_fitness[idx] = self.fitness_function(optimized)

            # Combine and select next generation
            combined_pop = np.vstack([population, offspring])
            combined_fitness = np.hstack([fitness, offspring_fitness])

            # Keep best individuals
            best_indices = np.argsort(combined_fitness)[-len(population):]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            # Store best individuals from each generation
            best_idx = np.argmax(fitness)
            best_individuals.append(population[best_idx].copy())

            print(f"Generation {generation + 1}: Best: {fitness[best_idx]:.4f}, Mean: {np.mean(fitness[fitness > 0]):.4f}")

        # Return all good individuals found
        final_population = []
        for individual in best_individuals:
            final_population.append(individual)

        # Add diverse population members
        diverse_indices = np.argsort(fitness)[-40:]  # Top 40 from final generation
        for idx in diverse_indices:
            if fitness[idx] > 0:
                final_population.append(population[idx])

        return final_population

def generate_population() -> List[Airfoil]:
    """Generate population using memetic algorithm."""
    memetic = MemeticAlgorithmInitializer(seed=252627)
    optimized_coeffs = memetic.run_memetic_algorithm()

    # Convert coefficient vectors back to airfoils
    population = []
    for coeffs in optimized_coeffs:
        upper_coeffs = coeffs[:5].tolist()
        lower_coeffs = coeffs[5:].tolist()
        airfoil = Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)
        population.append(airfoil)

    return population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate memetic population (with caching)."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    eval_cache = {}
    for airfoil in population:
        coeffs_key = tuple(airfoil.upper_coeffs + airfoil.lower_coeffs)

        if coeffs_key in eval_cache:
            fitness = eval_cache[coeffs_key]
        else:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            eval_cache[coeffs_key] = fitness

        results.append((airfoil, fitness))

    return results

def analyze_memetic_quality(population: List[Airfoil], evaluated: List[Tuple[Airfoil, float]]) -> Dict[str, float]:
    """Analyze quality of memetic algorithm results."""
    fitnesses = [f for _, f in evaluated]
    valid_fitnesses = [f for f in fitnesses if f > 0]

    if not valid_fitnesses:
        return {"error": "No valid airfoils generated"}

    # Evolution quality metrics
    quality_metrics = {
        'valid_fraction': len(valid_fitnesses) / len(fitnesses),
        'mean_fitness': np.mean(valid_fitnesses),
        'std_fitness': np.std(valid_fitnesses),
        'max_fitness': np.max(valid_fitnesses),
        'fitness_concentration': len([f for f in valid_fitnesses if f > np.mean(valid_fitnesses)]) / len(valid_fitnesses)
    }

    # Diversity after optimization
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    coeff_diversity = np.mean(np.std(all_coeffs, axis=0))

    quality_metrics.update({
        'coefficient_diversity': coeff_diversity,
        'memetic_effectiveness': quality_metrics['max_fitness'] / (quality_metrics['mean_fitness'] + 1e-6)
    })

    return quality_metrics

def main():
    """Main memetic algorithm initialization."""
    print("Generating memetic algorithm population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze quality
    memetic_quality = analyze_memetic_quality(population, evaluated)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:25]):
        filename = f"results/memetic_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['memetic_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "memetic_rank": i,
            "memetic_quality": memetic_quality
        })

    print(f"Generated {len(population)} memetic algorithm airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Memetic algorithm quality metrics:")
    for key, value in memetic_quality.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save analysis
    with open("results/memetic_analysis.json", 'w') as f:
        json.dump({
            "memetic_quality": memetic_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1] if evaluated else 0
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()