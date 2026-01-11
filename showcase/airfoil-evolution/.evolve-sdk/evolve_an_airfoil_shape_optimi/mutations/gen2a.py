#!/usr/bin/env python3
"""
Gen 2 Variant A: Vectorized Uniform Sampling with Cache-Optimized Evaluation
Replaces Sobol sequences with highly vectorized uniform sampling for better performance.
Uses batch matrix operations and optimized memory layouts.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class VectorizedUniformInitializer:
    """
    High-performance airfoil initialization using vectorized uniform sampling.
    Optimized for cache efficiency and SIMD operations.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Pre-allocate arrays for better cache performance
        self.upper_bounds = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        self.upper_ranges = np.array([0.25, 0.21, 0.17, 0.13, 0.09])  # max - min
        self.lower_bounds = np.array([-0.25, -0.2, -0.15, -0.12, -0.1])
        self.lower_ranges = np.array([0.21, 0.17, 0.13, 0.11, 0.095])  # max - min

    def generate_population_vectorized(self, size: int = 50) -> List[Airfoil]:
        """Generate population using fully vectorized operations."""
        # Generate all random numbers at once for better cache locality
        random_matrix = self.rng.random((size, 10))

        # Vectorized coefficient generation using broadcasting
        upper_coeffs_matrix = self.upper_bounds[np.newaxis, :] + random_matrix[:, :5] * self.upper_ranges[np.newaxis, :]
        lower_coeffs_matrix = self.lower_bounds[np.newaxis, :] + random_matrix[:, 5:] * self.lower_ranges[np.newaxis, :]

        # Create airfoils using list comprehension for speed
        population = [
            Airfoil(upper_coeffs=upper_coeffs_matrix[i].tolist(),
                   lower_coeffs=lower_coeffs_matrix[i].tolist(),
                   zte=0.0)
            for i in range(size)
        ]

        return population

class FastAdaptiveRefinement:
    """Fast adaptive refinement using vectorized operations."""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def refine_around_elite(self, elite_airfoils: List[Airfoil],
                           refinement_size: int = 20) -> List[Airfoil]:
        """Vectorized refinement around elite airfoils."""
        refined = []

        # Process all elite airfoils at once
        n_elite = min(len(elite_airfoils), 5)
        samples_per_elite = refinement_size // n_elite

        for airfoil in elite_airfoils[:n_elite]:
            # Vectorized perturbation generation
            perturbations = self.rng.normal(0, 0.05, (samples_per_elite, 10))

            # Current coefficients as arrays
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate all perturbations at once
            new_upper_matrix = center_upper[np.newaxis, :] + perturbations[:, :5]
            new_lower_matrix = center_lower[np.newaxis, :] + perturbations[:, 5:]

            # Vectorized clamping
            new_upper_matrix = np.clip(new_upper_matrix, 0.01, 0.3)
            new_lower_matrix = np.clip(new_lower_matrix, -0.3, -0.005)

            # Create refined airfoils
            for i in range(samples_per_elite):
                refined.append(Airfoil(
                    upper_coeffs=new_upper_matrix[i].tolist(),
                    lower_coeffs=new_lower_matrix[i].tolist(),
                    zte=0.0
                ))

        return refined

def generate_population() -> List[Airfoil]:
    """Generate population using vectorized uniform sampling."""
    initializer = VectorizedUniformInitializer(seed=789)

    # Generate initial population with vectorized operations
    base_population = initializer.generate_population_vectorized(40)

    # Fast initial evaluation using batch processing
    req = DesignRequirements(reynolds=200000, objective="max_ld")

    # Pre-allocate arrays for fitness tracking
    fitness_array = np.zeros(len(base_population))

    for i, airfoil in enumerate(base_population):
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness_array[i] = result.get('fitness', 0.0) if result.get('valid', False) else 0.0

    # Use NumPy argsort for fast sorting
    sorted_indices = np.argsort(fitness_array)[::-1]  # Descending order

    # Select elite airfoils using array indexing
    elite_indices = sorted_indices[:10]
    elite_airfoils = [base_population[i] for i in elite_indices if fitness_array[i] > 0]

    # Fast adaptive refinement
    refiner = FastAdaptiveRefinement(initializer.rng)
    refined = refiner.refine_around_elite(elite_airfoils, 20)

    # Combine populations
    all_population = base_population + refined

    return all_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population with optimized batch processing."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")

    # Pre-allocate results array
    n_pop = len(population)
    fitness_results = np.zeros(n_pop)

    # Optimized evaluation loop
    for i, airfoil in enumerate(population):
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness_results[i] = result.get('fitness', 0.0) if result.get('valid', False) else 0.0

    # Create results using zip for efficiency
    results = list(zip(population, fitness_results.tolist()))

    return results

def main():
    """Main vectorized uniform sampling optimization."""
    print("Generating vectorized uniform airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # NumPy-based sorting for performance
    fitness_array = np.array([fitness for _, fitness in evaluated])
    sorted_indices = np.argsort(fitness_array)[::-1]

    # Create sorted results
    evaluated_sorted = [evaluated[i] for i in sorted_indices]

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated_sorted[:20]):
        filename = f"results/vectorized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            json.dump(airfoil.to_dict(), f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "rank": i,
            "coeffs_variance": (
                np.var(airfoil.upper_coeffs),
                np.var(airfoil.lower_coeffs)
            )
        })

    print(f"Generated {len(population)} vectorized airfoils")
    print(f"Best fitness: {evaluated_sorted[0][1]:.4f}")
    print(f"Population diversity: {np.std(fitness_array):.3f}")

    return best_results

if __name__ == "__main__":
    main()