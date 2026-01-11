#!/usr/bin/env python3
"""
Gen 2 Variant B: Cache-Optimized Gradient-Free Bootstrap
Optimizes memory access patterns and caching for the Nelder-Mead bootstrap approach.
Adds vectorized coefficient processing and improved cache locality.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastNelderMead:
    """
    Cache-optimized Nelder-Mead optimizer for CST coefficients.
    Improved memory layout and vectorized operations.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 50, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Cache-optimized Nelder-Mead with vectorized operations."""
        n = len(x0)

        # Pre-allocate arrays for better cache locality
        simplex = np.zeros((n + 1, n), dtype=np.float64, order='C')
        f_values = np.zeros(n + 1, dtype=np.float64)
        temp_points = np.zeros((3, n), dtype=np.float64, order='C')  # Reusable workspace

        simplex[0] = x0

        # Vectorized simplex initialization
        step_sizes = np.maximum(0.05 * np.abs(x0), 0.01)
        identity_scaled = np.eye(n) * step_sizes[None, :]
        simplex[1:] = x0[None, :] + identity_scaled

        # Batch evaluate initial simplex
        f_values = np.array([objective_func(x) for x in simplex])

        for iteration in range(maxiter):
            # Sort by function values using argsort for cache efficiency
            indices = np.argsort(f_values)
            simplex[:] = simplex[indices]
            f_values[:] = f_values[indices]

            # Early convergence check
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Vectorized centroid calculation
            centroid = np.mean(simplex[:-1], axis=0, out=temp_points[0])

            # Vectorized reflection
            worst_point = simplex[-1]
            temp_points[1] = centroid + self.alpha * (centroid - worst_point)
            f_reflected = objective_func(temp_points[1])

            if f_values[0] <= f_reflected < f_values[-2]:
                # Accept reflection
                simplex[-1] = temp_points[1]
                f_values[-1] = f_reflected
                continue

            if f_reflected < f_values[0]:
                # Try expansion using pre-allocated array
                temp_points[2] = centroid + self.gamma * (temp_points[1] - centroid)
                f_expanded = objective_func(temp_points[2])

                if f_expanded < f_reflected:
                    simplex[-1] = temp_points[2]
                    f_values[-1] = f_expanded
                else:
                    simplex[-1] = temp_points[1]
                    f_values[-1] = f_reflected
                continue

            # Contraction with branch elimination
            use_reflected = f_reflected < f_values[-1]
            contract_point = centroid + self.rho * (
                (temp_points[1] if use_reflected else worst_point) - centroid
            )
            f_contracted = objective_func(contract_point)

            min_f = min(f_reflected, f_values[-1])
            if f_contracted < min_f:
                simplex[-1] = contract_point
                f_values[-1] = f_contracted
                continue

            # Vectorized shrink operation
            simplex[1:] = simplex[0] + self.sigma * (simplex[1:] - simplex[0])
            # Batch evaluate after shrink
            f_values[1:] = [objective_func(simplex[i]) for i in range(1, len(simplex))]

        # Return best point
        best_idx = np.argmin(f_values)
        return simplex[best_idx].copy(), f_values[best_idx]

class OptimizationBootstrapper:
    """Cache-optimized bootstrap with vectorized coefficient processing."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Pre-size cache with power-of-2 for better hash performance
        self.evaluator_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-allocate coefficient arrays for better memory locality
        self._coeff_buffer = np.zeros(10, dtype=np.float64)

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Vectorized coefficient processing with cache-friendly memory access."""
        # Use pre-allocated buffer to avoid repeated allocations
        np.copyto(self._coeff_buffer, coeffs)

        # Vectorized clipping operations
        upper_coeffs = np.clip(self._coeff_buffer[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(self._coeff_buffer[5:], -0.3, -0.01).tolist()

        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def objective_function(self, coeffs: np.ndarray) -> float:
        """Cache-optimized objective with faster key generation."""
        # Use faster hash-based cache key instead of tuple conversion
        coeffs_rounded = np.round(coeffs * 1000000).astype(np.int32)
        coeffs_key = hash(coeffs_rounded.tobytes())

        if coeffs_key in self.evaluator_cache:
            self._cache_hits += 1
            return self.evaluator_cache[coeffs_key]

        self._cache_misses += 1

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Vectorized penalty calculation
                max_t, _ = airfoil.max_thickness()
                penalties = np.array([
                    10 * max(0, 0.06 - max_t),  # Min thickness penalty
                    10 * max(0, max_t - 0.18)   # Max thickness penalty
                ])
                penalty = np.sum(penalties)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            # Limit cache size to prevent memory bloat
            if len(self.evaluator_cache) > 10000:
                # Remove oldest 20% of entries (simple FIFO)
                keys_to_remove = list(self.evaluator_cache.keys())[:2000]
                for key in keys_to_remove:
                    del self.evaluator_cache[key]

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def optimize_single_seed(self, seed_coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize with cache-friendly access patterns."""
        optimizer = FastNelderMead()

        try:
            result_coeffs, result_obj = optimizer.optimize(
                self.objective_function,
                seed_coeffs,
                maxiter=30,
                tolerance=1e-3
            )
            return result_coeffs, -result_obj
        except Exception:
            fitness = -self.objective_function(seed_coeffs)
            return seed_coeffs, fitness

    def generate_seed_population(self, n_seeds: int = 100) -> List[np.ndarray]:
        """Vectorized seed generation for better performance."""
        seeds = []

        # Pre-generate random numbers in batches for better cache performance
        strategies = self.rng.choice(['uniform', 'normal', 'beta', 'triangular'], size=n_seeds)

        for i in range(n_seeds):
            strategy = strategies[i]

            if strategy == 'uniform':
                coeffs = np.concatenate([
                    self.rng.uniform(0.05, 0.25, 5),
                    self.rng.uniform(-0.25, -0.05, 5)
                ])
            elif strategy == 'normal':
                upper = np.clip(self.rng.normal(0.15, 0.05, 5), 0.01, 0.3)
                lower = np.clip(self.rng.normal(-0.12, 0.04, 5), -0.3, -0.01)
                coeffs = np.concatenate([upper, lower])
            elif strategy == 'beta':
                beta_samples = self.rng.beta(2, 2, 10)
                upper = 0.01 + 0.24 * beta_samples[:5]
                lower = -0.25 + 0.24 * beta_samples[5:]
                coeffs = np.concatenate([upper, lower])
            else:  # triangular
                coeffs = np.concatenate([
                    self.rng.triangular(0.01, 0.15, 0.25, 5),
                    self.rng.triangular(-0.25, -0.12, -0.01, 5)
                ])

            seeds.append(coeffs)

        return seeds

def optimize_seed_wrapper(args):
    """Wrapper function for parallel optimization."""
    seed_coeffs, seed = args
    bootstrapper = OptimizationBootstrapper(seed=seed)
    return bootstrapper.optimize_single_seed(seed_coeffs)

def generate_population() -> List[Airfoil]:
    """Generate population using cache-optimized bootstrap."""
    print("Generating cache-optimized bootstrapped population...")

    bootstrapper = OptimizationBootstrapper(seed=192021)
    seeds = bootstrapper.generate_seed_population(60)

    # Prepare arguments for processing
    args_list = [(seed, 42 + i) for i, seed in enumerate(seeds)]

    optimized_results = []

    # Sequential processing with cache optimization
    for args in args_list[:50]:
        result = optimize_seed_wrapper(args)
        optimized_results.append(result)

    # Convert optimized coefficients back to airfoils
    population = []
    for coeffs, fitness in optimized_results:
        airfoil = bootstrapper.coeffs_to_airfoil(coeffs)
        population.append(airfoil)

    # Efficiently process remaining seeds
    remaining_seeds = seeds[50:]
    if remaining_seeds:
        # Batch process remaining seeds for cache efficiency
        seed_evaluations = []
        for seed in remaining_seeds:
            fitness = -bootstrapper.objective_function(seed)
            if fitness > 0:
                seed_evaluations.append((seed, fitness))

        # Add top seed designs
        seed_evaluations.sort(key=lambda x: x[1], reverse=True)
        for coeffs, fitness in seed_evaluations[:10]:
            airfoil = bootstrapper.coeffs_to_airfoil(coeffs)
            population.append(airfoil)

    print(f"Generated {len(population)} cache-optimized airfoils")
    print(f"Cache performance - Hits: {bootstrapper._cache_hits}, Misses: {bootstrapper._cache_misses}")
    return population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population with batch processing."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def analyze_optimization_quality(population: List[Airfoil], evaluated: List[Tuple[Airfoil, float]]) -> Dict[str, float]:
    """Analyze quality with vectorized operations."""
    fitnesses = np.array([f for _, f in evaluated if f > 0])

    if len(fitnesses) == 0:
        return {"error": "No valid airfoils generated"}

    # Vectorized fitness statistics
    fitness_stats = {
        'mean_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'max_fitness': np.max(fitnesses),
        'min_fitness': np.min(fitnesses),
        'valid_fraction': len(fitnesses) / len(population)
    }

    # Vectorized diversity calculation
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    coeff_ranges = np.ptp(all_coeffs, axis=0)  # ptp is faster than max-min

    fitness_stats.update({
        'coefficient_diversity': np.mean(coeff_ranges),
        'optimization_effectiveness': fitness_stats['max_fitness'] / (fitness_stats['mean_fitness'] + 1e-6)
    })

    return fitness_stats

def main():
    """Main cache-optimized bootstrap initialization."""
    print("Generating cache-optimized gradient-free bootstrapped population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze optimization quality
    opt_quality = analyze_optimization_quality(population, evaluated)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):
        filename = f"results/cache_optimized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['optimization_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "optimization_rank": i,
            "optimization_quality": opt_quality
        })

    print(f"Generated {len(population)} cache-optimized airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Optimization quality metrics:")
    for key, value in opt_quality.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save optimization analysis
    with open("results/cache_optimization_analysis.json", 'w') as f:
        json.dump({
            "optimization_quality": opt_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1] if evaluated else 0,
            "optimization_variant": "cache_optimized"
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()