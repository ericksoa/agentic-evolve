#!/usr/bin/env python3
"""
Gen 3 Variant B: Cache-Optimized Gradient-Free Bootstrap
Enhanced version with advanced caching, memoization, and lookup tables for improved performance.
Uses intelligent cache warming and coefficient discretization for maximum cache efficiency.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastNelderMead:
    """
    Simplified Nelder-Mead optimizer for CST coefficients.
    Optimized for speed with reduced function evaluations.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 50, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Fast Nelder-Mead optimization with early termination."""
        n = len(x0)

        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Create initial simplex with adaptive step size
        step_size = 0.05 * np.abs(x0)
        step_size = np.where(step_size < 0.01, 0.01, step_size)

        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += step_size[i]

        # Evaluate initial simplex
        f_values = np.array([objective_func(x) for x in simplex])

        for iteration in range(maxiter):
            # Sort simplex by function values
            indices = np.argsort(f_values)
            simplex = simplex[indices]
            f_values = f_values[indices]

            # Check convergence
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Centroid of n best points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + self.alpha * (centroid - simplex[-1])
            f_reflected = objective_func(reflected)

            if f_values[0] <= f_reflected < f_values[-2]:
                # Accept reflection
                simplex[-1] = reflected
                f_values[-1] = f_reflected
                continue

            if f_reflected < f_values[0]:
                # Try expansion
                expanded = centroid + self.gamma * (reflected - centroid)
                f_expanded = objective_func(expanded)

                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    f_values[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    f_values[-1] = f_reflected
                continue

            # Contraction
            if f_reflected < f_values[-1]:
                contracted = centroid + self.rho * (reflected - centroid)
            else:
                contracted = centroid + self.rho * (simplex[-1] - centroid)

            f_contracted = objective_func(contracted)

            if f_contracted < min(f_reflected, f_values[-1]):
                simplex[-1] = contracted
                f_values[-1] = f_contracted
                continue

            # Shrink
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                f_values[i] = objective_func(simplex[i])

        # Return best point
        best_idx = np.argmin(f_values)
        return simplex[best_idx], f_values[best_idx]

class AdvancedCacheManager:
    """Advanced multi-level cache with lookup tables and intelligent prefetching."""

    def __init__(self, discretization_precision: float = 0.001):
        # Multi-level caching
        self.evaluation_cache = {}  # Full evaluation results
        self.fitness_lookup = {}    # Fast fitness lookup table
        self.thickness_cache = {}   # Precomputed thickness values
        self.discretization = discretization_precision

        # Cache statistics for optimization
        self.cache_hits = 0
        self.cache_misses = 0

        # Precompute common thickness penalty lookup table
        self._build_thickness_penalty_lookup()

    def _build_thickness_penalty_lookup(self):
        """Precompute thickness penalty lookup table."""
        self.thickness_penalties = {}

        # Discretized thickness values from 0.01 to 0.30
        thickness_range = np.arange(0.01, 0.31, 0.001)

        for t in thickness_range:
            penalty = 0
            if t < 0.06:
                penalty += 10 * (0.06 - t)
            if t > 0.18:
                penalty += 10 * (t - 0.18)

            # Store with discretized key
            key = round(t / self.discretization) * self.discretization
            self.thickness_penalties[key] = penalty

    def get_thickness_penalty_fast(self, thickness: float) -> float:
        """Fast thickness penalty lookup with interpolation."""
        # Discretize thickness
        discretized = round(thickness / self.discretization) * self.discretization

        # Direct lookup if available
        if discretized in self.thickness_penalties:
            return self.thickness_penalties[discretized]

        # Fallback calculation
        penalty = 0
        if thickness < 0.06:
            penalty += 10 * (0.06 - thickness)
        if thickness > 0.18:
            penalty += 10 * (thickness - 0.18)

        return penalty

    def discretize_coeffs(self, coeffs: np.ndarray) -> tuple:
        """Discretize coefficients for consistent caching."""
        discretized = np.round(coeffs / self.discretization) * self.discretization
        return tuple(discretized)

    def get_cache_key(self, coeffs: np.ndarray) -> str:
        """Generate consistent cache key with discretization."""
        discretized = self.discretize_coeffs(coeffs)
        return str(hash(discretized))

    def warm_cache_with_neighborhood(self, base_coeffs: np.ndarray, radius: float = 0.02):
        """Warm cache by precomputing evaluations in coefficient neighborhood."""
        # Generate neighboring points for cache warming
        n_coeffs = len(base_coeffs)
        variations = []

        # Generate systematic variations around base point
        for i in range(n_coeffs):
            for delta in [-radius, radius]:
                var_coeffs = base_coeffs.copy()
                var_coeffs[i] += delta
                variations.append(var_coeffs)

        return variations

class OptimizationBootstrapper:
    """Bootstrap airfoil population using gradient-free optimization with advanced caching."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.cache_manager = AdvancedCacheManager()

        # Batch processing for cache efficiency
        self.evaluation_batch = []
        self.batch_size = 10

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Convert coefficient vector to Airfoil object."""
        # Ensure coefficients are in valid ranges
        upper_coeffs = np.clip(coeffs[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(coeffs[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def objective_function(self, coeffs: np.ndarray) -> float:
        """Objective function with multi-level caching and lookup tables."""
        # Use advanced cache key
        cache_key = self.cache_manager.get_cache_key(coeffs)

        if cache_key in self.cache_manager.evaluation_cache:
            self.cache_manager.cache_hits += 1
            return self.cache_manager.evaluation_cache[cache_key]

        self.cache_manager.cache_misses += 1

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']

                # Fast thickness penalty using lookup table
                max_t, _ = airfoil.max_thickness()
                penalty = self.cache_manager.get_thickness_penalty_fast(max_t)

                objective_value = -fitness + penalty  # Minimize negative fitness
            else:
                objective_value = 1000  # Large penalty for invalid designs

            # Cache the result with multiple keys for future hits
            self.cache_manager.evaluation_cache[cache_key] = objective_value

            # Also cache in fitness lookup for quick access
            discretized = self.cache_manager.discretize_coeffs(coeffs)
            self.cache_manager.fitness_lookup[discretized] = -objective_value + penalty

            return objective_value

        except Exception:
            penalty_value = 1000
            self.cache_manager.evaluation_cache[cache_key] = penalty_value
            return penalty_value

    def objective_function_batch(self, coeffs_batch: List[np.ndarray]) -> List[float]:
        """Batch evaluation for improved cache locality."""
        results = []
        uncached_coeffs = []
        uncached_indices = []

        # Separate cached and uncached evaluations
        for i, coeffs in enumerate(coeffs_batch):
            cache_key = self.cache_manager.get_cache_key(coeffs)
            if cache_key in self.cache_manager.evaluation_cache:
                results.append(self.cache_manager.evaluation_cache[cache_key])
                self.cache_manager.cache_hits += 1
            else:
                results.append(None)  # Placeholder
                uncached_coeffs.append(coeffs)
                uncached_indices.append(i)

        # Batch process uncached evaluations
        for idx, coeffs in zip(uncached_indices, uncached_coeffs):
            result = self.objective_function(coeffs)
            results[idx] = result

        return results

    def optimize_single_seed(self, seed_coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize single random seed using Nelder-Mead with cache warming."""
        # Warm cache around starting point
        warm_points = self.cache_manager.warm_cache_with_neighborhood(seed_coeffs, radius=0.01)
        self.objective_function_batch(warm_points[:5])  # Warm a few key points

        optimizer = FastNelderMead()

        try:
            result_coeffs, result_obj = optimizer.optimize(
                self.objective_function,
                seed_coeffs,
                maxiter=30,  # Reduced iterations for speed
                tolerance=1e-3
            )
            return result_coeffs, -result_obj  # Convert back to fitness (positive)
        except Exception:
            # Fallback: return original seed with evaluation
            fitness = -self.objective_function(seed_coeffs)
            return seed_coeffs, fitness

    def generate_seed_population(self, n_seeds: int = 100) -> List[np.ndarray]:
        """Generate diverse random seeds for optimization."""
        seeds = []

        for _ in range(n_seeds):
            # Random coefficients with different strategies
            strategy = self.rng.choice(['uniform', 'normal', 'beta', 'triangular'])

            if strategy == 'uniform':
                upper = self.rng.uniform(0.05, 0.25, 5)
                lower = self.rng.uniform(-0.25, -0.05, 5)
            elif strategy == 'normal':
                upper = np.clip(self.rng.normal(0.15, 0.05, 5), 0.01, 0.3)
                lower = np.clip(self.rng.normal(-0.12, 0.04, 5), -0.3, -0.01)
            elif strategy == 'beta':
                # Beta distribution for more variety
                upper = 0.01 + 0.24 * self.rng.beta(2, 2, 5)
                lower = -0.25 + 0.24 * self.rng.beta(2, 2, 5)
            else:  # triangular
                upper = self.rng.triangular(0.01, 0.15, 0.25, 5)
                lower = self.rng.triangular(-0.25, -0.12, -0.01, 5)

            coeffs = np.concatenate([upper, lower])
            seeds.append(coeffs)

        return seeds

def optimize_seed_wrapper(args):
    """Wrapper function for parallel optimization."""
    seed_coeffs, seed = args
    bootstrapper = OptimizationBootstrapper(seed=seed)
    return bootstrapper.optimize_single_seed(seed_coeffs)

def generate_population() -> List[Airfoil]:
    """Generate population using parallel optimization bootstrap."""
    print("Generating cache-optimized optimization-bootstrapped population...")

    # Generate seeds
    bootstrapper = OptimizationBootstrapper(seed=192021)
    seeds = bootstrapper.generate_seed_population(60)

    # Prepare arguments for parallel processing
    args_list = [(seed, 42 + i) for i, seed in enumerate(seeds)]

    # Optimize seeds in parallel (simulated - actual parallel processing needs careful setup)
    optimized_results = []

    # For now, use sequential processing to avoid import issues
    for args in args_list[:50]:  # Limit to 50 for reasonable runtime
        result = optimize_seed_wrapper(args)
        optimized_results.append(result)

    # Print cache performance stats
    print(f"Cache performance: {bootstrapper.cache_manager.cache_hits} hits, {bootstrapper.cache_manager.cache_misses} misses")
    hit_rate = bootstrapper.cache_manager.cache_hits / (bootstrapper.cache_manager.cache_hits + bootstrapper.cache_manager.cache_misses) if bootstrapper.cache_manager.cache_misses > 0 else 1.0
    print(f"Cache hit rate: {hit_rate:.3f}")

    # Convert optimized coefficients back to airfoils
    population = []
    for coeffs, fitness in optimized_results:
        airfoil = bootstrapper.coeffs_to_airfoil(coeffs)
        population.append(airfoil)

    # Add some diversity by including original seeds that were highly fit
    seed_evaluations = []
    for seed in seeds[50:]:  # Remaining seeds
        fitness = -bootstrapper.objective_function(seed)
        if fitness > 0:  # Valid design
            seed_evaluations.append((seed, fitness))

    # Add top seed designs
    seed_evaluations.sort(key=lambda x: x[1], reverse=True)
    for coeffs, fitness in seed_evaluations[:10]:
        airfoil = bootstrapper.coeffs_to_airfoil(coeffs)
        population.append(airfoil)

    print(f"Generated {len(population)} cache-optimized airfoils")
    return population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate optimized population."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def analyze_optimization_quality(population: List[Airfoil], evaluated: List[Tuple[Airfoil, float]]) -> Dict[str, float]:
    """Analyze quality of optimization-based initialization."""
    fitnesses = [f for _, f in evaluated if f > 0]

    if not fitnesses:
        return {"error": "No valid airfoils generated"}

    # Convergence metrics
    fitness_stats = {
        'mean_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'max_fitness': np.max(fitnesses),
        'min_fitness': np.min(fitnesses),
        'valid_fraction': len(fitnesses) / len(population)
    }

    # Design diversity after optimization
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    coeff_ranges = np.max(all_coeffs, axis=0) - np.min(all_coeffs, axis=0)

    fitness_stats.update({
        'coefficient_diversity': np.mean(coeff_ranges),
        'optimization_effectiveness': fitness_stats['max_fitness'] / (fitness_stats['mean_fitness'] + 1e-6)
    })

    return fitness_stats

def main():
    """Main optimization bootstrap initialization."""
    print("Generating cache-optimized gradient-free optimization bootstrapped population...")

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
        filename = f"results/cached_optimized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['optimization_rank'] = i
            data['fitness'] = fitness
            data['cache_optimized'] = True
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
    with open("results/cached_optimization_analysis.json", 'w') as f:
        json.dump({
            "optimization_quality": opt_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1] if evaluated else 0,
            "cache_optimized": True
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()