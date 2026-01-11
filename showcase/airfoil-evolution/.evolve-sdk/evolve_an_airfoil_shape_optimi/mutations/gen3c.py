#!/usr/bin/env python3
"""
Gen 3 Variant C: Cache-Optimized CST Population Initializer
Mutation: Cache optimization with pre-computed lookup tables and memory-efficient operations.
Reduces memory allocations and improves cache locality through structured data access.
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

class CacheOptimizedCSTInitializer:
    """Cache-optimized CST coefficient initialization with lookup tables."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Pre-compute lookup tables for cache efficiency
        self._init_lookup_tables()

        # Memory-aligned arrays for better cache performance
        self.upper_bounds = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)
        self.lower_bounds = np.array([-0.25, -0.2, -0.15, -0.12, -0.1], dtype=np.float32)

    def _init_lookup_tables(self):
        """Pre-compute coefficient distributions for cache optimization."""
        # Pre-computed Gaussian samples for faster access
        n_samples = 1000
        self.gaussian_cache_upper = self.rng.normal(0.15, 0.05, n_samples).astype(np.float32)
        self.gaussian_cache_lower = self.rng.normal(-0.12, 0.04, n_samples).astype(np.float32)
        self.cache_index = 0

        # Pre-computed scaling factors
        self.thick_scale = 1.3
        self.camber_upper_scale = 1.2
        self.camber_lower_scale = 0.8

    def _get_cached_gaussian(self, size: int, cache_type: str) -> np.ndarray:
        """Get Gaussian samples from cache for better memory locality."""
        if cache_type == 'upper':
            cache = self.gaussian_cache_upper
        else:
            cache = self.gaussian_cache_lower

        # Wrap around cache if needed
        if self.cache_index + size > len(cache):
            self.cache_index = 0

        samples = cache[self.cache_index:self.cache_index + size]
        self.cache_index += size

        return samples

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate population using cache-optimized vectorized operations."""
        n_coeffs = 5

        # Use pre-allocated arrays for better cache performance
        all_upper = np.zeros((size, n_coeffs), dtype=np.float32)
        all_lower = np.zeros((size, n_coeffs), dtype=np.float32)

        # Fill arrays using cached Gaussian samples
        for i in range(n_coeffs):
            all_upper[:, i] = self._get_cached_gaussian(size, 'upper')
            all_lower[:, i] = self._get_cached_gaussian(size, 'lower')

        # In-place clipping to reduce memory allocations
        np.clip(all_upper, 0.05, self.upper_bounds[np.newaxis, :], out=all_upper)
        np.clip(all_lower, self.lower_bounds[np.newaxis, :], -0.05, out=all_lower)

        # Create airfoils with optimized memory layout
        population = []
        for i in range(size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        return population

def generate_diverse_population() -> List[Airfoil]:
    """Generate diverse initial population with cache-optimized strategies."""
    all_airfoils = []

    # Strategy 1: Standard distribution (20 airfoils) - cache-optimized
    init1 = CacheOptimizedCSTInitializer(seed=42)
    all_airfoils.extend(init1.generate_population(20))

    # Strategy 2: High-camber biased (10 airfoils) - in-place scaling
    init2 = CacheOptimizedCSTInitializer(seed=123)
    # Use in-place multiplication for better cache performance
    init2.upper_bounds *= init2.camber_upper_scale
    init2.lower_bounds *= init2.camber_lower_scale
    all_airfoils.extend(init2.generate_population(10))

    # Strategy 3: Thick airfoils (10 airfoils) - vectorized scaling
    init3 = CacheOptimizedCSTInitializer(seed=456)
    thick_pop = init3.generate_population(10)

    # Vectorized in-place coefficient scaling
    for airfoil in thick_pop:
        upper_array = np.array(airfoil.upper_coeffs, dtype=np.float32)
        lower_array = np.array(airfoil.lower_coeffs, dtype=np.float32)

        upper_array *= init3.thick_scale
        lower_array *= init3.thick_scale

        airfoil.upper_coeffs = upper_array.tolist()
        airfoil.lower_coeffs = lower_array.tolist()

    all_airfoils.extend(thick_pop)

    # Strategy 4: Symmetric airfoils (10 airfoils) - vectorized negation
    init4 = CacheOptimizedCSTInitializer(seed=789)
    sym_pop = init4.generate_population(10)

    # Vectorized symmetry operation
    for airfoil in sym_pop:
        upper_array = np.array(airfoil.upper_coeffs, dtype=np.float32)
        airfoil.lower_coeffs = (-upper_array).tolist()

    all_airfoils.extend(sym_pop)

    return all_airfoils

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population fitness using batch-optimized evaluation."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Pre-allocate results array for better memory locality
    fitness_cache = np.zeros(len(population), dtype=np.float32)

    # Batch evaluation with cache-friendly access pattern
    for i, airfoil in enumerate(population):
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness_cache[i] = result.get('fitness', 0.0) if result.get('valid', False) else 0.0

    # Create results using cached fitness values
    results = [(population[i], float(fitness_cache[i])) for i in range(len(population))]

    return results

def main():
    """Main population initialization with cache optimization."""
    print("Generating cache-optimized diverse CST airfoil population...")

    # Generate population
    population = generate_diverse_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness (in-place for memory efficiency)
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save best airfoils
    os.makedirs("results", exist_ok=True)

    best_airfoils = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):  # Top 20
        filename = f"results/airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            json.dump(airfoil.to_dict(), f, indent=2)

        best_airfoils.append({
            "filename": filename,
            "fitness": fitness,
            "upper_coeffs": airfoil.upper_coeffs,
            "lower_coeffs": airfoil.lower_coeffs
        })

    # Print results
    print(f"Generated {len(population)} airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Mean fitness: {np.mean([f for _, f in evaluated]):.4f}")
    print(f"Valid designs: {sum(1 for _, f in evaluated if f > 0)}")

    return best_airfoils

if __name__ == "__main__":
    main()