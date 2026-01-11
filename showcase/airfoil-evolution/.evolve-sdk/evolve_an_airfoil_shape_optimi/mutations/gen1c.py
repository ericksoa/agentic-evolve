#!/usr/bin/env python3
"""
Gen 1 Variant C: Latin Hypercube Sampling CST Population Initializer
Mutation: Replaced Gaussian distributions with LHS for better space-filling sampling.
Uses Latin Hypercube Sampling for more uniform exploration of the parameter space.
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

class LHSCSTInitializer:
    """Latin Hypercube Sampling-based CST coefficient initialization for optimal space coverage."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Pre-computed coefficient ranges for performance
        self.upper_bounds = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        self.lower_bounds = np.array([-0.25, -0.2, -0.15, -0.12, -0.1])
        self.upper_means = np.array([0.15, 0.12, 0.10, 0.08, 0.05])
        self.lower_means = np.array([-0.12, -0.10, -0.08, -0.06, -0.04])

    def _lhs_sample(self, n_samples: int, n_dims: int) -> np.ndarray:
        """Generate Latin Hypercube Sample in [0,1]^n_dims."""
        # Create Latin hypercube - each dimension divided into n_samples intervals
        samples = np.zeros((n_samples, n_dims))

        for i in range(n_dims):
            # Create stratified samples
            intervals = np.arange(n_samples, dtype=float) / n_samples
            # Add random jitter within each interval
            intervals += self.rng.uniform(0, 1/n_samples, n_samples)
            # Random permutation to decorrelate dimensions
            samples[:, i] = self.rng.permutation(intervals)

        return samples

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate population using Latin Hypercube Sampling for optimal space coverage."""
        population = []

        # Generate LHS samples for 10 dimensions (5 upper + 5 lower coefficients)
        n_coeffs = 5
        lhs_samples = self._lhs_sample(size, 2 * n_coeffs)

        # Transform LHS samples [0,1] to actual coefficient ranges
        # Upper coefficients (columns 0-4)
        upper_range = self.upper_bounds - 0.05  # Min bound is 0.05
        all_upper = 0.05 + lhs_samples[:, :n_coeffs] * upper_range

        # Lower coefficients (columns 5-9)
        lower_range = -0.05 - self.lower_bounds  # Max bound is -0.05
        all_lower = self.lower_bounds + lhs_samples[:, n_coeffs:] * lower_range

        # Create airfoils using list comprehension for speed
        population = [
            Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            )
            for i in range(size)
        ]

        return population

def generate_diverse_population() -> List[Airfoil]:
    """Generate diverse initial population with different LHS-based strategies."""
    all_airfoils = []

    # Strategy 1: Standard LHS distribution (20 airfoils)
    init1 = LHSCSTInitializer(seed=42)
    all_airfoils.extend(init1.generate_population(20))

    # Strategy 2: High-camber biased LHS (10 airfoils)
    init2 = LHSCSTInitializer(seed=123)
    init2.upper_bounds *= 1.2
    init2.lower_bounds *= 0.8  # Less negative = more camber
    all_airfoils.extend(init2.generate_population(10))

    # Strategy 3: Thick airfoils using LHS (10 airfoils)
    init3 = LHSCSTInitializer(seed=456)
    thick_pop = init3.generate_population(10)
    for airfoil in thick_pop:
        # Increase thickness by scaling coefficients
        airfoil.upper_coeffs = [c * 1.3 for c in airfoil.upper_coeffs]
        airfoil.lower_coeffs = [c * 1.3 for c in airfoil.lower_coeffs]
    all_airfoils.extend(thick_pop)

    # Strategy 4: Symmetric airfoils using LHS (10 airfoils)
    init4 = LHSCSTInitializer(seed=789)
    sym_pop = init4.generate_population(10)
    for airfoil in sym_pop:
        # Make symmetric
        airfoil.lower_coeffs = [-c for c in airfoil.upper_coeffs]
    all_airfoils.extend(sym_pop)

    return all_airfoils

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population fitness using optimized evaluation."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Use analytical evaluation for speed in population initialization
    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def main():
    """Main population initialization."""
    print("Generating diverse CST airfoil population using Latin Hypercube Sampling...")

    # Generate population
    population = generate_diverse_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
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
    print(f"Generated {len(population)} airfoils using LHS")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Mean fitness: {np.mean([f for _, f in evaluated]):.4f}")
    print(f"Valid designs: {sum(1 for _, f in evaluated if f > 0)}")

    return best_airfoils

if __name__ == "__main__":
    main()