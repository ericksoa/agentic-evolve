#!/usr/bin/env python3
"""
Gen 2 Variant C: Lookup Table Optimized CST Population Initializer
Performance-optimized using pre-computed lookup tables for coefficient bounds and scaling factors.
Generates diverse CST airfoil coefficients using Gaussian distributions with cached parameters.
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

class OptimizedGaussianCSTInitializer:
    """Vectorized Gaussian-based CST coefficient initialization with lookup tables."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Pre-computed lookup tables for performance optimization
        self._setup_lookup_tables()

    def _setup_lookup_tables(self):
        """Pre-compute all coefficient ranges and scaling factors for maximum performance."""
        # Base coefficient bounds (same as parent)
        self.upper_bounds = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        self.lower_bounds = np.array([-0.25, -0.2, -0.15, -0.12, -0.1])

        # Pre-computed scaling factors lookup table for different strategies
        self.scaling_lookup = {
            'standard': (np.ones(5), np.ones(5)),
            'high_camber': (np.ones(5) * 1.2, np.ones(5) * 0.8),
            'thick': (np.ones(5) * 1.3, np.ones(5) * 1.3),
            'symmetric': (np.ones(5), -np.ones(5))  # Special case for symmetric
        }

        # Pre-computed Gaussian parameters lookup
        self.gaussian_params = {
            'standard': (0.15, 0.05, -0.12, 0.04),
            'high_camber': (0.18, 0.06, -0.10, 0.03),
            'thick': (0.20, 0.07, -0.15, 0.05),
            'symmetric': (0.15, 0.05, 0.15, 0.05)  # Same mean for both
        }

        # Pre-computed clipping bounds for each strategy
        self.clip_bounds = {}
        for strategy, (upper_scale, lower_scale) in self.scaling_lookup.items():
            if strategy != 'symmetric':
                self.clip_bounds[strategy] = (
                    np.clip(self.upper_bounds * upper_scale, 0.05, np.inf),
                    np.clip(self.lower_bounds * lower_scale, -np.inf, -0.05)
                )
            else:
                self.clip_bounds[strategy] = (
                    self.upper_bounds,
                    self.lower_bounds
                )

    def generate_population(self, size: int = 50, strategy: str = 'standard') -> List[Airfoil]:
        """Generate population using pre-computed lookup tables for maximum performance."""

        # Fetch pre-computed parameters from lookup table
        upper_mean, upper_std, lower_mean, lower_std = self.gaussian_params[strategy]
        upper_clip_max, lower_clip_min = self.clip_bounds[strategy]

        # Vectorized coefficient generation using cached parameters
        n_coeffs = 5
        all_upper = self.rng.normal(upper_mean, upper_std, (size, n_coeffs))
        all_lower = self.rng.normal(lower_mean, lower_std, (size, n_coeffs))

        # Apply bounds using pre-computed clipping bounds
        all_upper = np.clip(all_upper, 0.05, upper_clip_max)
        all_lower = np.clip(all_lower, lower_clip_min, -0.05)

        # Handle symmetric case with lookup table
        if strategy == 'symmetric':
            all_lower = -all_upper

        # Create airfoils using optimized list comprehension
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
    """Generate diverse initial population with cached strategy parameters."""
    all_airfoils = []

    # Use lookup table-based strategies for optimal performance
    strategies = [
        ('standard', 20, 42),
        ('high_camber', 10, 123),
        ('thick', 10, 456),
        ('symmetric', 10, 789)
    ]

    # Batch generate using pre-computed strategy parameters
    for strategy, count, seed in strategies:
        initializer = OptimizedGaussianCSTInitializer(seed=seed)
        population = initializer.generate_population(count, strategy)
        all_airfoils.extend(population)

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
    print("Generating diverse CST airfoil population with lookup table optimization...")

    # Generate population using cached parameters
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
    print(f"Generated {len(population)} airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Mean fitness: {np.mean([f for _, f in evaluated]):.4f}")
    print(f"Valid designs: {sum(1 for _, f in evaluated if f > 0)}")

    return best_airfoils

if __name__ == "__main__":
    main()