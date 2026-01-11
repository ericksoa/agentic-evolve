#!/usr/bin/env python3
"""
Gen 0 Variant A: Random Gaussian CST Population Initializer
Performance-optimized using vectorized NumPy operations and SIMD-friendly algorithms.
Generates diverse CST airfoil coefficients using Gaussian distributions with different means/std.
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

class GaussianCSTInitializer:
    """Vectorized Gaussian-based CST coefficient initialization."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Pre-computed coefficient ranges for performance
        self.upper_bounds = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        self.lower_bounds = np.array([-0.25, -0.2, -0.15, -0.12, -0.1])

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate population using vectorized operations for maximum performance."""
        population = []

        # Vectorized coefficient generation - process all at once
        n_coeffs = 5
        all_upper = self.rng.normal(0.15, 0.05, (size, n_coeffs))
        all_lower = self.rng.normal(-0.12, 0.04, (size, n_coeffs))

        # Apply bounds using vectorized clipping
        all_upper = np.clip(all_upper, 0.05, self.upper_bounds)
        all_lower = np.clip(all_lower, self.lower_bounds, -0.05)

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
    """Generate diverse initial population with different strategies."""
    all_airfoils = []

    # Strategy 1: Standard distribution (20 airfoils)
    init1 = GaussianCSTInitializer(seed=42)
    all_airfoils.extend(init1.generate_population(20))

    # Strategy 2: High-camber biased (10 airfoils)
    init2 = GaussianCSTInitializer(seed=123)
    init2.upper_bounds *= 1.2
    init2.lower_bounds *= 0.8  # Less negative = more camber
    all_airfoils.extend(init2.generate_population(10))

    # Strategy 3: Thick airfoils (10 airfoils)
    init3 = GaussianCSTInitializer(seed=456)
    thick_pop = init3.generate_population(10)
    for airfoil in thick_pop:
        # Increase thickness by scaling coefficients
        airfoil.upper_coeffs = [c * 1.3 for c in airfoil.upper_coeffs]
        airfoil.lower_coeffs = [c * 1.3 for c in airfoil.lower_coeffs]
    all_airfoils.extend(thick_pop)

    # Strategy 4: Symmetric airfoils (10 airfoils)
    init4 = GaussianCSTInitializer(seed=789)
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
    print("Generating diverse CST airfoil population...")

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
    print(f"Generated {len(population)} airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Mean fitness: {np.mean([f for _, f in evaluated]):.4f}")
    print(f"Valid designs: {sum(1 for _, f in evaluated if f > 0)}")

    return best_airfoils

if __name__ == "__main__":
    main()