#!/usr/bin/env python3
"""
Gen 4 Variant C: Hybrid Halton-Sobol Quasi-Random Generation
Mutation: Algorithm family change - Replace pure Sobol with hybrid Halton-Sobol sequences.
Halton sequences often have better low-dimensional performance and faster generation.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple
import itertools

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastHaltonSobol:
    """
    Hybrid Halton-Sobol sequence generator optimized for airfoil design.
    Uses Halton for first 5 dimensions (upper coeffs) and Sobol for rest (lower coeffs).
    Halton sequences often converge faster for low dimensions.
    """

    def __init__(self, dimension: int = 10, max_points: int = 1024):
        self.dimension = dimension
        self.max_points = max_points

        # Optimized prime bases for Halton sequence (first 5 dimensions)
        self.halton_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dimension//2]

        # Precompute Sobol for remaining dimensions
        self.sobol_dimension = dimension - len(self.halton_bases)
        if self.sobol_dimension > 0:
            self._precompute_sobol_tables()

        # Precompute inverse bases for Halton
        self._precompute_halton_tables()

    def _precompute_halton_tables(self):
        """Precompute Halton sequence lookup tables."""
        # Precompute inverse powers for each base
        max_digits = 32  # Sufficient for most applications
        self.inv_powers = {}

        for base in self.halton_bases:
            inv_base = 1.0 / base
            powers = [1.0]
            for i in range(1, max_digits):
                powers.append(powers[-1] * inv_base)
            self.inv_powers[base] = np.array(powers, dtype=np.float64)

    def _precompute_sobol_tables(self):
        """Precompute simplified Sobol tables for remaining dimensions."""
        if self.sobol_dimension <= 0:
            return

        # Gray codes
        self.gray_codes = np.zeros(self.max_points, dtype=np.uint32)
        for i in range(self.max_points):
            self.gray_codes[i] = i ^ (i >> 1)

        # Simple direction numbers for remaining dimensions
        self.direction_numbers = np.zeros((self.sobol_dimension, 32), dtype=np.uint32)

        # Efficient bit-reversal for first dimension
        for i in range(32):
            self.direction_numbers[0, i] = 1 << (31 - i)

        # Simple XOR pattern for other dimensions
        for d in range(1, self.sobol_dimension):
            self.direction_numbers[d, 0] = 1 << 31
            for i in range(1, 32):
                self.direction_numbers[d, i] = self.direction_numbers[d, i-1] ^ (self.direction_numbers[d, i-1] >> 1)

    def _halton_coordinate(self, index: int, base: int) -> float:
        """Generate single Halton coordinate using precomputed powers."""
        if base not in self.inv_powers:
            return 0.0

        powers = self.inv_powers[base]
        result = 0.0
        i = index
        digit_pos = 0

        while i > 0 and digit_pos < len(powers):
            digit = i % base
            result += digit * powers[digit_pos + 1]
            i //= base
            digit_pos += 1

        return result

    def _generate_halton_points(self, n: int) -> np.ndarray:
        """Generate Halton points for first half of dimensions."""
        num_halton = len(self.halton_bases)
        points = np.zeros((n, num_halton), dtype=np.float64)

        # Vectorized Halton generation
        for d, base in enumerate(self.halton_bases):
            for i in range(n):
                points[i, d] = self._halton_coordinate(i + 1, base)  # Skip 0 index

        return points

    def _generate_sobol_points(self, n: int) -> np.ndarray:
        """Generate Sobol points for remaining dimensions."""
        if self.sobol_dimension <= 0:
            return np.empty((n, 0))

        points = np.zeros((n, self.sobol_dimension), dtype=np.float64)

        for i in range(n):
            if i < len(self.gray_codes):
                gray = self.gray_codes[i]

                for d in range(self.sobol_dimension):
                    point_val = 0
                    bit_pos = 0
                    gray_copy = gray

                    while gray_copy and bit_pos < 32:
                        if gray_copy & 1:
                            point_val ^= self.direction_numbers[d, bit_pos]
                        gray_copy >>= 1
                        bit_pos += 1

                    points[i, d] = point_val / (1 << 32)

        return points

    def generate_points(self, n: int) -> np.ndarray:
        """Generate hybrid Halton-Sobol sequence points."""
        n = min(n, self.max_points)

        # Generate Halton points for upper coefficients
        halton_points = self._generate_halton_points(n)

        # Generate Sobol points for lower coefficients
        if self.sobol_dimension > 0:
            sobol_points = self._generate_sobol_points(n)
            # Combine Halton and Sobol
            points = np.hstack([halton_points, sobol_points])
        else:
            points = halton_points

        return points.astype(np.float32)

class SobolCSTInitializer:
    """CST airfoil initialization using optimized Sobol sequences."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # CST parameter bounds (optimized for performance)
        self.bounds = self._setup_bounds()

    def _setup_bounds(self) -> dict:
        """Setup parameter bounds for CST coefficients."""
        return {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points to CST parameters using vectorized operations."""
        n_points = sobol_points.shape[0]
        upper_bounds = np.array([b for b, _ in self.bounds['upper_coeffs']], dtype=np.float32)
        upper_ranges = np.array([r - l for l, r in self.bounds['upper_coeffs']], dtype=np.float32)
        lower_bounds = np.array([b for b, _ in self.bounds['lower_coeffs']], dtype=np.float32)
        lower_ranges = np.array([r - l for l, r in self.bounds['lower_coeffs']], dtype=np.float32)

        # Vectorized transformation with float32 for speed
        upper_coeffs_all = upper_bounds + sobol_points[:, :5] * upper_ranges
        lower_coeffs_all = lower_bounds + sobol_points[:, 5:10] * lower_ranges

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate airfoil population using hybrid Halton-Sobol sequence."""
        # Generate hybrid points with Halton for upper, Sobol for lower
        generator = FastHaltonSobol(dimension=10, max_points=max(size, 1024))
        points = generator.generate_points(size)

        # Convert to CST parameters
        cst_params = self.sobol_to_cst_params(points)

        # Create airfoils using list comprehension for speed
        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

class AdaptiveSamplingMixin:
    """Mixin for adaptive sampling in regions of interest."""

    def refine_around_good_points(self, good_airfoils: List[Airfoil],
                                 refinement_size: int = 20) -> List[Airfoil]:
        """Refine sampling around promising airfoils."""
        refined = []

        for airfoil in good_airfoils[:5]:  # Top 5 to refine around
            # Create local Sobol sequence around this airfoil
            center_upper = np.array(airfoil.upper_coeffs, dtype=np.float32)
            center_lower = np.array(airfoil.lower_coeffs, dtype=np.float32)

            # Generate local perturbations with hybrid sequence
            local_generator = FastHaltonSobol(dimension=10, max_points=512)
            local_points = local_generator.generate_points(refinement_size // 5)

            # Scale to smaller neighborhood
            scale = 0.1  # 10% neighborhood

            for point in local_points:
                # Perturb around center
                new_upper = center_upper + (point[:5] - 0.5) * scale * np.abs(center_upper)
                new_lower = center_lower + (point[5:] - 0.5) * scale * np.abs(center_lower)

                # Clamp to valid bounds
                new_upper = np.clip(new_upper, 0.01, 0.3)
                new_lower = np.clip(new_lower, -0.3, -0.01)

                refined.append(Airfoil(
                    upper_coeffs=new_upper.tolist(),
                    lower_coeffs=new_lower.tolist(),
                    zte=0.0
                ))

        return refined

class SobolInitializerWithAdaptive(SobolCSTInitializer, AdaptiveSamplingMixin):
    """Combined Sobol initialization with adaptive refinement."""
    pass

def generate_population() -> List[Airfoil]:
    """Generate Sobol-based quasi-random population."""
    initializer = SobolInitializerWithAdaptive(seed=789)

    # Initial Sobol population
    base_population = initializer.generate_population(40)

    # Quick evaluation to find promising regions
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    evaluated = []
    for airfoil in base_population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        evaluated.append((airfoil, fitness))

    # Sort and get top performers
    evaluated.sort(key=lambda x: x[1], reverse=True)
    good_airfoils = [a for a, f in evaluated[:10] if f > 0]

    # Adaptive refinement around good points
    refined = initializer.refine_around_good_points(good_airfoils, 20)

    # Combine populations
    all_population = base_population + refined

    return all_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population using batch processing."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Batch evaluation for better cache performance
    batch_size = 10
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid Halton-Sobol population initialization."""
    print("Generating hybrid Halton-Sobol quasi-random airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):
        filename = f"results/halton_sobol_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            json.dump(airfoil.to_dict(), f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "sobol_index": i,
            "coeffs_range": (
                max(airfoil.upper_coeffs) - min(airfoil.upper_coeffs),
                max(airfoil.lower_coeffs) - min(airfoil.lower_coeffs)
            )
        })

    print(f"Generated {len(population)} hybrid Halton-Sobol sampled airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Space-filling efficiency: {len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated):.3f}")

    return best_results

if __name__ == "__main__":
    main()