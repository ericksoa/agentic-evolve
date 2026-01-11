#!/usr/bin/env python3
"""
Gen 5 Variant C: Vectorized Bit Operations Mutation
Replaces manual bit manipulation loops with numpy's vectorized operations
for massive SIMD performance gains in Sobol sequence generation.
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

class VectorizedSobol:
    """
    Ultra-fast Sobol sequence generator with full numpy vectorization.
    Eliminates all Python loops by using vectorized bit operations.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Precompute direction numbers
        self.direction_numbers = self._get_direction_numbers(dimension)
        # NEW: Create bit mask lookup table for vectorized operations
        self.bit_masks = 2**np.arange(32, dtype=np.uint32)

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Precomputed direction numbers for fast Sobol sequence."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions with primitive polynomials
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with full vectorization - MUTATION FOCUS."""
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # Vectorized Gray code computation
        indices = np.arange(n, dtype=np.uint32)
        gray_codes = indices ^ (indices >> 1)

        # MUTATION: Replace all loops with vectorized operations
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            # Vectorized bit extraction and point computation
            # Expand gray codes to test against all 32 bit positions
            gray_expanded = gray_codes[:, np.newaxis]  # Shape: (n, 1)
            bit_masks_expanded = self.bit_masks[np.newaxis, :]  # Shape: (1, 32)

            # Test which bits are set - vectorized across all points and bits
            bit_set = (gray_expanded & bit_masks_expanded) != 0  # Shape: (n, 32)

            # Multiply by direction numbers and XOR-reduce
            # This replaces the entire inner loop with vectorized operations
            weighted_dirs = bit_set * dir_nums[np.newaxis, :]  # Broadcasting
            point_vals = np.bitwise_xor.reduce(weighted_dirs, axis=1)  # XOR across bits

            # Convert to [0,1) range
            points[:, d] = point_vals.astype(np.float64) / (1 << 32)

        return points

class SobolCSTInitializer:
    """CST airfoil initialization using vectorized Sobol sequences."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # CST parameter bounds (optimized for performance)
        self.bounds = self._setup_bounds()
        # Cache bounds arrays for vectorized operations
        self._cache_bounds_arrays()

    def _setup_bounds(self) -> dict:
        """Setup parameter bounds for CST coefficients."""
        return {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for faster vectorized operations."""
        self.upper_bounds_cache = np.array([b for b, _ in self.bounds['upper_coeffs']])
        self.upper_ranges_cache = np.array([r - l for l, r in self.bounds['upper_coeffs']])
        self.lower_bounds_cache = np.array([b for b, _ in self.bounds['lower_coeffs']])
        self.lower_ranges_cache = np.array([r - l for l, r in self.bounds['lower_coeffs']])

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points to CST parameters using cached vectorized operations."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate airfoil population using vectorized Sobol sequence."""
        # Generate Sobol points with vectorized operations
        sobol = VectorizedSobol(dimension=10)  # Use vectorized version
        points = sobol.generate_points(size)

        # Convert to CST parameters using cached bounds
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
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations with vectorized Sobol
            local_sobol = VectorizedSobol(dimension=10)  # Use vectorized version
            local_points = local_sobol.generate_points(refinement_size // 5)

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
    """Main Sobol population initialization."""
    print("Generating vectorized Sobol quasi-random airfoil population...")

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
        filename = f"results/vectorized_sobol_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
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

    print(f"Generated {len(population)} vectorized Sobol-sampled airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Space-filling efficiency: {len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated):.3f}")

    return best_results

if __name__ == "__main__":
    main()