#!/usr/bin/env python3
"""
Gen 11a: Lookup Table Optimization Mutation
Performance enhancement through precomputed Gray code lookup tables.
Based on gen9x with added extensive lookup table caching for Gray code operations
and vectorized bit manipulation to eliminate repeated calculations.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import itertools

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class HyperVectorizedSobol:
    """
    Optimized Sobol sequence generator with extensive lookup table caching
    for maximum performance through precomputed Gray code operations.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        self.direction_numbers = self._get_direction_numbers(dimension)

        # MUTATION: Massive lookup table cache for Gray codes and bit operations
        self.gray_cache = self._precompute_gray_codes(8192)  # 4x larger cache
        self.xor_cache = self._precompute_xor_operations()

        # NEW: Precompute lookup tables for bit patterns and operations
        self.bit_lookup = self._precompute_bit_lookup_tables()
        self.gray_to_sobol_lut = self._precompute_gray_to_sobol_lookup()

        self.direction_matrix = self._prepare_direction_matrix()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes with massive cache size."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Enhanced XOR operation cache."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_lookup_tables(self) -> Dict[str, np.ndarray]:
        """NEW: Precompute lookup tables for common bit operations."""
        # Lookup table for bit positions
        bit_positions = np.arange(32, dtype=np.uint32)
        bit_masks = 1 << bit_positions

        # Precompute common bit patterns (first 256 values)
        common_patterns = np.arange(256, dtype=np.uint32)
        bit_counts = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            bit_counts[i] = bin(i).count('1')

        return {
            'bit_positions': bit_positions,
            'bit_masks': bit_masks,
            'common_patterns': common_patterns,
            'bit_counts': bit_counts
        }

    def _precompute_gray_to_sobol_lookup(self) -> Dict[int, np.ndarray]:
        """NEW: Precompute Sobol values for common Gray codes."""
        lookup = {}
        # Precompute for first 1024 Gray codes across all dimensions
        for gray in range(min(1024, len(self.gray_cache))):
            sobol_values = np.zeros(self.dimension, dtype=np.float64)

            for d in range(self.dimension):
                point_val = 0
                temp_gray = gray
                bit_pos = 0

                # Use lookup table for first 8 bits
                if temp_gray < 256:
                    # Fast lookup for common patterns
                    pattern = temp_gray & 0xFF
                    for bit in range(8):
                        if pattern & (1 << bit):
                            point_val ^= self.direction_numbers[d, bit]
                    temp_gray >>= 8
                    bit_pos = 8

                # Handle remaining bits
                while temp_gray and bit_pos < 32:
                    if temp_gray & 1:
                        point_val ^= self.direction_numbers[d, bit_pos]
                    temp_gray >>= 1
                    bit_pos += 1

                sobol_values[d] = point_val / (1 << 32)

            lookup[gray] = sobol_values

        return lookup

    def _prepare_direction_matrix(self) -> np.ndarray:
        """Prepare direction numbers for vectorized operations."""
        bit_positions = np.arange(32, dtype=np.uint32)
        self.bit_masks = 1 << bit_positions
        return self.direction_numbers

    def generate_points(self, n: int) -> np.ndarray:
        """Generate points using optimized lookup table approach."""
        points = np.zeros((n, self.dimension))

        # Use massive Gray code cache when possible
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Fallback to vectorized generation
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # NEW: Use lookup table for small Gray codes
        for i in range(n):
            gray = gray_codes[i]

            if gray in self.gray_to_sobol_lut:
                # Fast lookup table access
                points[i] = self.gray_to_sobol_lut[gray]
            else:
                # Optimized computation for larger Gray codes
                if n <= 64:  # Small batch - use vectorization
                    # Create bit mask matrix for batch processing
                    gray_expanded = gray_codes[i:i+1, np.newaxis]
                    bit_masks_expanded = self.bit_masks[np.newaxis, :]
                    bit_tests = (gray_expanded & bit_masks_expanded) != 0

                    for d in range(self.dimension):
                        dir_nums = self.direction_numbers[d, :]
                        masked_dirs = np.where(bit_tests[0], dir_nums, 0)
                        point_val = np.bitwise_xor.reduce(masked_dirs)
                        points[i, d] = point_val / (1 << 32)
                else:
                    # Optimized unrolled approach for larger batches
                    for d in range(self.dimension):
                        dir_nums = self.direction_numbers[d]
                        point_val = 0

                        # Use bit lookup for first 8 bits
                        pattern = gray & 0xFF
                        if pattern in self.bit_lookup['common_patterns']:
                            for bit in range(8):
                                if pattern & (1 << bit):
                                    point_val ^= dir_nums[bit]

                        # Handle remaining bits
                        remaining_gray = gray >> 8
                        bit_pos = 8
                        while remaining_gray and bit_pos < 32:
                            if remaining_gray & 1:
                                point_val ^= dir_nums[bit_pos]
                            remaining_gray >>= 1
                            bit_pos += 1

                        points[i, d] = point_val / (1 << 32)

        return points

class ApexNelderMead:
    """
    Ultimate Nelder-Mead optimizer combining all enhancements from gen5x and gen4x.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 45, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Apex Nelder-Mead optimization with all enhancements from both parents."""
        n = len(x0)

        # Enhanced simplex initialization from gen5x
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Enhanced adaptive step sizing from gen5x
        step_size = 0.095 * np.abs(x0)  # Slightly larger than gen5x
        step_size = np.where(step_size < 0.017, 0.017, step_size)

        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += step_size[i]

        # Evaluate initial simplex
        f_values = np.array([objective_func(x) for x in simplex])
        best_f_history = []

        for iteration in range(maxiter):
            # Sort simplex by function values
            indices = np.argsort(f_values)
            simplex = simplex[indices]
            f_values = f_values[indices]

            # Track history for enhanced convergence from gen5x
            best_f_history.append(f_values[0])

            # Enhanced convergence criteria from gen5x
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Ultra-advanced stagnation check from gen5x (enhanced to 9 iterations)
            if len(best_f_history) >= 10:
                recent_improvement = best_f_history[-10] - best_f_history[-1]
                if recent_improvement < tolerance * 0.06:  # Even stricter than gen5x
                    break

            # Centroid of n best points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + self.alpha * (centroid - simplex[-1])
            f_reflected = objective_func(reflected)

            if f_values[0] <= f_reflected < f_values[-2]:
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

class HyperCSTInitializer:
    """
    Ultimate hybrid CST initializer combining the best from all three parents:
    - SIMD vectorized Sobol generation from gen8a
    - Advanced multi-strategy Gaussian population from gen5x
    - Robust caching and optimization pipeline from gen4x
    - NEW: Enhanced adaptive scaling and improved convergence criteria
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen5x
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays from gen4x/gen8a optimization
        self._cache_bounds_arrays()

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for maximum vectorized speed."""
        self.upper_bounds_cache = np.array([b for b, _ in self.bounds['upper_coeffs']])
        self.upper_ranges_cache = np.array([r - l for l, r in self.bounds['upper_coeffs']])
        self.lower_bounds_cache = np.array([b for b, _ in self.bounds['lower_coeffs']])
        self.lower_ranges_cache = np.array([r - l for l, r in self.bounds['lower_coeffs']])

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Convert coefficient vector to Airfoil object."""
        upper_coeffs = np.clip(coeffs[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(coeffs[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def objective_function(self, coeffs: np.ndarray) -> float:
        """Enhanced objective function with advanced penalty system from gen5x."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Advanced penalty system from gen5x with slight enhancement
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 17 * (0.06 - max_t)  # Increased from gen5x
                if max_t > 0.18:
                    penalty += 17 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points using cached vectorized operations from gen8a/gen4x."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 42) -> List[Airfoil]:
        """Generate enhanced Sobol population using hybrid vectorized approach."""
        sobol = HyperVectorizedSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_apex_gaussian_population(self, size: int = 32) -> List[Airfoil]:
        """Generate ultimate Gaussian population using enhanced multi-strategy from gen5x."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution from gen5x with improvements
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.175, 0.068, (strategy1_size, n_coeffs))  # Slightly adjusted
        all_lower = self.rng.normal(-0.102, 0.050, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: Enhanced high-performance biased from gen5x
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.200, 0.075, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.072, 0.040, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Enhanced thickness-optimized from gen5x
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.055, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.138, 0.050, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Enhanced near-symmetric with improved bias from gen5x
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.140, 0.040, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor from gen5x with slight improvement
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """Enhanced optimization combining approaches from all parents."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Enhanced batch evaluation from gen8a approach
        batch_size = 12
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select enhanced number of top candidates from gen5x
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = ApexNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Enhanced optimization parameters combining all parents
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=40,  # Increased from gen5x
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def apex_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 26) -> List[Airfoil]:
        """Ultimate adaptive refinement combining vectorized generation with enhanced strategies."""
        refined = []
        sobol = HyperVectorizedSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Enhanced to top 8 from gen5x
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations with hybrid vectorized Sobol
            local_points = sobol.generate_points(refinement_size // 8)

            # Enhanced adaptive scale from gen5x with improvements
            base_scale = 0.100  # Slightly increased
            scale = base_scale * (1.0 + 0.15 * good_airfoils.index(airfoil))

            for point in local_points:
                # Enhanced perturbation strategy from gen5x
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

def generate_population() -> List[Airfoil]:
    """Generate ultimate hybrid population combining all parent strategies."""
    print("Generating lookup table optimized population...")

    initializer = HyperCSTInitializer(seed=999999)  # Ultimate apex seed

    # Stage 1: Generate enhanced base population with optimal sizing
    sobol_pop = initializer.generate_sobol_population(42)      # Increased from all parents
    gaussian_pop = initializer.generate_apex_gaussian_population(32)  # Ultimate multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Ultimate adaptive refinement around best solutions
    refined_pop = initializer.apex_adaptive_refinement(optimized_pop[:8], refinement_size=26)

    # Combine all populations with intelligent deduplication from gen5x
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with optimal tolerance
    unique_population = []
    tolerance = 6e-5  # Tighter than gen5x

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using lookup table optimization")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching from all parents."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing combining all approaches
    batch_size = 16
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main lookup table optimized initialization."""
    print("Generating lookup table optimized population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:35]):  # Save top 35
        filename = f"results/lookup_optimized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['lookup_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "11a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "lookup_rank": i
        })

    # Ultimate diversity metrics combining all parent approaches
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} lookup table optimized airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 15 mean: {np.mean(valid_fitnesses[:15]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()