#!/usr/bin/env python3
"""
Gen 11x: Ultimate Apex Crossover Hybrid
Supreme evolutionary crossover combining the absolute best from three top parents:
- gen9x (113.9914): HyperVectorizedSobol with adaptive batch processing + ApexNelderMead
- gen9c (113.6363): UltraFastSobol SIMD vectorization + enhanced convergence
- gen5x (113.4354): ApexSobol ultra-caching + advanced multi-strategy Gaussian

Hybrid innovations:
- Unified SIMD-vectorized Sobol with adaptive batch optimization from gen9x + gen9c
- Supreme Nelder-Mead combining strictest convergence from gen9x with robustness
- Apex multi-strategy Gaussian blending all three parameter sets
- Enhanced population sizing and refinement strategies
- Optimized penalty systems and caching from best performers
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

class SupremeSobol:
    """
    Ultimate Sobol sequence generator combining:
    - Adaptive batch vectorization from gen9x for optimal performance scaling
    - Full SIMD vectorization from gen9c for maximum speed
    - Ultra-large caching from gen5x (2048 points) for lookup optimization
    - Enhanced direction number computation and XOR optimizations
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Ultra-large cache from gen5x with gen9x/gen9c optimizations
        self.direction_numbers = self._get_direction_numbers(dimension)
        self.gray_cache = self._precompute_gray_codes(2048)  # From gen5x
        self.xor_cache = self._precompute_xor_operations()
        # Enhanced direction matrix from gen9x for vectorized operations
        self.direction_matrix = self._prepare_direction_matrix()
        # Additional optimization caches
        self.bit_mask_cache = self._precompute_bit_masks()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation combining all parents."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions combining robustness from all parents
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes with ultra-large cache from gen5x."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Enhanced XOR operation cache from gen5x."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_masks(self) -> np.ndarray:
        """Additional bit mask cache from gen5x for maximum performance."""
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def _prepare_direction_matrix(self) -> np.ndarray:
        """Prepare direction numbers for vectorized operations from gen9x."""
        bit_positions = np.arange(32, dtype=np.uint32)
        self.bit_masks = 1 << bit_positions
        return self.direction_numbers

    def generate_points(self, n: int) -> np.ndarray:
        """Supreme point generation combining adaptive batching with SIMD vectorization."""
        points = np.zeros((n, self.dimension))

        # Use ultra-large cache from gen5x with gen9x/gen9c vectorization
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Fallback to vectorized generation from gen9c
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # Adaptive approach: combine gen9x strategy with gen9c SIMD optimization
        if n <= 48:  # Optimized threshold combining gen9x (64) and practical testing
            # Small batch - use gen9c full SIMD vectorization approach
            gray_expanded = gray_codes[:, np.newaxis]  # (n, 1)
            bit_masks_expanded = self.bit_masks[np.newaxis, :]  # (1, 32)

            # Vectorized bit testing: (n, 32) boolean array
            bit_tests = (gray_expanded & bit_masks_expanded) != 0

            # Process all dimensions with full vectorization
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d, :]  # (32,)

                # Vectorized XOR operations for all points simultaneously
                masked_dirs = np.where(bit_tests, dir_nums[np.newaxis, :], 0)

                # XOR all bits for each point (reduction along bit axis)
                point_vals = np.bitwise_xor.reduce(masked_dirs, axis=1)

                # Convert to [0,1) range
                points[:, d] = point_vals.astype(np.float64) / (1 << 32)
        else:
            # Larger batch - use gen9x/gen5x optimized unrolled approach
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]

                for i in range(n):
                    gray = gray_codes[i]
                    point_val = 0

                    # Extended unrolled inner loop from gen9x/gen5x for maximum performance
                    if gray & 1:
                        point_val ^= dir_nums[0]
                    if gray & 2:
                        point_val ^= dir_nums[1]
                    if gray & 4:
                        point_val ^= dir_nums[2]
                    if gray & 8:
                        point_val ^= dir_nums[3]
                    if gray & 16:
                        point_val ^= dir_nums[4]
                    if gray & 32:
                        point_val ^= dir_nums[5]
                    if gray & 64:
                        point_val ^= dir_nums[6]
                    if gray & 128:
                        point_val ^= dir_nums[7]

                    # Handle remaining bits with optimized approach from gen5x
                    remaining_gray = gray >> 8
                    bit_pos = 8
                    while remaining_gray and bit_pos < 32:
                        if remaining_gray & 1:
                            point_val ^= dir_nums[bit_pos]
                        remaining_gray >>= 1
                        bit_pos += 1

                    points[i, d] = point_val / (1 << 32)

        return points

class SupremeNelderMead:
    """
    Ultimate Nelder-Mead optimizer combining the strictest convergence from gen9x
    with enhanced robustness from gen5x and optimization parameters from gen9c.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 42, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Supreme Nelder-Mead optimization combining all enhancements."""
        n = len(x0)

        # Enhanced simplex initialization from gen9x with gen5x robustness
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Optimized step sizing combining gen9x and gen5x approaches
        step_size = 0.096 * np.abs(x0)  # Between gen9x (0.095) and gen5x (0.09)
        step_size = np.where(step_size < 0.017, 0.017, step_size)  # From gen9x

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

            # Track history for enhanced convergence
            best_f_history.append(f_values[0])

            # Enhanced convergence criteria from gen9x
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Strictest stagnation check combining gen9x (10 iterations) with enhanced tolerance
            if len(best_f_history) >= 11:  # Even stricter than gen9x
                recent_improvement = best_f_history[-11] - best_f_history[-1]
                if recent_improvement < tolerance * 0.055:  # Stricter than gen9x (0.06)
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

class SupremeCSTInitializer:
    """
    Ultimate CST initializer combining the absolute best from all three parents:
    - Enhanced bounds and penalty systems from gen9x (penalty factor 17)
    - Multi-strategy Gaussian diversification from all parents
    - Vectorized operations and caching from gen5x/gen9c
    - Optimized population sizing and refinement strategies
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen9x with slight optimization
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for vectorized operations
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
        """Enhanced objective function with supreme penalty system from gen9x."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Supreme penalty system from gen9x with slight enhancement
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 17.5 * (0.06 - max_t)  # Enhanced from gen9x (17)
                if max_t > 0.18:
                    penalty += 17.5 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points using cached vectorized operations."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 40) -> List[Airfoil]:
        """Generate supreme Sobol population with optimal size blending all parents."""
        sobol = SupremeSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_supreme_gaussian_population(self, size: int = 30) -> List[Airfoil]:
        """Generate supreme Gaussian population blending strategies from all parents."""
        population = []
        n_coeffs = 5

        # Strategy 1: Blended standard distribution from all parents
        strategy1_size = size // 4
        # Blend parameters: gen9x(0.175,0.068), gen9c(0.16,0.06), gen5x(0.17,0.065)
        mean_upper = (0.175 + 0.16 + 0.17) / 3  # ≈ 0.168
        std_upper = (0.068 + 0.06 + 0.065) / 3  # ≈ 0.064
        mean_lower = (-0.102 + -0.11 + -0.105) / 3  # ≈ -0.106
        std_lower = (0.050 + 0.045 + 0.048) / 3  # ≈ 0.048

        all_upper = self.rng.normal(mean_upper, std_upper, (strategy1_size, n_coeffs))
        all_lower = self.rng.normal(mean_lower, std_lower, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: Blended high-performance biased from all parents
        strategy2_size = size // 4
        # Blend: gen9x(0.200,0.075), gen9c(0.19,0.07), gen5x(0.195,0.072)
        mean_upper_bias = (0.200 + 0.19 + 0.195) / 3  # ≈ 0.195
        std_upper_bias = (0.075 + 0.07 + 0.072) / 3  # ≈ 0.072
        mean_lower_bias = (-0.072 + -0.08 + -0.075) / 3  # ≈ -0.076
        std_lower_bias = (0.040 + 0.035 + 0.038) / 3  # ≈ 0.038

        upper_biased = self.rng.normal(mean_upper_bias, std_upper_bias, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(mean_lower_bias, std_lower_bias, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Blended thickness-optimized from all parents
        strategy3_size = size // 4
        # Blend: gen9x(0.225,0.055,1.15), gen9c(0.21,0.05,1.1), gen5x(0.22,0.052,1.12)
        mean_thick_upper = (0.225 + 0.21 + 0.22) / 3  # ≈ 0.218
        std_thick_upper = (0.055 + 0.05 + 0.052) / 3  # ≈ 0.052
        mean_thick_lower = (-0.138 + -0.14 + -0.142) / 3  # ≈ -0.140
        std_thick_lower = (0.050 + 0.045 + 0.047) / 3  # ≈ 0.047
        scale_factor = (1.15 + 1.1 + 1.12) / 3  # ≈ 1.12

        thick_upper = self.rng.normal(mean_thick_upper, std_thick_upper, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(mean_thick_lower, std_thick_lower, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * scale_factor, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * scale_factor, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Blended near-symmetric with optimized bias
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        # Blend: gen9x(0.140,0.040), gen9c(0.13,0.035), gen5x(0.135,0.038)
        mean_sym = (0.140 + 0.13 + 0.135) / 3  # ≈ 0.135
        std_sym = (0.040 + 0.035 + 0.038) / 3  # ≈ 0.038

        sym_upper = self.rng.normal(mean_sym, std_sym, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Blend asymmetry factors: gen9x(0.93,1.08), gen9c(0.95,1.05), gen5x(0.94,1.07)
            asym_min = (0.93 + 0.95 + 0.94) / 3  # ≈ 0.94
            asym_max = (1.08 + 1.05 + 1.07) / 3  # ≈ 1.067
            lower_list = [-c * self.rng.uniform(asym_min, asym_max) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 15) -> List[Airfoil]:
        """Enhanced optimization combining approaches from all parents."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Optimized batch size combining all approaches (gen9x:12, gen9c:8, gen5x:10)
        batch_size = 10
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select optimal number of top candidates (blend: 16,12,14 → 15)
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = SupremeNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Enhanced optimization parameters blending all parents
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=42,  # Enhanced from best performing gen9x (40)
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def supreme_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Supreme adaptive refinement combining all parent strategies."""
        refined = []
        sobol = SupremeSobol(dimension=10)

        # Blend top candidates: gen9x(8), gen9c(6), gen5x(7) → 8 (best performer)
        for airfoil in good_airfoils[:8]:
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations with supreme vectorized Sobol
            local_points = sobol.generate_points(refinement_size // 8)

            # Blend adaptive scales: gen9x(0.100,0.15), gen9c(0.09,0.1), gen5x(0.095,0.12)
            base_scale = (0.100 + 0.09 + 0.095) / 3  # ≈ 0.095
            scale_factor = (0.15 + 0.1 + 0.12) / 3  # ≈ 0.123
            scale = base_scale * (1.0 + scale_factor * good_airfoils.index(airfoil))

            for point in local_points:
                # Enhanced perturbation strategy from best parents
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
    """Generate supreme hybrid population combining all three parent strategies."""
    print("Generating supreme crossover hybrid population...")

    # Use supreme hybrid seed combining aspects of all three parents
    initializer = SupremeCSTInitializer(seed=777777)  # Supreme crossover seed

    # Stage 1: Generate supreme base population with optimal sizing from all parents
    # Sizes: gen9x(42+32), gen9c(35+25), gen5x(38+28) → optimized blend (40+30)
    sobol_pop = initializer.generate_sobol_population(40)
    gaussian_pop = initializer.generate_supreme_gaussian_population(30)

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=15)

    # Stage 3: Supreme adaptive refinement around best solutions
    # Sizes: gen9x(26), gen9c(18), gen5x(22) → optimized (24)
    refined_pop = initializer.supreme_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with optimal tolerance blending all approaches
    # Tolerances: gen9x(6e-5), gen9c(1e-4), gen5x(8e-5) → optimized (7e-5)
    unique_population = []
    tolerance = 7e-5

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using supreme crossover hybrid approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching from all parents."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing blending all approaches (gen9x:16, gen9c:12, gen5x:14)
    batch_size = 14
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main supreme crossover hybrid initialization."""
    print("Generating supreme crossover hybrid population with ultimate optimizations...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    # Save count: blend gen9x(35), gen9c(30), gen5x(32) → optimized (33)
    for i, (airfoil, fitness) in enumerate(evaluated[:33]):
        filename = f"results/supreme_crossover_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['supreme_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "11x"
            data['parents'] = ["gen9x", "gen9c", "gen5x"]
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "supreme_rank": i
        })

    # Supreme diversity metrics combining all parent approaches
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} supreme crossover hybrid airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        # Blend top means: gen9x(top15), gen9c(top10), gen5x(top12) → top13
        print(f"Top 13 mean: {np.mean(valid_fitnesses[:13]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()