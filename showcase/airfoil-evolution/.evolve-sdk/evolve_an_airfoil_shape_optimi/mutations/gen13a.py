#!/usr/bin/env python3
"""
Gen 13a: Memory-Optimized Cache Hierarchy Mutation
Performance mutation focused on cache optimization and memory access patterns:
- Pre-allocated memory pools to reduce garbage collection
- Spatial locality optimization in data structures
- Hot path computation caching with LRU eviction
- Memory-aligned vectorized operations
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

class MemoryOptimizedSobol:
    """
    Cache-optimized Sobol sequence generator with pre-allocated memory pools
    and spatial locality improvements for maximum performance.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        self.direction_numbers = self._get_direction_numbers(dimension)

        # Pre-allocate memory pools for hot path operations
        self.gray_cache = self._precompute_gray_codes(4096)  # Doubled cache size
        self.xor_cache = self._precompute_xor_operations()
        self.direction_matrix = self._prepare_direction_matrix()

        # Memory-aligned arrays for vectorized operations
        self._preallocate_work_arrays()

    def _preallocate_work_arrays(self):
        """Pre-allocate work arrays to eliminate allocations in hot paths."""
        max_batch_size = 256
        # Pre-allocate commonly used arrays with memory alignment
        self.work_gray_codes = np.empty(max_batch_size, dtype=np.uint32)
        self.work_bit_tests = np.empty((max_batch_size, 32), dtype=bool)
        self.work_masked_dirs = np.empty((max_batch_size, 32), dtype=np.uint32)
        self.work_points = np.empty((max_batch_size, self.dimension), dtype=np.float64)

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation with memory locality."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions with improved cache locality
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes with enlarged cache for better hit rates."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """XOR operation cache with memory alignment."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _prepare_direction_matrix(self) -> np.ndarray:
        """Prepare direction numbers for cache-optimized vectorized operations."""
        bit_positions = np.arange(32, dtype=np.uint32)
        self.bit_masks = 1 << bit_positions
        return self.direction_numbers

    def generate_points(self, n: int) -> np.ndarray:
        """Memory-optimized point generation with pre-allocated work arrays."""
        # Use pre-allocated array if possible, otherwise allocate
        if n <= self.work_points.shape[0]:
            points = self.work_points[:n]
        else:
            points = np.zeros((n, self.dimension))

        # Use enlarged cache for better hit rates
        if n <= len(self.gray_cache):
            if n <= self.work_gray_codes.shape[0]:
                gray_codes = self.work_gray_codes[:n]
                gray_codes[:] = self.gray_cache[:n]
            else:
                gray_codes = self.gray_cache[:n]
        else:
            # Fallback to vectorized generation
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # Optimized processing with better memory access patterns
        if n <= 128:  # Increased threshold for vectorized approach
            # Use pre-allocated work arrays for better cache locality
            if n <= self.work_bit_tests.shape[0]:
                bit_tests = self.work_bit_tests[:n]
                masked_dirs = self.work_masked_dirs[:n]
            else:
                bit_tests = np.empty((n, 32), dtype=bool)
                masked_dirs = np.empty((n, 32), dtype=np.uint32)

            # Create bit mask matrix with memory-aligned access
            gray_expanded = gray_codes[:, np.newaxis]
            bit_masks_expanded = self.bit_masks[np.newaxis, :]

            # Optimized bit testing with spatial locality
            bit_tests[:] = (gray_expanded & bit_masks_expanded) != 0

            # Process dimensions in order for better cache behavior
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d, :]

                # Cache-friendly vectorized XOR operations
                masked_dirs[:] = np.where(bit_tests, dir_nums[np.newaxis, :], 0)

                # Optimized XOR reduction with memory alignment
                point_vals = np.bitwise_xor.reduce(masked_dirs, axis=1)

                # Convert to [0,1) range with in-place operation
                points[:, d] = point_vals.astype(np.float64) / (1 << 32)
        else:
            # Memory-optimized unrolled approach for larger batches
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]

                # Process in chunks for better cache locality
                chunk_size = 64
                for chunk_start in range(0, n, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n)

                    for i in range(chunk_start, chunk_end):
                        gray = gray_codes[i]
                        point_val = 0

                        # Fully unrolled inner loop for maximum performance
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
                        if gray & 256:
                            point_val ^= dir_nums[8]
                        if gray & 512:
                            point_val ^= dir_nums[9]
                        if gray & 1024:
                            point_val ^= dir_nums[10]
                        if gray & 2048:
                            point_val ^= dir_nums[11]

                        # Handle remaining bits efficiently
                        remaining_gray = gray >> 12
                        bit_pos = 12
                        while remaining_gray and bit_pos < 32:
                            if remaining_gray & 1:
                                point_val ^= dir_nums[bit_pos]
                            remaining_gray >>= 1
                            bit_pos += 1

                        points[i, d] = point_val / (1 << 32)

        return points if n > self.work_points.shape[0] else points.copy()

class CachedNelderMead:
    """
    Cache-optimized Nelder-Mead with LRU computation cache and memory pools.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        # LRU cache for expensive computations
        self.function_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 512  # Limit cache size

    def _cache_key(self, x: np.ndarray) -> tuple:
        """Create cache key with reduced precision for better hit rates."""
        return tuple(np.round(x, 5))

    def _cached_objective(self, objective_func: Callable, x: np.ndarray) -> float:
        """Cached objective function evaluation with LRU eviction."""
        key = self._cache_key(x)

        if key in self.function_cache:
            self.cache_hits += 1
            return self.function_cache[key]

        self.cache_misses += 1
        result = objective_func(x)

        # LRU cache management
        if len(self.function_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO for performance)
            oldest_key = next(iter(self.function_cache))
            del self.function_cache[oldest_key]

        self.function_cache[key] = result
        return result

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 45, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Memory-optimized Nelder-Mead with computation caching."""
        n = len(x0)

        # Pre-allocate simplex array
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Adaptive step sizing with memory reuse
        step_size = 0.095 * np.abs(x0)
        step_size = np.where(step_size < 0.017, 0.017, step_size)

        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += step_size[i]

        # Pre-allocate arrays for operations
        f_values = np.empty(n + 1)
        indices = np.empty(n + 1, dtype=int)
        best_f_history = []

        # Initial evaluation with caching
        for i in range(n + 1):
            f_values[i] = self._cached_objective(objective_func, simplex[i])

        for iteration in range(maxiter):
            # In-place sorting for memory efficiency
            indices[:] = np.argsort(f_values)
            simplex[:] = simplex[indices]
            f_values[:] = f_values[indices]

            best_f_history.append(f_values[0])

            # Early convergence checks
            if f_values[-1] - f_values[0] < tolerance:
                break

            if len(best_f_history) >= 10:
                recent_improvement = best_f_history[-10] - best_f_history[-1]
                if recent_improvement < tolerance * 0.06:
                    break

            # Centroid computation with memory reuse
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + self.alpha * (centroid - simplex[-1])
            f_reflected = self._cached_objective(objective_func, reflected)

            if f_values[0] <= f_reflected < f_values[-2]:
                simplex[-1] = reflected
                f_values[-1] = f_reflected
                continue

            if f_reflected < f_values[0]:
                # Try expansion
                expanded = centroid + self.gamma * (reflected - centroid)
                f_expanded = self._cached_objective(objective_func, expanded)

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

            f_contracted = self._cached_objective(objective_func, contracted)

            if f_contracted < min(f_reflected, f_values[-1]):
                simplex[-1] = contracted
                f_values[-1] = f_contracted
                continue

            # Shrink with cached evaluations
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                f_values[i] = self._cached_objective(objective_func, simplex[i])

        # Return best point
        best_idx = np.argmin(f_values)
        return simplex[best_idx], f_values[best_idx]

class MemoryOptimizedCSTInitializer:
    """
    Cache-optimized CST initializer with memory pools and spatial locality optimization.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Enlarged evaluation cache with spatial locality
        self.evaluator_cache = {}
        self.cache_access_pattern = {}  # Track access patterns for optimization

        # Parameter bounds with memory-aligned storage
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for vectorized operations
        self._cache_bounds_arrays()

        # Pre-allocate work arrays for coefficient generation
        self._preallocate_coefficient_arrays()

    def _preallocate_coefficient_arrays(self):
        """Pre-allocate arrays for coefficient generation to reduce allocations."""
        max_batch_size = 128
        self.work_upper_coeffs = np.empty((max_batch_size, 5))
        self.work_lower_coeffs = np.empty((max_batch_size, 5))
        self.work_sobol_points = np.empty((max_batch_size, 10))

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for vectorized operations."""
        self.upper_bounds_cache = np.array([b for b, _ in self.bounds['upper_coeffs']])
        self.upper_ranges_cache = np.array([r - l for l, r in self.bounds['upper_coeffs']])
        self.lower_bounds_cache = np.array([b for b, _ in self.bounds['lower_coeffs']])
        self.lower_ranges_cache = np.array([r - l for l, r in self.bounds['lower_coeffs']])

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Convert coefficient vector to Airfoil object with bounds checking."""
        upper_coeffs = np.clip(coeffs[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(coeffs[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def objective_function(self, coeffs: np.ndarray) -> float:
        """Cache-optimized objective function with access pattern tracking."""
        coeffs_key = tuple(np.round(coeffs, 6))

        # Track access patterns for cache optimization
        if coeffs_key in self.cache_access_pattern:
            self.cache_access_pattern[coeffs_key] += 1
        else:
            self.cache_access_pattern[coeffs_key] = 1

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 17 * (0.06 - max_t)
                if max_t > 0.18:
                    penalty += 17 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            # Cache management with access pattern consideration
            if len(self.evaluator_cache) > 1024:  # Enlarged cache
                # Remove least frequently accessed entries
                sorted_keys = sorted(self.cache_access_pattern.keys(),
                                   key=lambda k: self.cache_access_pattern.get(k, 0))
                for key_to_remove in sorted_keys[:128]:  # Remove 128 entries at once
                    self.evaluator_cache.pop(key_to_remove, None)
                    self.cache_access_pattern.pop(key_to_remove, None)

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Vectorized conversion with pre-allocated arrays for memory efficiency."""
        n_points = sobol_points.shape[0]

        # Use pre-allocated arrays if possible
        if n_points <= self.work_upper_coeffs.shape[0]:
            upper_coeffs_all = self.work_upper_coeffs[:n_points]
            lower_coeffs_all = self.work_lower_coeffs[:n_points]
        else:
            upper_coeffs_all = np.empty((n_points, 5))
            lower_coeffs_all = np.empty((n_points, 5))

        # Vectorized computation with memory reuse
        upper_coeffs_all[:] = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all[:] = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 42) -> List[Airfoil]:
        """Memory-optimized Sobol population generation."""
        sobol = MemoryOptimizedSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_apex_gaussian_population(self, size: int = 32) -> List[Airfoil]:
        """Memory-optimized Gaussian population with vectorized generation."""
        population = []
        n_coeffs = 5

        # Pre-allocate arrays for all strategies
        strategy_sizes = [size // 4, size // 4, size // 4, size - 3 * (size // 4)]

        # Strategy 1: Standard distribution with vectorized generation
        all_upper = self.rng.normal(0.175, 0.068, (strategy_sizes[0], n_coeffs))
        all_lower = self.rng.normal(-0.102, 0.050, (strategy_sizes[0], n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy_sizes[0]):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased
        upper_biased = self.rng.normal(0.200, 0.075, (strategy_sizes[1], n_coeffs))
        lower_biased = self.rng.normal(-0.072, 0.040, (strategy_sizes[1], n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy_sizes[1]):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized
        thick_upper = self.rng.normal(0.225, 0.055, (strategy_sizes[2], n_coeffs))
        thick_lower = self.rng.normal(-0.138, 0.050, (strategy_sizes[2], n_coeffs))
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy_sizes[2]):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with enhanced asymmetry
        sym_upper = self.rng.normal(0.140, 0.040, (strategy_sizes[3], n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy_sizes[3]):
            upper_list = sym_upper[i].tolist()
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """Memory-optimized candidate optimization with cached evaluations."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Optimized batch processing
        batch_size = 16
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = CachedNelderMead()

        for airfoil in top_candidates:
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=40,
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                optimized.append(airfoil)

        return optimized

    def apex_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 26) -> List[Airfoil]:
        """Memory-optimized adaptive refinement with vectorized perturbations."""
        refined = []
        sobol = MemoryOptimizedSobol(dimension=10)

        for airfoil in good_airfoils[:8]:
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Adaptive scale
            base_scale = 0.100
            scale = base_scale * (1.0 + 0.15 * good_airfoils.index(airfoil))

            # Vectorized perturbation computation
            perturbations_upper = (local_points[:, :5] - 0.5) * scale
            perturbations_lower = (local_points[:, 5:] - 0.5) * scale

            for i, point in enumerate(local_points):
                new_upper = center_upper + perturbations_upper[i] * np.abs(center_upper)
                new_lower = center_lower + perturbations_lower[i] * np.abs(center_lower)

                new_upper = np.clip(new_upper, 0.01, 0.3)
                new_lower = np.clip(new_lower, -0.3, -0.01)

                refined.append(Airfoil(
                    upper_coeffs=new_upper.tolist(),
                    lower_coeffs=new_lower.tolist(),
                    zte=0.0
                ))

        return refined

def generate_population() -> List[Airfoil]:
    """Generate memory-optimized population with cache hierarchy."""
    print("Generating cache-optimized population with memory pools...")

    initializer = MemoryOptimizedCSTInitializer(seed=999999)

    # Stage 1: Generate base population
    sobol_pop = initializer.generate_sobol_population(42)
    gaussian_pop = initializer.generate_apex_gaussian_population(32)

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Adaptive refinement
    refined_pop = initializer.apex_adaptive_refinement(optimized_pop[:8], refinement_size=26)

    # Combine with efficient deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Memory-efficient deduplication
    unique_population = []
    tolerance = 6e-5

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using cache-optimized approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Cache-optimized population evaluation."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing
    batch_size = 16
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main cache-optimized initialization."""
    print("Generating cache-optimized population with memory hierarchy...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:35]):
        filename = f"results/cache_optimized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['cache_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "13a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "cache_rank": i
        })

    # Cache performance metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} cache-optimized airfoils")
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