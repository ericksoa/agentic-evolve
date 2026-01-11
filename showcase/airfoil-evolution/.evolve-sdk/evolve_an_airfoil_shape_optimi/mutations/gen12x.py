#!/usr/bin/env python3
"""
Gen 12x: Apex Hybrid Vectorized Crossover
Ultimate performance hybrid combining the best from three top parents:
- SIMD-vectorized Sobol sequences from gen9c for maximum computational efficiency
- Advanced multi-strategy Gaussian population from gen5x for superior diversity
- Ultra-cached optimization pipeline from gen9x for robustness and performance
- NEW: Hybrid vectorization strategy combining batch and full vectorization
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

class HybridVectorizedSobol:
    """
    Hybrid vectorized Sobol sequence generator combining:
    - Full SIMD vectorization from gen9c for maximum efficiency
    - Extended caching from gen5x for ultra-fast lookups (2048 points)
    - Adaptive batch processing from gen9x for optimal performance across different sizes
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        self.direction_numbers = self._get_direction_numbers(dimension)
        # Ultra-large cache from gen5x for maximum performance
        self.gray_cache = self._precompute_gray_codes(2048)
        # Enhanced XOR cache from gen5x
        self.xor_cache = self._precompute_xor_operations()
        # Pre-compute bit masks for vectorized operations from gen9c
        self.bit_masks = np.uint32(1) << np.arange(32, dtype=np.uint32)

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation from gen5x."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions combining gen5x robustness
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes with expanded cache size from gen5x."""
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

    def generate_points(self, n: int) -> np.ndarray:
        """
        Hybrid point generation combining vectorization from gen9c with caching from gen5x.
        Uses adaptive strategy: full vectorization for small batches, cached approach for larger.
        """
        # Pre-allocate output array for cache efficiency (from gen9c)
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # Use cached Gray codes from gen5x approach when available
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Vectorized Gray code generation for all points at once (from gen9c)
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # Adaptive approach: use gen9c vectorization for small batches, gen5x unrolling for larger
        if n <= 128:  # Small batch - use full vectorization from gen9c
            # Completely vectorized point generation for each dimension
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]
                # Initialize point values for this dimension
                point_vals = np.zeros(n, dtype=np.uint32)

                # Vectorized bit checking and XOR operations for all 32 bits (from gen9c)
                for bit_pos in range(32):
                    # Check if bit is set for ALL points at once
                    mask = (gray_codes & self.bit_masks[bit_pos]) != 0
                    # XOR direction number where mask is True
                    point_vals[mask] ^= dir_nums[bit_pos]

                # Convert to [0,1] range with vectorized division
                points[:, d] = point_vals.astype(np.float64) / np.float64(1 << 32)
        else:
            # Larger batch - use optimized unrolled approach from gen5x
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]

                for i in range(n):
                    gray = gray_codes[i]
                    point_val = 0

                    # Extended unrolled inner loop from gen5x for maximum performance
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

                    # Handle remaining bits with gen5x optimized approach
                    remaining_gray = gray >> 8
                    bit_pos = 8
                    while remaining_gray and bit_pos < 32:
                        if remaining_gray & 1:
                            point_val ^= dir_nums[bit_pos]
                        remaining_gray >>= 1
                        bit_pos += 1

                    points[i, d] = point_val / (1 << 32)

        return points

class HybridNelderMead:
    """
    Enhanced Nelder-Mead optimizer combining improvements from gen9x and gen5x.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 42, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Hybrid Nelder-Mead optimization with enhanced convergence from both parents."""
        n = len(x0)

        # Enhanced simplex initialization from gen9x
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Enhanced adaptive step sizing from gen9x with gen5x robustness
        step_size = 0.097 * np.abs(x0)  # Optimal size between both parents
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

            # Track history for enhanced convergence
            best_f_history.append(f_values[0])

            # Enhanced convergence criteria
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Advanced stagnation check combining both approaches (9 iterations from gen9x)
            if len(best_f_history) >= 10:
                recent_improvement = best_f_history[-10] - best_f_history[-1]
                if recent_improvement < tolerance * 0.07:  # Optimized threshold
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

class HybridCSTInitializer:
    """
    Ultimate hybrid CST initializer combining the best from all three parents:
    - Vectorized Sobol generation from gen9c for efficiency
    - Advanced multi-strategy Gaussian population from gen5x for diversity
    - Ultra-cached optimization pipeline from gen9x for robustness
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen5x/gen9x
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
        """Enhanced objective function with advanced penalty system from gen9x."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Advanced penalty system from gen9x with optimized values
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 16.5 * (0.06 - max_t)  # Average of best performing parents
                if max_t > 0.18:
                    penalty += 16.5 * (max_t - 0.18)

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
        """Generate enhanced Sobol population using hybrid vectorization."""
        sobol = HybridVectorizedSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 30) -> List[Airfoil]:
        """
        Generate hybrid Gaussian population using enhanced multi-strategy approach from gen5x
        with parameters optimized from all three parents.
        """
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution (optimized parameters from gen5x/gen9x)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.172, 0.066, (strategy1_size, n_coeffs))  # Optimized mean/std
        all_lower = self.rng.normal(-0.103, 0.049, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased (combining gen5x and gen9x approaches)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.197, 0.073, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.074, 0.039, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized airfoils (enhanced from gen5x)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.223, 0.053, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.140, 0.048, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.13, 0.05, 0.3)  # Optimized scaling
        thick_lower = np.clip(thick_lower * 1.13, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with enhanced bias (combining all parent approaches)
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.137, 0.039, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor optimized from all parents
            lower_list = [-c * self.rng.uniform(0.935, 1.075) for c in upper_list]
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

        # Enhanced batch evaluation (optimized size from all parents)
        batch_size = 11  # Optimal size between parent approaches
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select enhanced number of top candidates
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = HybridNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Enhanced optimization parameters combining all parents
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=38,  # Optimized iterations
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Hybrid adaptive refinement combining vectorized generation with enhanced strategies."""
        refined = []
        sobol = HybridVectorizedSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Enhanced to top 8 from gen9x
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations with hybrid vectorized Sobol
            local_points = sobol.generate_points(refinement_size // 8)

            # Enhanced adaptive scale optimized from all parents
            base_scale = 0.098  # Optimal scale between approaches
            scale = base_scale * (1.0 + 0.13 * good_airfoils.index(airfoil))

            for point in local_points:
                # Enhanced perturbation strategy
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
    print("Generating hybrid vectorized population with advanced multi-strategy optimization...")

    initializer = HybridCSTInitializer(seed=777777)  # Ultimate hybrid seed

    # Stage 1: Generate enhanced base population with optimal sizing
    sobol_pop = initializer.generate_sobol_population(40)      # Optimized from all parents
    gaussian_pop = initializer.generate_hybrid_gaussian_population(30)  # Hybrid multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=15)

    # Stage 3: Hybrid adaptive refinement around best solutions
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with optimal tolerance from all parents
    unique_population = []
    tolerance = 7e-5  # Optimized tolerance

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using hybrid vectorized approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching from all parents."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing combining all approaches
    batch_size = 13  # Optimal size between all parents
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid vectorized initialization."""
    print("Generating hybrid vectorized population with apex optimizations from all parents...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:33]):  # Save top 33 (between all parents)
        filename = f"results/hybrid_vectorized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "12x"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Enhanced diversity metrics combining all parent approaches
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} hybrid vectorized airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 13 mean: {np.mean(valid_fitnesses[:13]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()