#!/usr/bin/env python3
"""
Gen 6 Crossover: Ultimate Performance Hybrid
Combines the best features from three high-performing parents:
- gen5x: Advanced penalty system, enhanced population sizing, stricter convergence
- gen5a: Aggressive 16+8 bit loop unrolling for maximum Sobol performance
- gen4x: Solid caching foundation and proven multi-strategy Gaussian approach

Key hybrid features:
- Ultra-aggressive loop unrolling (from gen5a) with enhanced algorithms (from gen5x)
- Advanced penalty system and population sizing (from gen5x)
- Proven caching foundation (from gen4x/gen5a)
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

class HybridUltraFastSobol:
    """
    Hybrid Sobol generator combining:
    - gen5a's aggressive 16+8 bit loop unrolling for maximum performance
    - gen5x's ultra-large cache (2048) and enhanced XOR optimizations
    - gen4x's solid foundation and robust direction number computation
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Enhanced direction numbers from gen5x/gen4x
        self.direction_numbers = self._get_direction_numbers(dimension)
        # Ultra-large cache from gen5x for maximum performance
        self.gray_cache = self._precompute_gray_codes(2048)
        # Enhanced XOR cache from gen5x
        self.xor_cache = self._precompute_xor_operations()
        # Additional bit mask cache from gen5x
        self.bit_mask_cache = self._precompute_bit_masks()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation from gen5x."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions from gen5x
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Ultra-large Gray code cache from gen5x."""
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

    def generate_points(self, n: int) -> np.ndarray:
        """
        Hybrid Sobol point generation combining:
        - gen5a's aggressive 16+8 bit loop unrolling for maximum performance
        - gen5x's ultra-large cache and enhanced optimizations
        """
        points = np.zeros((n, self.dimension))

        # Use ultra-large cached Gray codes from gen5x
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Aggressive loop unrolling from gen5a with gen5x optimizations
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Ultra-aggressive 16-bit unrolling from gen5a (with hex constants for clarity)
                if gray & 0x1:
                    point_val ^= dir_nums[0]
                if gray & 0x2:
                    point_val ^= dir_nums[1]
                if gray & 0x4:
                    point_val ^= dir_nums[2]
                if gray & 0x8:
                    point_val ^= dir_nums[3]
                if gray & 0x10:
                    point_val ^= dir_nums[4]
                if gray & 0x20:
                    point_val ^= dir_nums[5]
                if gray & 0x40:
                    point_val ^= dir_nums[6]
                if gray & 0x80:
                    point_val ^= dir_nums[7]
                if gray & 0x100:
                    point_val ^= dir_nums[8]
                if gray & 0x200:
                    point_val ^= dir_nums[9]
                if gray & 0x400:
                    point_val ^= dir_nums[10]
                if gray & 0x800:
                    point_val ^= dir_nums[11]
                if gray & 0x1000:
                    point_val ^= dir_nums[12]
                if gray & 0x2000:
                    point_val ^= dir_nums[13]
                if gray & 0x4000:
                    point_val ^= dir_nums[14]
                if gray & 0x8000:
                    point_val ^= dir_nums[15]

                # Additional 8-bit unrolling from gen5a for moderate sequences
                remaining_gray = gray >> 16
                if remaining_gray:
                    if remaining_gray & 0x1:
                        point_val ^= dir_nums[16]
                    if remaining_gray & 0x2:
                        point_val ^= dir_nums[17]
                    if remaining_gray & 0x4:
                        point_val ^= dir_nums[18]
                    if remaining_gray & 0x8:
                        point_val ^= dir_nums[19]
                    if remaining_gray & 0x10:
                        point_val ^= dir_nums[20]
                    if remaining_gray & 0x20:
                        point_val ^= dir_nums[21]
                    if remaining_gray & 0x40:
                        point_val ^= dir_nums[22]
                    if remaining_gray & 0x80:
                        point_val ^= dir_nums[23]

                    # Enhanced fallback handling from gen5x for very large sequences
                    final_remaining = remaining_gray >> 8
                    bit_pos = 24
                    while final_remaining and bit_pos < 32:
                        if final_remaining & 1:
                            point_val ^= dir_nums[bit_pos]
                        final_remaining >>= 1
                        bit_pos += 1

                points[i, d] = point_val / (1 << 32)

        return points

class HybridNelderMead:
    """
    Enhanced Nelder-Mead combining:
    - gen5x's advanced convergence with 8-iteration stagnation check
    - gen4x/gen5a's solid optimization foundation
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Hybrid Nelder-Mead with gen5x's advanced convergence criteria."""
        n = len(x0)

        # Enhanced simplex initialization from gen5x
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Adaptive step sizing from gen5x (slightly larger than gen4x)
        step_size = 0.09 * np.abs(x0)
        step_size = np.where(step_size < 0.016, 0.016, step_size)

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

            # Advanced stagnation check from gen5x (8 iterations for stricter convergence)
            if len(best_f_history) >= 8:
                recent_improvement = best_f_history[-8] - best_f_history[-1]
                if recent_improvement < tolerance * 0.08:
                    break

            # Centroid of n best points
            centroid = np.mean(simplex[:-1], axis=0)

            # Standard Nelder-Mead operations
            reflected = centroid + self.alpha * (centroid - simplex[-1])
            f_reflected = objective_func(reflected)

            if f_values[0] <= f_reflected < f_values[-2]:
                simplex[-1] = reflected
                f_values[-1] = f_reflected
                continue

            if f_reflected < f_values[0]:
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
    Hybrid CST initializer combining:
    - gen5x's enhanced penalty system and population sizing
    - gen4x's solid multi-strategy foundation
    - gen5a's vectorized bounds caching
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen5x
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for vectorized operations
        self._cache_bounds_arrays()

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for faster vectorized operations."""
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
        """Enhanced objective function with gen5x's advanced penalty system."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Advanced penalty system from gen5x (stricter than gen4x)
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 16 * (0.06 - max_t)
                if max_t > 0.18:
                    penalty += 16 * (max_t - 0.18)

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

    def generate_sobol_population(self, size: int = 38) -> List[Airfoil]:
        """Generate enhanced Sobol population using hybrid optimization."""
        sobol = HybridUltraFastSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 28) -> List[Airfoil]:
        """
        Hybrid Gaussian population combining:
        - gen5x's enhanced multi-strategy approach with optimized parameters
        - gen4x's proven 4-strategy foundation
        """
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution from gen5x
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.17, 0.065, (strategy1_size, n_coeffs))
        all_lower = self.rng.normal(-0.105, 0.048, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased from gen5x (enhanced)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.195, 0.072, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.075, 0.038, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized from gen5x (enhanced)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.22, 0.052, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.142, 0.047, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.12, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.12, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with enhanced bias from gen5x
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.135, 0.038, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor from gen5x
            lower_list = [-c * self.rng.uniform(0.94, 1.07) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 14) -> List[Airfoil]:
        """Enhanced optimization with gen5x's increased selection."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Batch evaluation for efficiency
        batch_size = 10
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select top candidates (enhanced from gen5x)
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = HybridNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Enhanced optimization parameters from gen5x
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=35,
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 22) -> List[Airfoil]:
        """Hybrid adaptive refinement with gen5x's enhanced sizing."""
        refined = []
        sobol = HybridUltraFastSobol(dimension=10)

        # Enhanced to top 7 from gen5x
        for airfoil in good_airfoils[:7]:
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 7)

            # Enhanced adaptive scale from gen5x
            base_scale = 0.095
            scale = base_scale * (1.0 + 0.12 * good_airfoils.index(airfoil))

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
    """Generate hybrid population combining the best from all three parents."""
    print("Generating hybrid ultra-performance multi-stage population...")

    initializer = HybridCSTInitializer(seed=626666)  # Unique hybrid seed

    # Stage 1: Generate hybrid base population with gen5x's optimal sizing
    sobol_pop = initializer.generate_sobol_population(38)      # Enhanced from gen5x
    gaussian_pop = initializer.generate_hybrid_gaussian_population(28)  # Hybrid multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with gen5x's enhanced selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=14)

    # Stage 3: Hybrid adaptive refinement with gen5x's enhanced sizing
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:7], refinement_size=22)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with gen5x's tighter tolerance
    unique_population = []
    tolerance = 8e-5

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using hybrid ultra-performance approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing
    batch_size = 14
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid ultra-performance initialization."""
    print("Generating hybrid population combining ultra-aggressive Sobol, advanced optimization, and enhanced multi-strategy approaches...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:32]):  # Save top 32
        filename = f"results/hybrid_ultra_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "6x"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Enhanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} hybrid ultra-performance airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 12 mean: {np.mean(valid_fitnesses[:12]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()