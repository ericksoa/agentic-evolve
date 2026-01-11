#!/usr/bin/env python3
"""
Gen 5a: Extended Loop Unrolling for Sobol Generation
Performance mutation focusing on aggressive loop unrolling in the critical
Sobol point generation pathway to reduce branch overhead and improve cache locality.
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

class UltraFastSobol:
    """
    Ultra-fast Sobol sequence generator with aggressive loop unrolling
    for maximum performance in the critical path.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Precompute direction numbers (from gen3x/gen0_c)
        self.direction_numbers = self._get_direction_numbers(dimension)
        # Caching optimizations from gen3a
        self.gray_cache = self._precompute_gray_codes(2048)  # Increased cache size
        self.xor_cache = self._precompute_xor_operations()

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

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes for common sequence lengths."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Precompute XOR masks for bit positions to accelerate inner loops."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with aggressive loop unrolling optimization."""
        points = np.zeros((n, self.dimension))

        # Use cached Gray codes if available, otherwise compute on-demand
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Aggressively optimized point generation with extended unrolling
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Extended unrolled inner loop for first 16 bit positions
                # This eliminates branching overhead for the most common cases
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

                # Handle remaining bits with optimized bit shifting
                # Most sequences won't need these higher bits, so this is rarely executed
                remaining_gray = gray >> 16
                if remaining_gray:
                    # Unroll next 8 bits for moderate-sized sequences
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

                    # Final fallback for very large sequences (rare)
                    final_remaining = remaining_gray >> 8
                    bit_pos = 24
                    while final_remaining and bit_pos < 32:
                        if final_remaining & 1:
                            point_val ^= dir_nums[bit_pos]
                        final_remaining >>= 1
                        bit_pos += 1

                points[i, d] = point_val / (1 << 32)

        return points

class OptimizedNelderMead:
    """
    Enhanced Nelder-Mead optimizer from gen3x with additional convergence criteria.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 35, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Enhanced Nelder-Mead optimization with improved convergence."""
        n = len(x0)

        # Initialize simplex with adaptive step sizing
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Create initial simplex with better step size strategy
        step_size = 0.08 * np.abs(x0)  # Slightly larger steps
        step_size = np.where(step_size < 0.015, 0.015, step_size)

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

            # Track best fitness for additional convergence check
            best_f_history.append(f_values[0])

            # Enhanced convergence criteria
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Stagnation check: if no improvement in last 5 iterations
            if len(best_f_history) >= 6:
                recent_improvement = best_f_history[-6] - best_f_history[-1]
                if recent_improvement < tolerance * 0.1:
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
    Ultimate hybrid CST initializer combining best features from all parents.
    Integrates cached Sobol (gen3a), multi-stage pipeline (gen3x), and clean adaptive sampling (gen0_c).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds combining insights from all parents
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for vectorized operations (from gen3a)
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
        """Enhanced objective function with improved caching."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Enhanced penalty system
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 15 * (0.06 - max_t)  # Increased penalty
                if max_t > 0.18:
                    penalty += 15 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points to CST parameters using cached vectorized operations."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed (from gen3a)
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 35) -> List[Airfoil]:
        """Generate enhanced Sobol-based quasi-random population."""
        sobol = UltraFastSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_diverse_gaussian_population(self, size: int = 25) -> List[Airfoil]:
        """Generate diverse Gaussian population using enhanced multi-strategy approach."""
        population = []
        n_coeffs = 5

        # Strategy 1: Standard distribution (enhanced)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.16, 0.06, (strategy1_size, n_coeffs))  # Slightly adjusted
        all_lower = self.rng.normal(-0.11, 0.045, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.19, 0.07, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.08, 0.035, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized airfoils
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.21, 0.05, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.14, 0.045, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.1, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.1, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with slight bias
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.13, 0.035, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Add slight asymmetry for better performance
            lower_list = [-c * self.rng.uniform(0.95, 1.05) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 12) -> List[Airfoil]:
        """Enhanced optimization of promising candidates using batch processing."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Batch evaluation for efficiency (from gen0_c approach)
        batch_size = 8
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select top candidates
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = OptimizedNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Optimize with enhanced parameters
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=30,
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 18) -> List[Airfoil]:
        """Enhanced adaptive refinement combining strategies from gen0_c and gen3a."""
        refined = []
        sobol = UltraFastSobol(dimension=10)

        for airfoil in good_airfoils[:6]:  # Increased to top 6
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations with variable scale
            local_points = sobol.generate_points(refinement_size // 6)

            # Adaptive scale based on airfoil performance rank
            base_scale = 0.09
            scale = base_scale * (1.0 + 0.1 * good_airfoils.index(airfoil))

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
    """Generate ultimate hybrid population combining all three strategies."""
    print("Generating ultra-optimized hybrid multi-stage population...")

    initializer = HybridCSTInitializer(seed=424444)  # Unique seed

    # Stage 1: Generate diverse base population with optimal sizing
    sobol_pop = initializer.generate_sobol_population(35)      # Cached Sobol quasi-random
    gaussian_pop = initializer.generate_diverse_gaussian_population(25)  # Enhanced multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=12)

    # Stage 3: Enhanced adaptive refinement around best solutions
    refined_pop = initializer.adaptive_refinement(optimized_pop[:6], refinement_size=18)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Remove near-duplicates to maintain diversity
    unique_population = []
    tolerance = 1e-4

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using ultra-optimized hybrid approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing from gen0_c
    batch_size = 12
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main ultra-optimized hybrid initialization."""
    print("Generating ultimate hybrid population combining cached Sobol, optimization, and multi-strategy approaches...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:30]):  # Save top 30
        filename = f"results/ultra_hybrid_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "5a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Enhanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} ultra-hybrid airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 10 mean: {np.mean(valid_fitnesses[:10]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()