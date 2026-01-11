#!/usr/bin/env python3
"""
Gen 10x: Supreme Crossover Hybrid - Performance Elite
Ultimate fusion of the three highest-performing parents (gen9x, gen9c, gen5x):

From gen9x (113.9914):
- HyperVectorizedSobol with adaptive batch processing
- Advanced 4-strategy Gaussian population with optimal parameters
- Extended convergence criteria (10-iteration stagnation check)
- Ultra-sophisticated adaptive refinement on top 8 airfoils

From gen9c (113.6363):
- UltraFastSobol complete SIMD vectorization approach
- Clean efficient batch processing implementation
- Simplified but effective optimization pipeline

From gen5x (113.4354):
- ApexSobol ultra-large cache optimization (2048 points)
- Robust unrolled bit operations for maximum speed
- Enhanced penalty system with advanced thickness constraints
- Proven multi-strategy parameter combinations

New Enhancements:
- Hybrid SIMD vectorization combining gen9x adaptive batching with gen9c complete vectorization
- Supreme multi-strategy Gaussian with the best parameters from all three parents
- Enhanced convergence criteria combining all parent approaches
- Ultra-optimized caching system with maximum performance
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

class SupremeHybridSobol:
    """
    Supreme Sobol sequence generator combining the best of all three parents:
    - gen9x: HyperVectorizedSobol with adaptive batch processing
    - gen9c: UltraFastSobol complete SIMD vectorization
    - gen5x: ApexSobol ultra-large cache optimization

    Creates the ultimate performance hybrid with maximum optimization.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Ultra-large cache from gen5x/gen9x
        self.direction_numbers = self._get_direction_numbers(dimension)
        # Maximum cache from gen9x/gen5x
        self.gray_cache = self._precompute_gray_codes(2048)
        # Enhanced XOR cache from all parents
        self.xor_cache = self._precompute_xor_operations()
        # Direction matrix for vectorization from gen9x
        self.direction_matrix = self._prepare_direction_matrix()
        # Bit mask cache from gen5x
        self.bit_mask_cache = self._precompute_bit_masks()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers from gen9x/gen5x."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions from gen9x/gen5x
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Ultra-large Gray code cache from gen9x/gen5x."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Enhanced XOR operation cache from all parents."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_masks(self) -> np.ndarray:
        """Additional bit mask cache from gen5x."""
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def _prepare_direction_matrix(self) -> np.ndarray:
        """Direction matrix for vectorization from gen9x."""
        bit_positions = np.arange(32, dtype=np.uint32)
        self.bit_masks = 1 << bit_positions
        return self.direction_numbers

    def generate_points(self, n: int) -> np.ndarray:
        """
        Supreme hybrid point generation combining:
        - gen9x: Adaptive batch processing with vectorization
        - gen9c: Complete SIMD vectorization
        - gen5x: Ultra-optimized unrolled operations
        """
        points = np.zeros((n, self.dimension))

        # Use ultra-large cache from gen9x/gen5x
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Vectorized generation from gen9c
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # Hybrid approach combining all three strategies
        if n <= 32:  # Small batch - use gen9c complete vectorization
            # Pre-compute all bit masks for vectorized operations from gen9c
            bit_masks = np.uint32(1) << np.arange(32, dtype=np.uint32)

            # Completely vectorized point generation for each dimension
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]
                point_vals = np.zeros(n, dtype=np.uint32)

                # Vectorized bit checking and XOR operations from gen9c
                for bit_pos in range(32):
                    mask = (gray_codes & bit_masks[bit_pos]) != 0
                    point_vals[mask] ^= dir_nums[bit_pos]

                points[:, d] = point_vals.astype(np.float64) / np.float64(1 << 32)

        elif n <= 128:  # Medium batch - use gen9x adaptive vectorization
            # Create bit mask matrix from gen9x approach
            gray_expanded = gray_codes[:, np.newaxis]
            bit_masks_expanded = self.bit_masks[np.newaxis, :]
            bit_tests = (gray_expanded & bit_masks_expanded) != 0

            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d, :]
                masked_dirs = np.where(bit_tests, dir_nums[np.newaxis, :], 0)
                point_vals = np.bitwise_xor.reduce(masked_dirs, axis=1)
                points[:, d] = point_vals.astype(np.float64) / (1 << 32)

        else:  # Large batch - use gen5x/gen9x ultra-optimized unrolled operations
            for d in range(self.dimension):
                dir_nums = self.direction_numbers[d]

                for i in range(n):
                    gray = gray_codes[i]
                    point_val = 0

                    # Ultra-extended unrolled inner loop from gen9x/gen5x
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

                    # Handle remaining bits with optimized approach from gen9x/gen5x
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
    Supreme Nelder-Mead optimizer combining all three parents:
    - gen9x: Extended convergence criteria (10-iteration stagnation)
    - gen9c: Solid 6-iteration convergence check
    - gen5x: Robust implementation with enhanced parameters
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 45, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Supreme Nelder-Mead with ultimate convergence criteria from all parents."""
        n = len(x0)

        # Enhanced simplex initialization from gen9x
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Supreme adaptive step sizing combining all parents
        step_size = 0.095 * np.abs(x0)  # Optimal from gen9x
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

            # Track history for supreme convergence from gen9x
            best_f_history.append(f_values[0])

            # Enhanced convergence criteria
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Supreme stagnation check combining gen9x (10 iter) with gen9c/gen5x robustness
            if len(best_f_history) >= 11:  # Extended from gen9x
                recent_improvement = best_f_history[-11] - best_f_history[-1]
                if recent_improvement < tolerance * 0.05:  # Stricter than all parents
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
    Supreme CST initializer fusing the best from all three high-performing parents:
    - gen9x: Advanced 4-strategy Gaussian with optimal parameters
    - gen9c: Clean efficient implementation and batch processing
    - gen5x: Enhanced penalty system and robust parameter bounds
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Supreme parameter bounds combining insights from all parents
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for maximum vectorized speed from all parents
        self._cache_bounds_arrays()

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for vectorized operations."""
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
        """Supreme objective function with enhanced penalty system from all parents."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Supreme penalty system combining all parents with optimization
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 18 * (0.06 - max_t)  # Enhanced from all parents
                if max_t > 0.18:
                    penalty += 18 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points using cached vectorized operations from all parents."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 44) -> List[Airfoil]:
        """Generate supreme Sobol population with hybrid optimization."""
        sobol = SupremeHybridSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_supreme_gaussian_population(self, size: int = 36) -> List[Airfoil]:
        """
        Generate supreme Gaussian population combining optimal parameters from all parents:
        - gen9x: Advanced 4-strategy approach with optimal parameters
        - gen9c: Effective strategy parameters
        - gen5x: Proven parameter combinations
        """
        population = []
        n_coeffs = 5

        # Strategy 1: Supreme standard distribution (best from gen9x + gen5x)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.175, 0.068, (strategy1_size, n_coeffs))  # Optimal from gen9x
        all_lower = self.rng.normal(-0.102, 0.050, (strategy1_size, n_coeffs))  # Optimal from gen9x
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: Supreme high-performance biased (best from all parents)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.200, 0.075, (strategy2_size, n_coeffs))  # From gen9x
        lower_biased = self.rng.normal(-0.072, 0.040, (strategy2_size, n_coeffs))  # From gen9x
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Supreme thickness-optimized (enhanced from all parents)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.055, (strategy3_size, n_coeffs))  # From gen9x
        thick_lower = self.rng.normal(-0.138, 0.050, (strategy3_size, n_coeffs))  # From gen9x
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)  # Enhanced from gen9x
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Supreme near-symmetric (best asymmetry from gen9x)
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.140, 0.040, (strategy4_size, n_coeffs))  # From gen9x
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Supreme asymmetry factor from gen9x
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 18) -> List[Airfoil]:
        """Supreme optimization combining all parent approaches."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Supreme batch evaluation combining gen9c efficiency with gen9x size
        batch_size = 14  # Optimal from gen9c/gen5x
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for airfoil in batch:
                result = evaluate_airfoil(airfoil, req, use_xfoil=False)
                fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
                evaluated.append((airfoil, fitness))

        # Sort and select supreme number of top candidates
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = SupremeNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Supreme optimization parameters from gen9x
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=45,  # From gen9x
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def supreme_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 32) -> List[Airfoil]:
        """Supreme adaptive refinement combining the best from all parents."""
        refined = []
        sobol = SupremeHybridSobol(dimension=10)

        # Work on top 8 airfoils from gen9x approach
        for airfoil in good_airfoils[:8]:
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations using supreme hybrid Sobol
            local_points = sobol.generate_points(refinement_size // 8)

            # Supreme adaptive scale from gen9x with enhancements
            base_scale = 0.100  # From gen9x
            scale = base_scale * (1.0 + 0.15 * good_airfoils.index(airfoil))  # From gen9x

            for point in local_points:
                # Supreme perturbation strategy from gen9x
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
    """Generate supreme hybrid population combining all three elite parents."""
    print("Generating supreme hybrid population with ultimate optimization...")

    initializer = SupremeCSTInitializer(seed=1010101)  # Supreme unique seed

    # Stage 1: Generate supreme base population with optimal sizing
    sobol_pop = initializer.generate_sobol_population(44)      # Enhanced from gen9x
    gaussian_pop = initializer.generate_supreme_gaussian_population(36)  # Supreme multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with supreme selection
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=18)

    # Stage 3: Supreme adaptive refinement around best solutions
    refined_pop = initializer.supreme_adaptive_refinement(optimized_pop[:8], refinement_size=32)

    # Combine all populations with supreme deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Supreme deduplication with optimal tolerance from gen9x
    unique_population = []
    tolerance = 6e-5  # From gen9x (tightest)

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using supreme hybrid approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Supreme population evaluation with optimized batching from all parents."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Supreme batch processing combining all parent approaches
    batch_size = 16  # Optimal from gen9x
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main supreme hybrid initialization."""
    print("Generating supreme hybrid population with ultimate performance optimization...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:40]):  # Save top 40 (supreme)
        filename = f"results/supreme_hybrid_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['supreme_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "10x"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "supreme_rank": i
        })

    # Supreme diversity metrics combining all parent approaches
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} supreme hybrid airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 20 mean: {np.mean(valid_fitnesses[:20]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()