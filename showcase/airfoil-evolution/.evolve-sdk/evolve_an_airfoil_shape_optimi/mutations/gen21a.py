#!/usr/bin/env python3
"""
Gen 21a: Vectorized Array Processing Mutation
Performance mutation: Replace scalar operations with vectorized NumPy operations
throughout PSO to leverage SIMD instructions and eliminate Python loops.
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

class ApexSobol:
    """
    Apex Sobol sequence generator combining all optimization techniques:
    - Ultra-large cache from gen4x (2048 points)
    - Unrolled bit operations from gen3a
    - Advanced XOR optimizations from gen4x
    - Robust direction number computation from gen3x
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Max cache size from gen4x for ultimate performance
        self.direction_numbers = self._get_direction_numbers(dimension)
        # Expanded cache from gen4x
        self.gray_cache = self._precompute_gray_codes(2048)
        # Enhanced XOR cache from gen4x
        self.xor_cache = self._precompute_xor_operations()
        # NEW: Additional bit mask cache for even faster operations
        self.bit_mask_cache = self._precompute_bit_masks()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation from gen4x."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions combining gen4x robustness with gen3x approach
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Precompute Gray codes with expanded cache size from gen4x."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Enhanced XOR operation cache from gen4x."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_masks(self) -> np.ndarray:
        """Additional bit mask cache for maximum performance."""
        # Precompute common bit patterns for ultra-fast lookup
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with apex optimization combining all techniques."""
        points = np.zeros((n, self.dimension))

        # Use cached Gray codes from gen4x approach
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Enhanced point generation combining gen4x unrolling with gen3a optimization
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Extended unrolled inner loop from gen3a/gen4x for maximum performance
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

                # Handle remaining bits with gen4x optimized approach
                remaining_gray = gray >> 8
                bit_pos = 8
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

        return points

class ApexPSO:
    """
    MUTATION: Vectorized Array Processing PSO
    Eliminates scalar operations and Python loops with pure NumPy vectorization
    to leverage SIMD instructions for maximum computational efficiency.
    """

    def __init__(self, swarm_size=8, w=0.7, c1=1.4, c2=1.4):
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

        # MUTATION: Pre-allocate arrays for vectorized operations
        self._initialize_vectorized_arrays()

    def _initialize_vectorized_arrays(self):
        """MUTATION: Initialize arrays for vectorized PSO operations."""
        # Pre-allocate random number arrays for entire optimization run
        max_iterations = 40
        total_randoms_needed = max_iterations * self.swarm_size * 2

        np.random.seed(42)  # Reproducible randomness
        self.random_pool = np.random.random((total_randoms_needed, 2))
        self.random_idx = 0

        # Pre-allocate arrays to avoid memory allocation during optimization
        self.velocity_updates = np.zeros((self.swarm_size, 10))  # Assuming 10D
        self.position_diffs = np.zeros((self.swarm_size, 10))
        self.cognitive_components = np.zeros((self.swarm_size, 10))
        self.social_components = np.zeros((self.swarm_size, 10))

    def _vectorized_random_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """MUTATION: Get vectorized random numbers in batch."""
        end_idx = self.random_idx + batch_size
        if end_idx > len(self.random_pool):
            # Wrap around if we run out of pre-computed randoms
            self.random_idx = 0
            end_idx = batch_size

        randoms = self.random_pool[self.random_idx:end_idx]
        self.random_idx = end_idx

        return randoms[:, 0], randoms[:, 1]

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """MUTATION: Fully vectorized PSO for maximum SIMD utilization."""
        n_dims = len(x0)

        # MUTATION: Initialize all arrays at once using vectorized operations
        positions = np.zeros((self.swarm_size, n_dims))
        positions[0] = x0  # Best guess as first particle

        # Vectorized initialization of remaining particles
        perturbations = np.random.normal(0, 0.05, (self.swarm_size - 1, n_dims))
        positions[1:] = x0 + perturbations

        # Vectorized velocity initialization
        velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, n_dims))

        # MUTATION: Vectorized evaluation (though objective_func calls are still scalar)
        fitness_values = np.array([objective_func(pos) for pos in positions])

        # Initialize personal best arrays
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness_values.copy()

        # Find global best using vectorized operations
        global_best_idx = np.argmin(fitness_values)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness_values[global_best_idx]

        no_improvement_count = 0

        for iteration in range(maxiter):
            # MUTATION: Vectorized adaptive inertia weight
            adaptive_w = self.w * (0.95 + 0.1 * np.exp(-iteration / 20.0))

            # MUTATION: Vectorized random number generation for all particles
            r1_batch, r2_batch = self._vectorized_random_batch(self.swarm_size)

            # MUTATION: Fully vectorized velocity updates for entire swarm
            # Broadcast operations for maximum SIMD utilization
            self.position_diffs = personal_best_positions - positions
            self.cognitive_components = (self.c1 * r1_batch[:, np.newaxis]) * self.position_diffs

            self.position_diffs = global_best_position - positions  # Reuse array
            self.social_components = (self.c2 * r2_batch[:, np.newaxis]) * self.position_diffs

            # Vectorized velocity update for entire swarm at once
            velocities = adaptive_w * velocities + self.cognitive_components + self.social_components

            # Vectorized position update for entire swarm
            positions += velocities

            # MUTATION: Vectorized fitness evaluation and best update
            new_fitness_values = np.array([objective_func(pos) for pos in positions])

            # Vectorized personal best updates using boolean indexing
            improvement_mask = new_fitness_values < personal_best_fitness
            personal_best_positions[improvement_mask] = positions[improvement_mask]
            personal_best_fitness[improvement_mask] = new_fitness_values[improvement_mask]

            # Vectorized global best update
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < global_best_fitness:
                global_best_position = personal_best_positions[current_best_idx].copy()
                global_best_fitness = personal_best_fitness[current_best_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Vectorized convergence check
            if no_improvement_count > 15:
                break

            # Vectorized tolerance check
            fitness_range = np.ptp(personal_best_fitness)  # Peak-to-peak (max - min)
            if fitness_range < tolerance:
                break

        return global_best_position, global_best_fitness

class ApexCSTInitializer:
    """
    Apex CST initializer combining all the best features:
    - Ultra-cached Sobol from gen4x/gen3a
    - Advanced multi-strategy Gaussian from gen4x
    - Clean pipeline structure from gen3x
    - Vectorized bounds caching from gen3a
    - Enhanced penalty systems from gen4x
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen4x
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays from gen3a optimization
        self._cache_bounds_arrays()

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays from gen3a for maximum vectorized speed."""
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
        """Enhanced objective function with advanced penalty system from gen4x."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Advanced penalty system from gen4x
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 16 * (0.06 - max_t)  # Slightly increased from gen4x
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
        """Convert Sobol points using cached vectorized operations from gen3a."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed from gen3a
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 38) -> List[Airfoil]:
        """Generate enhanced Sobol population using apex optimization."""
        sobol = ApexSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_apex_gaussian_population(self, size: int = 28) -> List[Airfoil]:
        """Generate apex Gaussian population combining all strategies from gen4x."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution from gen4x
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.17, 0.065, (strategy1_size, n_coeffs))  # Optimized params
        all_lower = self.rng.normal(-0.105, 0.048, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased from gen4x (enhanced)
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

        # Strategy 3: Thickness-optimized from gen4x (enhanced)
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

        # Strategy 4: Near-symmetric with enhanced bias from gen4x
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.135, 0.038, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor
            lower_list = [-c * self.rng.uniform(0.94, 1.07) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 14) -> List[Airfoil]:
        """MUTATION: Vectorized array processing PSO replacing Nelder-Mead."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Batch evaluation from gen3a approach
        batch_size = 10
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
        # MUTATION: Use vectorized array processing PSO
        optimizer = ApexPSO(swarm_size=8, w=0.7, c1=1.4, c2=1.4)

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Vectorized PSO with SIMD-optimized operations
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

    def apex_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 22) -> List[Airfoil]:
        """Apex adaptive refinement combining gen4x and gen3a approaches."""
        refined = []
        sobol = ApexSobol(dimension=10)

        for airfoil in good_airfoils[:7]:  # Enhanced to top 7
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 7)

            # Enhanced adaptive scale from gen4x
            base_scale = 0.095
            scale = base_scale * (1.0 + 0.12 * good_airfoils.index(airfoil))

            for point in local_points:
                # Enhanced perturbation strategy from gen4x
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
    """Generate vectorized array processing optimized hybrid population."""
    print("Generating vectorized array processing optimized apex hybrid population...")

    initializer = ApexCSTInitializer(seed=525555)  # Unique apex seed

    # Stage 1: Generate apex base population with optimal sizing from all parents
    sobol_pop = initializer.generate_sobol_population(38)      # Enhanced from gen4x
    gaussian_pop = initializer.generate_apex_gaussian_population(28)  # Apex multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with vectorized PSO
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=14)

    # Stage 3: Apex adaptive refinement around best solutions
    refined_pop = initializer.apex_adaptive_refinement(optimized_pop[:7], refinement_size=22)

    # Combine all populations with intelligent deduplication from gen4x
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with tighter tolerance
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

    print(f"Generated {len(unique_population)} unique airfoils using vectorized array processing approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching from gen3a."""
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
    """Main vectorized array processing optimized hybrid initialization."""
    print("Generating vectorized population with SIMD-optimized PSO operations...")

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
        filename = f"results/vectorized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['vectorized_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "21a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "vectorized_rank": i
        })

    # Enhanced diversity metrics from gen4x
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} vectorized array processing airfoils")
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