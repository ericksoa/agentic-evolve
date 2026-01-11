#!/usr/bin/env python3
"""
Gen 18a: Cache-Optimized Apex Optimizer
Performance mutation: Precompute bit operation lookup tables and optimize memory access
patterns in Sobol sequence generation for improved cache performance.
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
    Cache-optimized Apex Sobol sequence generator:
    - Ultra-large cache from gen4x (2048 points)
    - Unrolled bit operations from gen3a
    - Advanced XOR optimizations from gen4x
    - Robust direction number computation from gen3x
    - NEW: Precomputed bit operation lookup tables for maximum cache efficiency
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
        # Additional bit mask cache for even faster operations
        self.bit_mask_cache = self._precompute_bit_masks()
        # NEW: Precomputed bit operation lookup tables
        self.bit_op_lookup = self._precompute_bit_operation_lookup()
        # NEW: Precomputed direction number combinations for common patterns
        self.dir_combo_cache = self._precompute_direction_combinations()

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

    def _precompute_bit_operation_lookup(self) -> np.ndarray:
        """NEW: Precompute lookup table for all 8-bit combinations and their XOR patterns."""
        # Create lookup table for all 256 possible byte values
        lookup = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                lookup[i, bit] = bool(i & (1 << bit))
        return lookup

    def _precompute_direction_combinations(self) -> Dict[int, np.ndarray]:
        """NEW: Precompute common direction number combinations for each dimension."""
        combo_cache = {}
        for d in range(self.dimension):
            # Precompute all 256 possible combinations for first 8 bits
            combos = np.zeros(256, dtype=np.uint32)
            dir_nums = self.direction_numbers[d]
            for i in range(256):
                point_val = 0
                for bit in range(8):
                    if i & (1 << bit):
                        point_val ^= dir_nums[bit]
                combos[i] = point_val
            combo_cache[d] = combos
        return combo_cache

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with cache-optimized apex optimization."""
        points = np.zeros((n, self.dimension))

        # Use cached Gray codes from gen4x approach
        if n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Cache-optimized point generation using precomputed lookup tables
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]
            combo_lookup = self.dir_combo_cache[d]

            for i in range(n):
                gray = gray_codes[i]

                # NEW: Use precomputed lookup for first 8 bits (most common case)
                first_byte = gray & 0xFF
                point_val = combo_lookup[first_byte]

                # Handle remaining bits with optimized approach
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
    Particle Swarm Optimization replacing Nelder-Mead for improved performance.
    PSO is better suited for continuous optimization with fewer function evaluations.
    """

    def __init__(self, swarm_size=8, w=0.7, c1=1.4, c2=1.4):
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """PSO optimization for improved performance over Nelder-Mead."""
        n_dims = len(x0)

        # Initialize swarm positions around x0
        positions = np.zeros((self.swarm_size, n_dims))
        positions[0] = x0  # Best guess as first particle

        # Initialize other particles with small perturbations
        for i in range(1, self.swarm_size):
            positions[i] = x0 + np.random.normal(0, 0.05, n_dims)

        # Initialize velocities
        velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, n_dims))

        # Evaluate initial positions
        fitness_values = np.array([objective_func(pos) for pos in positions])

        # Initialize personal best positions and fitness
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness_values.copy()

        # Find global best
        global_best_idx = np.argmin(fitness_values)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness_values[global_best_idx]

        # Track convergence
        no_improvement_count = 0

        for iteration in range(maxiter):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity

                # Update position
                positions[i] += velocities[i]

                # Evaluate new position
                new_fitness = objective_func(positions[i])

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < global_best_fitness:
                        global_best_position = positions[i].copy()
                        global_best_fitness = new_fitness
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

            # Check convergence
            if no_improvement_count > 15:  # Early stopping
                break

            # Check tolerance
            fitness_range = np.max(personal_best_fitness) - np.min(personal_best_fitness)
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
        """Generate enhanced Sobol population using cache-optimized apex optimization."""
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
        """PSO-enhanced optimization replacing Nelder-Mead for better performance."""
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
        optimizer = ApexPSO(swarm_size=8, w=0.7, c1=1.4, c2=1.4)

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # PSO optimization with improved parameters
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
    """Generate cache-optimized apex hybrid population combining all three parent strategies."""
    print("Generating cache-optimized apex hybrid population...")

    initializer = ApexCSTInitializer(seed=525555)  # Unique apex seed

    # Stage 1: Generate apex base population with optimal sizing from all parents
    sobol_pop = initializer.generate_sobol_population(38)      # Enhanced from gen4x
    gaussian_pop = initializer.generate_apex_gaussian_population(28)  # Apex multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with PSO enhancement
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

    print(f"Generated {len(unique_population)} unique airfoils using cache-optimized approach")
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
    """Main cache-optimized hybrid initialization."""
    print("Generating cache-optimized population with improved memory access patterns...")

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
        filename = f"results/cache_optimized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['cache_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "18a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "cache_rank": i
        })

    # Enhanced diversity metrics from gen4x
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} cache-optimized airfoils")
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