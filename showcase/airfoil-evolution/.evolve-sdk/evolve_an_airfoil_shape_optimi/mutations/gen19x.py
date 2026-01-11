#!/usr/bin/env python3
"""
Gen 19x: Ultimate Hybrid Crossover
Combining the best innovations from gen13c, gen14a, and gen14b:

From gen13c (116.5472):
- PSO optimization replacing Nelder-Mead for superior global search
- Enhanced penalty system and multi-strategy population generation

From gen14a (116.5472):
- Massive lookup tables for 2x-3x Sobol generation speedup
- Ultra-fast table-driven point generation with 65536 Gray code cache

From gen14b (114.8344):
- SIMD vectorization concepts for bit operations
- Advanced broadcasting techniques for eliminating nested loops
- Enhanced population sizing and diversity strategies
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

class HybridUltimateSobol:
    """
    Ultimate hybrid Sobol combining:
    - Lookup table optimization from gen14a for maximum speed
    - SIMD vectorization concepts from gen14b for advanced operations
    - Robust caching strategies from gen13c
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0

        # Direction numbers computation
        self.direction_numbers = self._get_direction_numbers(dimension)

        # HYBRID: Combine gen14a's massive lookup tables with gen14b's SIMD concepts
        self.gray_lookup = self._precompute_gray_lookup_table()  # From gen14a
        self.xor_lookup = self._precompute_xor_lookup_table()    # From gen14a
        self.bit_contribution_lookup = self._precompute_bit_contributions()  # From gen14a

        # Enhanced caching from gen13c with gen14a's larger sizes
        self.gray_cache = self._precompute_gray_codes(4096)  # Even larger than gen14a
        self.xor_cache = self._precompute_xor_operations()
        self.bit_mask_cache = self._precompute_bit_masks()

        # HYBRID: Add gen14b's SIMD preparation for advanced operations
        self.vectorized_bit_masks = np.uint32(1) << np.arange(32, dtype=np.uint32)
        self.direction_matrix_t = self.direction_numbers.T
        self.performance_threshold = 64  # From gen14b

    def _precompute_gray_lookup_table(self) -> np.ndarray:
        """Massive Gray code lookup from gen14a for instant computation."""
        lookup_size = 65536  # 2^16 from gen14a
        gray_lookup = np.zeros(lookup_size, dtype=np.uint32)
        for i in range(lookup_size):
            gray_lookup[i] = i ^ (i >> 1)
        return gray_lookup

    def _precompute_xor_lookup_table(self) -> np.ndarray:
        """XOR lookup table from gen14a for ultra-fast operations."""
        xor_lookup = np.zeros((256, 32), dtype=np.uint32)
        for byte_val in range(256):
            for bit_pos in range(8):
                if byte_val & (1 << bit_pos):
                    xor_lookup[byte_val, bit_pos] = 1 << (31 - bit_pos)
        return xor_lookup

    def _precompute_bit_contributions(self) -> np.ndarray:
        """Bit contribution lookup from gen14a."""
        bit_contrib = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                bit_contrib[i, bit] = bool(i & (1 << bit))
        return bit_contrib

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers combining all parent approaches."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Enhanced subsequent dimensions
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Enhanced Gray code cache from gen13c/gen14a."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """XOR operation cache from gen13c."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_masks(self) -> np.ndarray:
        """Bit mask cache from gen13c."""
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def generate_points(self, n: int) -> np.ndarray:
        """
        HYBRID: Ultimate point generation combining:
        - gen14a's lookup table speed for immediate Gray code access
        - gen14b's adaptive strategy selection
        - gen13c's robust fallback mechanisms
        """
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # HYBRID: Use gen14a's massive lookup table first, then fallback hierarchy
        if n <= len(self.gray_lookup):
            gray_codes = self.gray_lookup[:n]  # Instant lookup from gen14a
        elif n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]   # Large cache from enhanced gen13c
        else:
            # Final fallback with gen14b's vectorized approach
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # HYBRID: Adaptive algorithm selection from gen14b with gen14a's optimizations
        if n <= self.performance_threshold:
            # Small batch: Use gen14a's lookup table optimization
            self._generate_lookup_optimized_points(points, gray_codes, n)
        else:
            # Large batch: Use gen13c's proven unrolled approach
            self._generate_unrolled_points(points, gray_codes, n)

        return points

    def _generate_lookup_optimized_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """HYBRID: gen14a's lookup table optimization with gen14b's vectorization concepts."""
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Use gen14a's lookup table for first 8 bits
                lower_byte = gray & 0xFF
                bit_mask = self.bit_contribution_lookup[lower_byte]

                # Vectorized XOR using precomputed contributions from gen14a
                if bit_mask[0]: point_val ^= dir_nums[0]
                if bit_mask[1]: point_val ^= dir_nums[1]
                if bit_mask[2]: point_val ^= dir_nums[2]
                if bit_mask[3]: point_val ^= dir_nums[3]
                if bit_mask[4]: point_val ^= dir_nums[4]
                if bit_mask[5]: point_val ^= dir_nums[5]
                if bit_mask[6]: point_val ^= dir_nums[6]
                if bit_mask[7]: point_val ^= dir_nums[7]

                # Handle remaining bits with gen13c's optimized approach
                remaining_gray = gray >> 8
                bit_pos = 8
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

    def _generate_unrolled_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """gen13c's proven unrolled approach for large batches."""
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Extended unrolled inner loop from gen13c
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

                # Handle remaining bits
                remaining_gray = gray >> 8
                bit_pos = 8
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

class HybridPSO:
    """
    PSO optimization from gen13c - the key performance differentiator
    over Nelder-Mead in the other approaches.
    """

    def __init__(self, swarm_size=8, w=0.7, c1=1.4, c2=1.4):
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """PSO optimization from gen13c for superior global search."""
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

class HybridCSTInitializer:
    """
    HYBRID CST initializer combining:
    - gen13c's PSO optimization and penalty system
    - gen14a's vectorized bounds caching
    - gen14b's enhanced population strategies
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from gen13c/gen14a
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays from gen14a/gen13c optimization
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
        """HYBRID: gen13c's enhanced penalty system with gen14b's insights."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Enhanced penalty system from gen13c with gen14b's insights
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 17 * (0.06 - max_t)  # Balanced between gen13c and gen14b
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
        """Convert Sobol points using cached vectorized operations from gen14a."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 40) -> List[Airfoil]:
        """Generate hybrid Sobol population using ultimate optimization."""
        sobol = HybridUltimateSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 32) -> List[Airfoil]:
        """
        HYBRID: Combine gen13c's proven Gaussian strategies with
        gen14b's enhanced parameters for optimal diversity.
        """
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution (gen13c base with gen14b insights)
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

        # Strategy 2: High-performance biased (gen13c enhanced)
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

        # Strategy 3: Thickness-optimized (gen13c with gen14b scaling)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.22, 0.052, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.142, 0.047, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.14, 0.05, 0.3)  # Balanced scaling
        thick_lower = np.clip(thick_lower * 1.14, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric enhanced (gen13c with gen14b improvements)
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.135, 0.038, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Balanced asymmetry factor
            lower_list = [-c * self.rng.uniform(0.94, 1.07) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """HYBRID: PSO optimization from gen13c with enhanced selection."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Batch evaluation from gen13c/gen14a approach
        batch_size = 12  # Balanced across parents
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
        optimizer = HybridPSO(swarm_size=8, w=0.7, c1=1.4, c2=1.4)  # PSO from gen13c

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # PSO optimization with hybrid parameters
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=38,  # Balanced between gen13c and gen14b
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """HYBRID: Combine gen13c's refinement with gen14b's enhanced scaling."""
        refined = []
        sobol = HybridUltimateSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Balanced selection
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Hybrid adaptive scale combining gen13c and gen14b insights
            base_scale = 0.096  # Balanced between gen13c (0.095) and gen14b (0.098)
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
    """Generate ultimate hybrid population combining all parent innovations."""
    print("Generating ultimate hybrid crossover population...")

    initializer = HybridCSTInitializer(seed=192019)  # Generation 19 hybrid seed

    # Stage 1: Generate hybrid base population with balanced sizing
    sobol_pop = initializer.generate_sobol_population(40)      # Balanced across parents
    gaussian_pop = initializer.generate_hybrid_gaussian_population(32)  # Hybrid multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with PSO from gen13c
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Hybrid adaptive refinement around best solutions
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Hybrid deduplication with balanced tolerance
    unique_population = []
    tolerance = 7e-5  # Balanced between gen13c and gen14b

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using hybrid crossover optimization")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Hybrid population evaluation with optimized batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing
    batch_size = 13  # Balanced across all parents
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid crossover initialization."""
    print("Generating ultimate hybrid crossover population combining PSO + Lookup Tables + SIMD concepts...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:35]):  # Balanced save count
        filename = f"results/hybrid_crossover_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "19x"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Hybrid diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} hybrid crossover airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
        print(f"Top 16 mean: {np.mean(valid_fitnesses[:16]):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()