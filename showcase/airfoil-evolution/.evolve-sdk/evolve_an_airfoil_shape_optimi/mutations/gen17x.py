#!/usr/bin/env python3
"""
Gen 17x: PSO-Lookup Table Hybrid Crossover
Combining the best innovations from multiple high-performing parents:
- PSO optimization from gen13c (116.5472) for superior global optimization
- Lookup table optimizations from gen14a (116.5472) for 2x-3x speedup
- Selected vectorization enhancements from gen14b (114.8344)

CROSSOVER STRATEGY:
- Core Algorithm: PSO from gen13c (proven best fitness)
- Performance Layer: Lookup tables from gen14a (massive speedup)
- Enhancement Layer: Selective vectorization from gen14b
- Hybrid Penalty System: Balanced approach from all parents
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

class HybridApexSobol:
    """
    Hybrid Sobol generator combining lookup table optimization (gen14a)
    with selective vectorization (gen14b) and proven caching (gen13c).
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0

        # Core direction numbers from gen13c approach
        self.direction_numbers = self._get_direction_numbers(dimension)

        # Massive lookup tables from gen14a for ultra-performance
        self.gray_lookup = self._precompute_gray_lookup_table()
        self.xor_lookup = self._precompute_xor_lookup_table()
        self.bit_contribution_lookup = self._precompute_bit_contributions()

        # Enhanced cache sizes from gen14a
        self.gray_cache = self._precompute_gray_codes(4096)  # Doubled cache from gen14a
        self.xor_cache = self._precompute_xor_operations()
        self.bit_mask_cache = self._precompute_bit_masks()

        # Performance threshold from gen14b for adaptive algorithm selection
        self.performance_threshold = 64

    def _precompute_gray_lookup_table(self) -> np.ndarray:
        """Massive Gray code lookup table from gen14a for instant computation."""
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
        """Bit contribution lookup from gen14a for vectorized operations."""
        bit_contrib = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                bit_contrib[i, bit] = bool(i & (1 << bit))
        return bit_contrib

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers from gen13c with proven robustness."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions with gen13c robustness
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Large Gray code cache from gen14a."""
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
        """Hybrid point generation combining lookup tables with adaptive strategy."""
        points = np.zeros((n, self.dimension))

        # Use massive lookup table from gen14a when possible
        if n <= len(self.gray_lookup):
            gray_codes = self.gray_lookup[:n]
        elif n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Fallback to computation
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Adaptive strategy: choose algorithm based on size
        if n <= self.performance_threshold:
            # Small batch: Use lookup table optimized approach from gen14a
            self._generate_lookup_optimized_points(points, gray_codes, n)
        else:
            # Large batch: Use proven unrolled approach from gen13c
            self._generate_unrolled_points(points, gray_codes, n)

        return points

    def _generate_lookup_optimized_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """Lookup table optimized generation from gen14a for small batches."""
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Use lookup table for first 8 bits from gen14a
                lower_byte = gray & 0xFF
                bit_mask = self.bit_contribution_lookup[lower_byte]

                # Vectorized XOR using precomputed contributions
                if bit_mask[0]: point_val ^= dir_nums[0]
                if bit_mask[1]: point_val ^= dir_nums[1]
                if bit_mask[2]: point_val ^= dir_nums[2]
                if bit_mask[3]: point_val ^= dir_nums[3]
                if bit_mask[4]: point_val ^= dir_nums[4]
                if bit_mask[5]: point_val ^= dir_nums[5]
                if bit_mask[6]: point_val ^= dir_nums[6]
                if bit_mask[7]: point_val ^= dir_nums[7]

                # Handle remaining bits
                remaining_gray = gray >> 8
                bit_pos = 8
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

    def _generate_unrolled_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """Proven unrolled approach from gen13c for large batches."""
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
    Proven PSO implementation from gen13c with the best fitness results.
    Keeping exact parameters that achieved 116.5472 fitness.
    """

    def __init__(self, swarm_size=8, w=0.7, c1=1.4, c2=1.4):
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Exact PSO implementation from gen13c that achieved 116.5472 fitness."""
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

            # Check convergence - exact same logic from gen13c
            if no_improvement_count > 15:  # Early stopping
                break

            # Check tolerance
            fitness_range = np.max(personal_best_fitness) - np.min(personal_best_fitness)
            if fitness_range < tolerance:
                break

        return global_best_position, global_best_fitness

class HybridCSTInitializer:
    """
    Hybrid CST initializer combining proven approaches:
    - PSO optimization from gen13c (best fitness)
    - Lookup table speedups from gen14a
    - Balanced penalty system from all parents
    - Enhanced population strategies
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Proven parameter bounds from gen13c
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for vectorized operations
        self._cache_bounds_arrays()

    def _cache_bounds_arrays(self):
        """Cache bounds as numpy arrays for vectorized speed."""
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
        """Hybrid penalty system balancing all parents."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Balanced penalty system - average of parents (16 from gen13c, 16 from gen14a, 18 from gen14b)
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 17 * (0.06 - max_t)  # Balanced penalty multiplier
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
        """Convert Sobol points using cached vectorized operations."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 40) -> List[Airfoil]:
        """Generate Sobol population using hybrid optimization."""
        sobol = HybridApexSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 32) -> List[Airfoil]:
        """Generate Gaussian population combining strategies from all parents."""
        population = []
        n_coeffs = 5

        # Strategy 1: Standard distribution from gen13c (proven performance)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.17, 0.065, (strategy1_size, n_coeffs))  # gen13c params
        all_lower = self.rng.normal(-0.105, 0.048, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased (averaging gen13c and gen14b)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.198, 0.074, (strategy2_size, n_coeffs))  # Balanced
        lower_biased = self.rng.normal(-0.072, 0.040, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized (balanced across parents)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.054, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.141, 0.049, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.14, 0.05, 0.3)  # Balanced multiplier
        thick_lower = np.clip(thick_lower * 1.14, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric (balanced approach)
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.138, 0.039, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Balanced asymmetry factor
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """PSO optimization from gen13c with balanced selection."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Balanced batch processing
        batch_size = 12  # Average of parents
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
        optimizer = HybridPSO(swarm_size=8, w=0.7, c1=1.4, c2=1.4)  # Exact gen13c params

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # PSO optimization with proven parameters from gen13c
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=35,  # gen13c proven value
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Adaptive refinement balancing all parent approaches."""
        refined = []
        sobol = HybridApexSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Balanced selection
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Balanced adaptive scale
            base_scale = 0.097  # Average of parent scales
            scale = base_scale * (1.0 + 0.13 * good_airfoils.index(airfoil))

            for point in local_points:
                # Balanced perturbation strategy
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
    """Generate hybrid population combining best features from all parents."""
    print("Generating PSO-Lookup Table Hybrid population...")

    initializer = HybridCSTInitializer(seed=171717)  # Generation 17 seed

    # Stage 1: Generate base population with balanced sizing
    sobol_pop = initializer.generate_sobol_population(40)      # Balanced across parents
    gaussian_pop = initializer.generate_hybrid_gaussian_population(32)  # Hybrid strategies

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize with PSO from gen13c (best performer)
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Adaptive refinement with balanced approach
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with balanced deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Balanced deduplication tolerance
    unique_population = []
    tolerance = 7e-5  # Average of parent tolerances

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using PSO-Lookup Table hybrid")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Balanced population evaluation combining parent approaches."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Balanced batch processing
    batch_size = 13  # Average of parents (10, 14, 15)
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid crossover initialization."""
    print("Generating PSO-Lookup Table Hybrid population with best features from all parents...")

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
            data['generation'] = "17x"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Balanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} PSO-Lookup Table hybrid airfoils")
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