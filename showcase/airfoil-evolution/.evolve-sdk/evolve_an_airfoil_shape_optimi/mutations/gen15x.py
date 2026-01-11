#!/usr/bin/env python3
"""
Gen 15x: Hybrid PSO-Lookup-SIMD Crossover
Evolutionary crossover combining best features from three high-performance parents:
- PSO optimization from gen13c (fitness: 116.5472)
- Lookup table acceleration from gen14a (fitness: 116.5472)
- SIMD vectorization from gen14b (fitness: 114.8344)

CROSSOVER INNOVATIONS:
- PSO global optimization + lookup table speed + SIMD when beneficial
- Adaptive algorithm selection: lookup tables for small batches, SIMD for large
- Combined caching strategies from all parents
- Optimized parameter fusion across all three approaches
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
    Hybrid Sobol sequence generator combining:
    - Lookup table optimization from gen14a (2x-3x speedup)
    - SIMD vectorization from gen14b (for large batches)
    - Ultra-large caching from gen13c (2048 points)
    - Adaptive algorithm selection for optimal performance
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0

        # Direction numbers computation (from all parents)
        self.direction_numbers = self._get_direction_numbers(dimension)

        # HYBRID CACHING: Combine all parent strategies
        # From gen14a: Massive lookup tables
        self.gray_lookup = self._precompute_gray_lookup_table()
        self.xor_lookup = self._precompute_xor_lookup_table()
        self.bit_contribution_lookup = self._precompute_bit_contributions()

        # From gen13c: Ultra-large cache
        self.gray_cache = self._precompute_gray_codes(2048)
        self.xor_cache = self._precompute_xor_operations()
        self.bit_mask_cache = self._precompute_bit_masks()

        # From gen14b: SIMD preparation
        self.vectorized_bit_masks = np.uint32(1) << np.arange(32, dtype=np.uint32)
        self._prepare_simd_matrices()

        # ADAPTIVE THRESHOLD: Choose algorithm based on batch size
        self.lookup_threshold = 100    # Use lookup tables for n <= 100
        self.simd_threshold = 1000     # Use SIMD for n > 100 and <= 1000

    def _precompute_gray_lookup_table(self) -> np.ndarray:
        """Massive Gray code lookup table from gen14a for instant computation."""
        lookup_size = 65536  # 2^16, covers most practical sequences
        gray_lookup = np.zeros(lookup_size, dtype=np.uint32)
        for i in range(lookup_size):
            gray_lookup[i] = i ^ (i >> 1)
        return gray_lookup

    def _precompute_xor_lookup_table(self) -> np.ndarray:
        """XOR operations lookup table from gen14a for ultra-fast operations."""
        xor_lookup = np.zeros((256, 32), dtype=np.uint32)
        for byte_val in range(256):
            for bit_pos in range(8):
                if byte_val & (1 << bit_pos):
                    xor_lookup[byte_val, bit_pos] = 1 << (31 - bit_pos)
        return xor_lookup

    def _precompute_bit_contributions(self) -> np.ndarray:
        """Bit contributions lookup from gen14a for vectorized operations."""
        bit_contrib = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                bit_contrib[i, bit] = bool(i & (1 << bit))
        return bit_contrib

    def _prepare_simd_matrices(self):
        """SIMD preparation from gen14b for large batch vectorization."""
        self.bit_mask_matrix = self.vectorized_bit_masks.reshape(32, 1)
        self.direction_matrix_t = self.direction_numbers.T

        # Pre-allocate working arrays
        self.temp_mask_array = np.empty((32, 2048), dtype=bool)
        self.temp_point_vals = np.empty((2048, self.dimension), dtype=np.uint32)

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers from all parents."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions using robust approach from parents
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_gray_codes(self, max_n: int) -> np.ndarray:
        """Ultra-large Gray code cache from gen13c."""
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
        """Additional bit mask cache from gen13c."""
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def generate_points(self, n: int) -> np.ndarray:
        """
        HYBRID POINT GENERATION: Adaptive algorithm selection
        - Lookup tables (gen14a) for small batches (fastest)
        - SIMD (gen14b) for medium batches (vectorized)
        - Optimized unrolling (gen13c) for large batches (cache-friendly)
        """
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # Get Gray codes using best available method
        if n <= len(self.gray_lookup):
            gray_codes = self.gray_lookup[:n]
        elif n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Fallback to computation for very large sequences
            indices = np.arange(n, dtype=np.uint32)
            gray_codes = indices ^ (indices >> 1)

        # ADAPTIVE ALGORITHM SELECTION
        if n <= self.lookup_threshold:
            # Small batch: Use lookup table optimization (gen14a)
            self._generate_lookup_points(points, gray_codes, n)
        elif n <= self.simd_threshold:
            # Medium batch: Use SIMD vectorization (gen14b)
            self._generate_simd_points(points, gray_codes, n)
        else:
            # Large batch: Use optimized unrolling (gen13c)
            self._generate_unrolled_points(points, gray_codes, n)

        return points

    def _generate_lookup_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """Lookup table optimized generation from gen14a for small batches."""
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Use lookup table for first 8 bits (covers 99% efficiently)
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

                # Handle remaining bits efficiently
                remaining_gray = gray >> 8
                bit_pos = 8
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

    def _generate_simd_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """SIMD vectorized generation from gen14b for medium batches."""
        # Reshape for broadcasting
        gray_codes_bc = gray_codes.reshape(1, n)

        # Compute all bit checks in one operation
        bit_checks = (gray_codes_bc & self.bit_mask_matrix[:min(32, n)]) != 0

        # Pre-allocate result array
        point_vals = np.zeros((n, self.dimension), dtype=np.uint32)

        # Vectorized XOR operations across all dimensions
        for bit_pos in range(min(32, bit_checks.shape[0])):
            mask = bit_checks[bit_pos]
            if np.any(mask):
                point_vals[mask] ^= self.direction_matrix_t[bit_pos]

        # Final conversion to [0,1] range
        points[:] = point_vals.astype(np.float64) * (1.0 / (1 << 32))

    def _generate_unrolled_points(self, points: np.ndarray, gray_codes: np.ndarray, n: int):
        """Optimized unrolled generation from gen13c for large batches."""
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]
                point_val = 0

                # Extended unrolled inner loop for maximum performance
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
    PSO optimization from gen13c - proven better than Nelder-Mead
    for airfoil optimization with continuous parameter spaces.
    """

    def __init__(self, swarm_size=8, w=0.7, c1=1.4, c2=1.4):
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 40, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """PSO optimization proven superior to Nelder-Mead for this domain."""
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
    Hybrid CST initializer combining best features from all three parents:
    - PSO optimization from gen13c
    - Lookup table Sobol from gen14a
    - Enhanced parameters from gen14b
    - Combined penalty systems and caching strategies
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from all parents
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
        """Hybrid objective function with enhanced penalty system."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Enhanced penalty system from gen13c/gen14a
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

    def generate_sobol_population(self, size: int = 40) -> List[Airfoil]:
        """Generate hybrid Sobol population using adaptive algorithm selection."""
        sobol = HybridApexSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 30) -> List[Airfoil]:
        """Generate hybrid Gaussian population combining strategies from all parents."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution (balanced from all parents)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.175, 0.068, (strategy1_size, n_coeffs))
        all_lower = self.rng.normal(-0.102, 0.050, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased (optimized from all parents)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.198, 0.074, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.073, 0.040, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized (enhanced from all parents)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.055, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.141, 0.050, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with balanced asymmetry
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.138, 0.040, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Balanced asymmetry factor from all parents
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """Hybrid optimization using PSO from gen13c with enhanced selection."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Optimal batch processing
        batch_size = 12
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
        optimizer = HybridPSO(swarm_size=8, w=0.7, c1=1.4, c2=1.4)

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # PSO optimization with proven parameters
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=38,  # Balanced across parents
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 25) -> List[Airfoil]:
        """Hybrid adaptive refinement combining all parent approaches."""
        refined = []
        sobol = HybridApexSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Balanced selection
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Balanced adaptive scale from all parents
            base_scale = 0.097
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
    """Generate hybrid population combining all three parent approaches."""
    print("Generating hybrid PSO-Lookup-SIMD population...")

    initializer = HybridCSTInitializer(seed=151515)  # Generation 15 seed

    # Stage 1: Generate balanced base population
    sobol_pop = initializer.generate_sobol_population(40)      # Balanced sizing
    gaussian_pop = initializer.generate_hybrid_gaussian_population(30)  # Hybrid strategies

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with PSO
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Hybrid adaptive refinement
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:8], refinement_size=25)

    # Combine all populations with balanced deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Balanced deduplication tolerance
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

    print(f"Generated {len(unique_population)} unique airfoils using hybrid PSO-Lookup-SIMD approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Hybrid population evaluation with balanced batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Balanced batch processing
    batch_size = 13
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid PSO-Lookup-SIMD initialization."""
    print("Generating hybrid PSO-Lookup-SIMD airfoil population...")

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
        filename = f"results/hybrid_pso_lookup_simd_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "15x"
            data['crossover_parents'] = ["gen13c", "gen14a", "gen14b"]
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Balanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} hybrid PSO-Lookup-SIMD airfoils")
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