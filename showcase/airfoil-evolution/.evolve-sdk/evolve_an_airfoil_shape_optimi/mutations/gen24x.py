#!/usr/bin/env python3
"""
Gen 24x: Ultra-Hybrid Performance Crossover
Crossover mutation: Combines the best optimizations from gen13c, gen14a, and gen20a
for maximum performance in all aspects of the airfoil optimization pipeline.
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

class UltraApexSobol:
    """
    Ultra-Hybrid Sobol sequence generator combining ALL optimization techniques:
    - Massive lookup tables from gen14a (65536-entry Gray lookup)
    - SIMD vectorization from gen20a (parallel processing)
    - Enhanced caching from all parents
    - Hybrid bit processing combining lookup + vectorization
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0

        # Direction numbers computation
        self.direction_numbers = self._get_direction_numbers(dimension)

        # HYBRID: Combine gen14a lookup tables with gen20a vectorization
        self.gray_lookup = self._precompute_gray_lookup_table()  # From gen14a
        self.bit_contribution_lookup = self._precompute_bit_contributions()  # From gen14a
        self.xor_lookup = self._precompute_xor_lookup_table()  # From gen14a

        # Enhanced caches from all parents
        self.gray_cache = self._precompute_gray_codes(4096)  # Expanded from gen14a
        self.xor_cache = self._precompute_xor_operations()
        self.bit_mask_cache = self._precompute_bit_masks()

    def _precompute_gray_lookup_table(self) -> np.ndarray:
        """Massive Gray code lookup table from gen14a for instant computation."""
        lookup_size = 65536  # 2^16 coverage
        gray_lookup = np.zeros(lookup_size, dtype=np.uint32)
        for i in range(lookup_size):
            gray_lookup[i] = i ^ (i >> 1)
        return gray_lookup

    def _precompute_bit_contributions(self) -> np.ndarray:
        """Bit contribution lookup from gen14a for fast 8-bit pattern processing."""
        bit_contrib = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                bit_contrib[i, bit] = bool(i & (1 << bit))
        return bit_contrib

    def _precompute_xor_lookup_table(self) -> np.ndarray:
        """XOR lookup table from gen14a for ultra-fast operations."""
        xor_lookup = np.zeros((256, 32), dtype=np.uint32)
        for byte_val in range(256):
            for bit_pos in range(8):
                if byte_val & (1 << bit_pos):
                    xor_lookup[byte_val, bit_pos] = 1 << (31 - bit_pos)
        return xor_lookup

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation."""
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
        """Precompute Gray codes with expanded cache size."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_xor_operations(self) -> np.ndarray:
        """Enhanced XOR operation cache."""
        bit_masks = np.zeros(32, dtype=np.uint32)
        for j in range(32):
            bit_masks[j] = 1 << j
        return bit_masks

    def _precompute_bit_masks(self) -> np.ndarray:
        """Additional bit mask cache for maximum performance."""
        bit_patterns = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            bit_patterns[i] = i
        return bit_patterns

    def generate_points(self, n: int) -> np.ndarray:
        """
        Ultra-Hybrid Sobol point generation combining:
        - gen14a lookup tables for fast Gray code computation
        - gen20a SIMD vectorization for parallel processing
        - Optimized hybrid approach for maximum speed
        """
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # HYBRID: Use massive lookup table from gen14a with fallbacks
        if n <= len(self.gray_lookup):
            gray_codes = self.gray_lookup[:n]
        elif n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Vectorized fallback for very large sequences
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # HYBRID: Combine gen14a lookup + gen20a vectorization
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            # VECTORIZED: Process all points simultaneously (from gen20a)
            gray_array = gray_codes[:, np.newaxis]  # Shape: (n, 1)

            # LOOKUP TABLE: Use gen14a bit contribution lookup for first 8 bits
            lower_bytes = gray_codes & 0xFF  # Extract lower 8 bits

            # Vectorized lookup of bit contributions
            bit_masks = self.bit_contribution_lookup[lower_bytes]  # Shape: (n, 8)

            # Vectorized XOR operations for first 8 bits
            dir_nums_subset = dir_nums[:8][np.newaxis, :]  # Shape: (1, 8)
            xor_values = np.where(bit_masks, dir_nums_subset, 0)
            point_vals = np.bitwise_xor.reduce(xor_values, axis=1)  # Shape: (n,)

            # HYBRID: Handle remaining bits (9-31) with optimized vectorized approach
            remaining_gray = gray_codes >> 8
            remaining_mask = remaining_gray != 0

            if np.any(remaining_mask):
                # Only process points that have remaining bits
                active_indices = np.where(remaining_mask)[0]
                active_gray = remaining_gray[active_indices]

                # Vectorized processing of remaining bits (from gen20a approach)
                for bit_pos in range(8, min(32, len(dir_nums))):
                    bit_mask = 1 << (bit_pos - 8)
                    bit_test = (active_gray & bit_mask) != 0

                    # Apply XOR where bit is set
                    point_vals[active_indices] = np.where(
                        bit_test,
                        point_vals[active_indices] ^ dir_nums[bit_pos],
                        point_vals[active_indices]
                    )

            # Convert to floating point in vectorized manner
            points[:, d] = point_vals.astype(np.float64) / (1 << 32)

        return points

class UltraApexPSO:
    """
    Ultra-Enhanced PSO from gen13c with additional optimizations:
    - Improved parameter tuning
    - Enhanced convergence detection
    - Adaptive parameter adjustment
    """

    def __init__(self, swarm_size=10, w=0.729, c1=1.494, c2=1.494):
        # Optimized PSO parameters based on research
        self.swarm_size = swarm_size
        self.w = w      # Optimized inertia weight
        self.c1 = c1    # Optimized cognitive component
        self.c2 = c2    # Optimized social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 45, tolerance: float = 1e-5) -> Tuple[np.ndarray, float]:
        """Ultra-enhanced PSO optimization with adaptive parameters."""
        n_dims = len(x0)

        # Initialize swarm positions around x0
        positions = np.zeros((self.swarm_size, n_dims))
        positions[0] = x0  # Best guess as first particle

        # Enhanced initialization with better diversity
        for i in range(1, self.swarm_size):
            perturbation = np.random.normal(0, 0.04, n_dims)  # Slightly reduced noise
            positions[i] = x0 + perturbation

        # Enhanced velocity initialization
        velocities = np.random.uniform(-0.08, 0.08, (self.swarm_size, n_dims))

        # Evaluate initial positions
        fitness_values = np.array([objective_func(pos) for pos in positions])

        # Initialize personal best positions and fitness
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness_values.copy()

        # Find global best
        global_best_idx = np.argmin(fitness_values)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness_values[global_best_idx]

        # Enhanced convergence tracking
        no_improvement_count = 0
        best_fitness_history = [global_best_fitness]

        for iteration in range(maxiter):
            # Adaptive parameter adjustment
            progress = iteration / maxiter
            adaptive_w = self.w * (0.5 + 0.5 * (1 - progress))  # Decrease inertia over time

            for i in range(self.swarm_size):
                # Update velocity with adaptive parameters
                r1, r2 = np.random.random(2)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = adaptive_w * velocities[i] + cognitive_velocity + social_velocity

                # Velocity clamping for stability
                v_max = 0.15 * (1 - progress * 0.5)  # Decrease max velocity over time
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

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

            # Enhanced convergence detection
            best_fitness_history.append(global_best_fitness)

            # Check for stagnation
            if no_improvement_count > 18:  # Enhanced early stopping
                break

            # Check tolerance with improved criteria
            if len(best_fitness_history) >= 5:
                recent_improvement = best_fitness_history[-5] - global_best_fitness
                if recent_improvement < tolerance:
                    break

            fitness_range = np.max(personal_best_fitness) - np.min(personal_best_fitness)
            if fitness_range < tolerance * 0.5:
                break

        return global_best_position, global_best_fitness

class UltraApexCSTInitializer:
    """
    Ultra-Hybrid CST initializer combining ALL best features:
    - Ultra-hybrid Sobol from gen24x combining lookup tables + vectorization
    - Enhanced PSO optimization from gen13c with improvements
    - Advanced multi-strategy Gaussian from all parents
    - Vectorized bounds caching from gen3a
    - Enhanced penalty systems from gen4x
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds (best from all parents)
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

        # Cache bounds arrays for maximum vectorized speed
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
        """Ultra-enhanced objective function with advanced penalty system."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Enhanced penalty system combining all parent approaches
                penalty = 0
                max_t, _ = airfoil.max_thickness()

                # Thickness constraints with enhanced penalties
                if max_t < 0.06:
                    penalty += 18 * (0.06 - max_t)**1.2  # Stronger penalty with non-linear term
                if max_t > 0.18:
                    penalty += 18 * (max_t - 0.18)**1.2

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
        """Generate ultra-hybrid Sobol population using combined optimizations."""
        sobol = UltraApexSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_ultra_gaussian_population(self, size: int = 32) -> List[Airfoil]:
        """Generate ultra-enhanced Gaussian population combining all strategies."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution (best from all parents)
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.175, 0.062, (strategy1_size, n_coeffs))  # Fine-tuned
        all_lower = self.rng.normal(-0.098, 0.045, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: Ultra high-performance biased
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.205, 0.068, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.072, 0.035, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Ultra thickness-optimized
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.048, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.138, 0.043, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)  # Enhanced scaling
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Ultra near-symmetric with enhanced bias
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.142, 0.035, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor with better tuning
            lower_list = [-c * self.rng.uniform(0.92, 1.09) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """Ultra-enhanced optimization using improved PSO."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Enhanced batch evaluation
        batch_size = 12
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
        optimizer = UltraApexPSO(swarm_size=10, w=0.729, c1=1.494, c2=1.494)

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Ultra-enhanced PSO optimization
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=45,
                    tolerance=1e-5
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def ultra_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Ultra-enhanced adaptive refinement combining all approaches."""
        refined = []
        sobol = UltraApexSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Enhanced to top 8
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Ultra-enhanced adaptive scale
            base_scale = 0.088  # Fine-tuned base scale
            rank_factor = good_airfoils.index(airfoil)
            scale = base_scale * (1.0 + 0.11 * rank_factor)

            for point in local_points:
                # Ultra-enhanced perturbation strategy
                perturbation_upper = (point[:5] - 0.5) * scale * (np.abs(center_upper) + 0.01)
                perturbation_lower = (point[5:] - 0.5) * scale * (np.abs(center_lower) + 0.01)

                new_upper = center_upper + perturbation_upper
                new_lower = center_lower + perturbation_lower

                # Enhanced bounds clamping
                new_upper = np.clip(new_upper, 0.01, 0.3)
                new_lower = np.clip(new_lower, -0.3, -0.01)

                refined.append(Airfoil(
                    upper_coeffs=new_upper.tolist(),
                    lower_coeffs=new_lower.tolist(),
                    zte=0.0
                ))

        return refined

def generate_population() -> List[Airfoil]:
    """Generate ultra-hybrid population combining ALL optimization techniques."""
    print("Generating ultra-hybrid population with combined optimizations from gen13c, gen14a, and gen20a...")

    initializer = UltraApexCSTInitializer(seed=246810)  # Ultra-unique seed

    # Stage 1: Generate ultra base population with enhanced sizing
    sobol_pop = initializer.generate_sobol_population(40)      # Enhanced from all parents
    gaussian_pop = initializer.generate_ultra_gaussian_population(32)  # Ultra multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with ultra PSO
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Ultra adaptive refinement around best solutions
    refined_pop = initializer.ultra_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with ultra-intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Ultra-enhanced deduplication with adaptive tolerance
    unique_population = []
    tolerance = 7e-5  # Tighter tolerance for better uniqueness

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using ultra-hybrid crossover approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Ultra-enhanced population evaluation with optimized batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Enhanced batch processing
    batch_size = 16
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main ultra-hybrid crossover initialization."""
    print("Generating ultra-hybrid population combining gen13c PSO, gen14a lookup tables, and gen20a vectorization...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:35]):  # Save top 35
        filename = f"results/ultra_hybrid_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['ultra_hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "24x"
            data['crossover_parents'] = ["gen13c", "gen14a", "gen20a"]
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "ultra_hybrid_rank": i
        })

    # Ultra-enhanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} ultra-hybrid airfoils")
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