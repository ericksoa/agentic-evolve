#!/usr/bin/env python3
"""
Gen 25x: Ultimate Hybrid Crossover Optimizer
Crossover hybrid combining the best innovations from multiple high-performing parents:
- PSO optimization from gen13c for superior convergence
- Lookup table optimization from gen14a for ultra-fast computation
- SIMD vectorization from gen20a for massive parallel speedup
- Enhanced multi-level caching and adaptive refinement
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
    Ultimate hybrid Sobol sequence generator combining ALL optimization techniques:
    - Ultra-large cache from gen13c (2048 points)
    - Massive lookup tables from gen14a (65536 Gray codes)
    - SIMD vectorization from gen20a for parallel processing
    - Enhanced bit operations and XOR optimizations
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Direction numbers computation
        self.direction_numbers = self._get_direction_numbers(dimension)

        # HYBRID FEATURE 1: Massive lookup tables from gen14a
        self.gray_lookup = self._precompute_gray_lookup_table()
        self.xor_lookup = self._precompute_xor_lookup_table()
        self.bit_contribution_lookup = self._precompute_bit_contributions()

        # HYBRID FEATURE 2: Ultra-large cache from gen13c
        self.gray_cache = self._precompute_gray_codes(4096)  # Even larger than parents
        self.xor_cache = self._precompute_xor_operations()
        self.bit_mask_cache = self._precompute_bit_masks()

    def _precompute_gray_lookup_table(self) -> np.ndarray:
        """Precompute Gray codes for all 16-bit values for instant lookup (from gen14a)."""
        lookup_size = 65536
        gray_lookup = np.zeros(lookup_size, dtype=np.uint32)
        for i in range(lookup_size):
            gray_lookup[i] = i ^ (i >> 1)
        return gray_lookup

    def _precompute_xor_lookup_table(self) -> np.ndarray:
        """Precompute XOR operations for 8-bit values for ultra-fast lookup (from gen14a)."""
        xor_lookup = np.zeros((256, 32), dtype=np.uint32)
        for byte_val in range(256):
            for bit_pos in range(8):
                if byte_val & (1 << bit_pos):
                    xor_lookup[byte_val, bit_pos] = 1 << (31 - bit_pos)
        return xor_lookup

    def _precompute_bit_contributions(self) -> np.ndarray:
        """Precompute bit contributions for each 8-bit pattern (from gen14a)."""
        bit_contrib = np.zeros((256, 8), dtype=np.bool_)
        for i in range(256):
            for bit in range(8):
                bit_contrib[i, bit] = bool(i & (1 << bit))
        return bit_contrib

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation from all parents."""
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
        """Precompute Gray codes with ultra-large cache."""
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
        HYBRID: Ultra-optimized Sobol point generation combining:
        - Lookup tables from gen14a for instant computation
        - SIMD vectorization from gen20a for parallel processing
        - Enhanced caching from gen13c for maximum performance
        """
        points = np.zeros((n, self.dimension), dtype=np.float64)

        # HYBRID: Use massive Gray code lookup table for instant computation (gen14a)
        if n <= len(self.gray_lookup):
            gray_codes = self.gray_lookup[:n]
        elif n <= len(self.gray_cache):
            gray_codes = self.gray_cache[:n]
        else:
            # Fallback to computation only for very large sequences
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # HYBRID: SIMD VECTORIZATION from gen20a + Lookup optimization from gen14a
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            # Use lookup table for first 8 bits (gen14a approach)
            lower_bytes = gray_codes & 0xFF

            # Vectorized lookup for all points simultaneously (hybrid approach)
            bit_masks = self.bit_contribution_lookup[lower_bytes]  # Shape: (n, 8)

            # Vectorized XOR operations for first 8 bits using both techniques
            point_vals = np.zeros(n, dtype=np.uint32)
            for bit_pos in range(8):
                mask = bit_masks[:, bit_pos]
                point_vals = np.where(mask, point_vals ^ dir_nums[bit_pos], point_vals)

            # Handle remaining bits (9-31) in vectorized fashion (from gen20a)
            remaining_gray = gray_codes >> 8
            remaining_mask = remaining_gray != 0

            if np.any(remaining_mask):
                # Only process points that have remaining bits
                active_indices = np.where(remaining_mask)[0]
                active_gray = remaining_gray[active_indices]

                # Vectorized processing of remaining bits
                for bit_pos in range(8, 32):
                    if bit_pos >= len(dir_nums):
                        break

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

class HybridApexPSO:
    """
    Enhanced PSO from gen13c with improved convergence parameters.
    """

    def __init__(self, swarm_size=10, w=0.72, c1=1.49, c2=1.49):
        # Slightly enhanced parameters for better performance
        self.swarm_size = swarm_size
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive component
        self.c2 = c2    # Social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 42, tolerance: float = 1e-4) -> Tuple[np.ndarray, float]:
        """Enhanced PSO optimization with improved convergence."""
        n_dims = len(x0)

        # Initialize swarm positions around x0
        positions = np.zeros((self.swarm_size, n_dims))
        positions[0] = x0  # Best guess as first particle

        # Initialize other particles with small perturbations
        for i in range(1, self.swarm_size):
            positions[i] = x0 + np.random.normal(0, 0.045, n_dims)

        # Initialize velocities
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
            if no_improvement_count > 18:  # Enhanced early stopping
                break

            # Check tolerance
            fitness_range = np.max(personal_best_fitness) - np.min(personal_best_fitness)
            if fitness_range < tolerance:
                break

        return global_best_position, global_best_fitness

class HybridApexCSTInitializer:
    """
    Ultimate hybrid CST initializer combining ALL the best features:
    - Hybrid Sobol with lookup tables + SIMD vectorization
    - Enhanced PSO optimization from gen13c
    - Advanced multi-strategy Gaussian from all parents
    - Vectorized bounds caching for maximum performance
    - Enhanced penalty systems
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds from all parents
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
        """Enhanced objective function with advanced penalty system."""
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
                    penalty += 17 * (0.06 - max_t)  # Slightly enhanced penalty
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
        """Generate enhanced Sobol population using hybrid optimization."""
        sobol = HybridApexSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_hybrid_gaussian_population(self, size: int = 30) -> List[Airfoil]:
        """Enhanced Gaussian population combining all strategies from all parents."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution
        strategy1_size = size // 4
        all_upper = self.rng.normal(0.175, 0.068, (strategy1_size, n_coeffs))
        all_lower = self.rng.normal(-0.102, 0.051, (strategy1_size, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(strategy1_size):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-performance biased (enhanced)
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.198, 0.075, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.072, 0.041, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized (enhanced)
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.055, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.145, 0.049, (strategy3_size, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.15, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.15, -0.3, -0.05)

        for i in range(strategy3_size):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Near-symmetric with enhanced bias
        strategy4_size = size - strategy1_size - strategy2_size - strategy3_size
        sym_upper = self.rng.normal(0.138, 0.041, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor
            lower_list = [-c * self.rng.uniform(0.93, 1.08) for c in upper_list]
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=lower_list,
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 16) -> List[Airfoil]:
        """Enhanced PSO optimization with improved parameters."""
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        # Batch evaluation
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
        optimizer = HybridApexPSO(swarm_size=10, w=0.72, c1=1.49, c2=1.49)

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Enhanced PSO optimization
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=42,
                    tolerance=1e-4
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def hybrid_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Hybrid adaptive refinement using enhanced Sobol generator."""
        refined = []
        sobol = HybridApexSobol(dimension=10)

        for airfoil in good_airfoils[:8]:  # Enhanced to top 8
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Enhanced adaptive scale
            base_scale = 0.098
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
    """Generate ultimate hybrid population combining ALL best features."""
    print("Generating ultimate hybrid population with combined optimizations...")

    initializer = HybridApexCSTInitializer(seed=252525)  # Unique hybrid seed

    # Stage 1: Generate hybrid base population with enhanced sizing
    sobol_pop = initializer.generate_sobol_population(40)      # Enhanced from all parents
    gaussian_pop = initializer.generate_hybrid_gaussian_population(30)  # Hybrid multi-strategy

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced PSO
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Hybrid adaptive refinement around best solutions
    refined_pop = initializer.hybrid_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with optimal tolerance
    unique_population = []
    tolerance = 7e-5  # Tighter tolerance for better quality

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using ultimate hybrid approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimal batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimal batch processing
    batch_size = 15
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main ultimate hybrid initialization."""
    print("Generating ultimate hybrid population with combined optimizations from all parents...")

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
        filename = f"results/hybrid_ultimate_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['hybrid_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "25x"
            data['hybrid_features'] = [
                "PSO_optimization_from_gen13c",
                "lookup_tables_from_gen14a",
                "SIMD_vectorization_from_gen20a",
                "enhanced_multi_strategy_gaussian",
                "hybrid_adaptive_refinement"
            ]
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "hybrid_rank": i
        })

    # Enhanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} ultimate hybrid airfoils")
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