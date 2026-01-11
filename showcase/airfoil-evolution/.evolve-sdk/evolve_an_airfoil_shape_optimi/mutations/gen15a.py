#!/usr/bin/env python3
"""
Gen 15a: Lookup Table Enhanced PSO Optimizer
Performance mutation: Replace bit-wise operations with precomputed lookup tables
for ultra-fast Sobol sequence generation and enhanced convergence.
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

class ApexSobolTurbo:
    """
    Turbo Sobol sequence generator with precomputed lookup tables:
    - Massive 65536-entry lookup table for all bit patterns
    - Precomputed XOR results for all common operations
    - Vectorized point generation using lookup tables
    - Zero runtime bit manipulation overhead
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        # Enhanced direction numbers
        self.direction_numbers = self._get_direction_numbers(dimension)
        # MASSIVE lookup table for all 16-bit patterns
        self.bit_lookup_table = self._precompute_massive_bit_lookup()
        # XOR operation lookup table
        self.xor_lookup_table = self._precompute_xor_lookup()
        # Gray code lookup table (expanded)
        self.gray_lookup_table = self._precompute_gray_lookup(65536)
        # Point value lookup table for ultra-fast generation
        self.point_value_cache = self._precompute_point_value_cache()

    def _get_direction_numbers(self, dim: int) -> np.ndarray:
        """Enhanced direction numbers computation."""
        directions = np.zeros((dim, 32), dtype=np.uint32)

        # First dimension (van der Corput sequence base 2)
        for i in range(32):
            directions[0, i] = 1 << (31 - i)

        # Subsequent dimensions with enhanced robustness
        for d in range(1, min(dim, 10)):
            directions[d, 0] = 1 << 31
            for i in range(1, 32):
                directions[d, i] = directions[d, i-1] ^ (directions[d, i-1] >> 1)

        return directions

    def _precompute_massive_bit_lookup(self) -> np.ndarray:
        """Precompute lookup table for all 16-bit patterns for zero-overhead bit ops."""
        # Create massive lookup table: 65536 entries covering all 16-bit patterns
        lookup_table = np.zeros((65536, 16), dtype=np.uint32)

        for i in range(65536):
            for bit_pos in range(16):
                if i & (1 << bit_pos):
                    lookup_table[i, bit_pos] = 1
                else:
                    lookup_table[i, bit_pos] = 0

        return lookup_table

    def _precompute_xor_lookup(self) -> np.ndarray:
        """Precompute XOR lookup table for common direction number operations."""
        # Create lookup table for XOR operations with direction numbers
        xor_table = np.zeros((256, 8), dtype=np.uint32)

        for i in range(256):
            for bit_pos in range(8):
                if i & (1 << bit_pos):
                    xor_table[i, bit_pos] = 1

        return xor_table

    def _precompute_gray_lookup(self, max_n: int) -> np.ndarray:
        """Precompute massive Gray code lookup table."""
        gray_codes = np.zeros(max_n, dtype=np.uint32)
        for i in range(max_n):
            gray_codes[i] = i ^ (i >> 1)
        return gray_codes

    def _precompute_point_value_cache(self) -> Dict:
        """Precompute common point value calculations for ultra-fast lookup."""
        cache = {}

        # Cache common Gray code to point value mappings for first 1024 values
        for gray in range(1024):
            for dim in range(self.dimension):
                dir_nums = self.direction_numbers[dim]
                point_val = 0

                # Use lookup table for bit operations
                bit_pattern = gray & 0xFFFF  # Take first 16 bits
                if bit_pattern < len(self.bit_lookup_table):
                    bits = self.bit_lookup_table[bit_pattern]

                    # Compute XOR using lookup
                    for bit_pos in range(min(16, len(dir_nums))):
                        if bits[bit_pos]:
                            point_val ^= dir_nums[bit_pos]

                # Handle remaining bits if any
                remaining_gray = gray >> 16
                bit_pos = 16
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                cache_key = (gray, dim)
                cache[cache_key] = point_val / (1 << 32)

        return cache

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with turbo lookup table optimization."""
        points = np.zeros((n, self.dimension))

        # Use massive Gray code lookup table
        if n <= len(self.gray_lookup_table):
            gray_codes = self.gray_lookup_table[:n]
        else:
            gray_codes = np.array([i ^ (i >> 1) for i in range(n)], dtype=np.uint32)

        # Ultra-fast point generation using lookup tables
        for d in range(self.dimension):
            dir_nums = self.direction_numbers[d]

            for i in range(n):
                gray = gray_codes[i]

                # Try cache lookup first for maximum speed
                cache_key = (gray, d)
                if cache_key in self.point_value_cache:
                    points[i, d] = self.point_value_cache[cache_key]
                    continue

                point_val = 0

                # Use massive bit lookup table for first 16 bits
                bit_pattern = gray & 0xFFFF
                if bit_pattern < len(self.bit_lookup_table):
                    bits = self.bit_lookup_table[bit_pattern]

                    # Ultra-fast XOR using precomputed lookup
                    for bit_pos in range(min(16, len(dir_nums))):
                        if bits[bit_pos]:
                            point_val ^= dir_nums[bit_pos]

                # Handle remaining bits with optimized approach
                remaining_gray = gray >> 16
                bit_pos = 16
                while remaining_gray and bit_pos < 32:
                    if remaining_gray & 1:
                        point_val ^= dir_nums[bit_pos]
                    remaining_gray >>= 1
                    bit_pos += 1

                points[i, d] = point_val / (1 << 32)

        return points

class ApexPSO:
    """
    Enhanced Particle Swarm Optimization with improved convergence parameters.
    """

    def __init__(self, swarm_size=10, w=0.65, c1=1.5, c2=1.5):
        self.swarm_size = swarm_size  # Slightly larger swarm
        self.w = w      # Reduced inertia for faster convergence
        self.c1 = c1    # Enhanced cognitive component
        self.c2 = c2    # Enhanced social component

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 45, tolerance: float = 8e-5) -> Tuple[np.ndarray, float]:
        """PSO optimization with enhanced parameters."""
        n_dims = len(x0)

        # Initialize swarm positions around x0
        positions = np.zeros((self.swarm_size, n_dims))
        positions[0] = x0  # Best guess as first particle

        # Initialize other particles with refined perturbations
        for i in range(1, self.swarm_size):
            positions[i] = x0 + np.random.normal(0, 0.045, n_dims)

        # Initialize velocities with better range
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
                # Update velocity with enhanced parameters
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

            # Enhanced convergence check
            if no_improvement_count > 18:  # Adjusted early stopping
                break

            # Check tolerance
            fitness_range = np.max(personal_best_fitness) - np.min(personal_best_fitness)
            if fitness_range < tolerance:
                break

        return global_best_position, global_best_fitness

class ApexCSTInitializer:
    """
    Lookup table enhanced CST initializer with turbo Sobol generation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Enhanced parameter bounds
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
        """Enhanced objective function with caching."""
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
                    penalty += 18 * (0.06 - max_t)  # Increased penalty
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
        """Convert Sobol points using cached vectorized operations."""
        n_points = sobol_points.shape[0]

        # Use cached bounds arrays for maximum speed
        upper_coeffs_all = self.upper_bounds_cache + sobol_points[:, :5] * self.upper_ranges_cache
        lower_coeffs_all = self.lower_bounds_cache + sobol_points[:, 5:10] * self.lower_ranges_cache

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 42) -> List[Airfoil]:
        """Generate turbo Sobol population using lookup table optimization."""
        sobol = ApexSobolTurbo(dimension=10)  # Use turbo version
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_apex_gaussian_population(self, size: int = 30) -> List[Airfoil]:
        """Generate enhanced Gaussian population."""
        population = []
        n_coeffs = 5

        # Strategy 1: Enhanced standard distribution
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

        # Strategy 2: High-performance biased
        strategy2_size = size // 4
        upper_biased = self.rng.normal(0.205, 0.075, (strategy2_size, n_coeffs))
        lower_biased = self.rng.normal(-0.078, 0.040, (strategy2_size, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(strategy2_size):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thickness-optimized
        strategy3_size = size // 4
        thick_upper = self.rng.normal(0.225, 0.055, (strategy3_size, n_coeffs))
        thick_lower = self.rng.normal(-0.145, 0.048, (strategy3_size, n_coeffs))
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
        sym_upper = self.rng.normal(0.140, 0.040, (strategy4_size, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(strategy4_size):
            upper_list = sym_upper[i].tolist()
            # Enhanced asymmetry factor
            lower_list = [-c * self.rng.uniform(0.92, 1.08) for c in upper_list]
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

        # Sort and select top candidates
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = ApexPSO(swarm_size=10, w=0.65, c1=1.5, c2=1.5)  # Enhanced params

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # PSO optimization with improved parameters
                opt_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    coeffs,
                    maxiter=40,
                    tolerance=8e-5
                )
                optimized.append(self.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def apex_adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 24) -> List[Airfoil]:
        """Enhanced adaptive refinement using turbo Sobol."""
        refined = []
        sobol = ApexSobolTurbo(dimension=10)  # Use turbo version

        for airfoil in good_airfoils[:8]:  # Top 8
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 8)

            # Enhanced adaptive scale
            base_scale = 0.088
            scale = base_scale * (1.0 + 0.10 * good_airfoils.index(airfoil))

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
    """Generate turbo lookup table enhanced population."""
    print("Generating lookup table enhanced turbo population...")

    initializer = ApexCSTInitializer(seed=777777)  # Unique seed for this generation

    # Stage 1: Generate enhanced base population
    sobol_pop = initializer.generate_sobol_population(42)      # Increased size
    gaussian_pop = initializer.generate_apex_gaussian_population(30)  # Increased size

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates with enhanced PSO
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=16)

    # Stage 3: Enhanced adaptive refinement
    refined_pop = initializer.apex_adaptive_refinement(optimized_pop[:8], refinement_size=24)

    # Combine all populations with intelligent deduplication
    all_airfoils = base_population + optimized_pop + refined_pop

    # Enhanced deduplication with tighter tolerance
    unique_population = []
    tolerance = 6e-5

    for airfoil in all_airfoils:
        is_duplicate = False
        for existing in unique_population:
            if (np.allclose(airfoil.upper_coeffs, existing.upper_coeffs, atol=tolerance) and
                np.allclose(airfoil.lower_coeffs, existing.lower_coeffs, atol=tolerance)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_population.append(airfoil)

    print(f"Generated {len(unique_population)} unique airfoils using lookup table enhanced approach")
    return unique_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Enhanced population evaluation with optimized batching."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Optimized batch processing
    batch_size = 16
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]

        for airfoil in batch:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            results.append((airfoil, fitness))

    return results

def main():
    """Main lookup table enhanced initialization."""
    print("Generating lookup table enhanced population with turbo Sobol generation...")

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
        filename = f"results/turbo_enhanced_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['turbo_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "15a"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "turbo_rank": i
        })

    # Enhanced diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} lookup table enhanced airfoils")
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