#!/usr/bin/env python3
"""
Gen 5 Variant b: Vectorized Batch Evaluation Performance Optimization
Mutation: Added vectorized batch evaluation to eliminate sequential bottlenecks.
Focus: Significant performance improvement through parallel candidate evaluation.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastSobol:
    """
    Fast Sobol sequence generator optimized for CST coefficient generation.
    Inherited from gen0_c.py for optimal space-filling design coverage.
    """

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self.count = 0
        self.direction_numbers = self._get_direction_numbers(dimension)

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

    def generate_points(self, n: int) -> np.ndarray:
        """Generate n Sobol points with vectorized operations."""
        points = np.zeros((n, self.dimension))

        for i in range(n):
            gray = i ^ (i >> 1)
            for d in range(self.dimension):
                point_val = 0
                for j in range(32):
                    if gray & (1 << j):
                        point_val ^= self.direction_numbers[d, j]
                points[i, d] = point_val / (1 << 32)

        return points

class VectorizedBatchEvaluator:
    """
    NEW: Vectorized batch evaluator for high-performance parallel candidate assessment.
    MUTATION: Replaces sequential evaluation with parallel batch processing.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.evaluator_cache = {}

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Convert coefficient vector to Airfoil object."""
        upper_coeffs = np.clip(coeffs[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(coeffs[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def _single_evaluation(self, coeffs: np.ndarray) -> float:
        """Single evaluation function for parallel execution."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Add penalty for constraint violations
                penalty = 0
                max_t, _ = airfoil.max_thickness()
                if max_t < 0.06:
                    penalty += 10 * (0.06 - max_t)
                if max_t > 0.18:
                    penalty += 10 * (max_t - 0.18)

                objective_value = -fitness + penalty
            else:
                objective_value = 1000

            self.evaluator_cache[coeffs_key] = objective_value
            return objective_value

        except Exception:
            return 1000

    def evaluate_batch(self, coeffs_batch: List[np.ndarray]) -> List[float]:
        """
        PERFORMANCE MUTATION: Vectorized batch evaluation using ThreadPoolExecutor.
        Evaluates multiple candidates in parallel instead of sequentially.
        """
        if len(coeffs_batch) == 1:
            return [self._single_evaluation(coeffs_batch[0])]

        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluations
            futures = [executor.submit(self._single_evaluation, coeffs) for coeffs in coeffs_batch]

            # Collect results in order
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception:
                    results.append(1000)  # Fallback for failed evaluations

        return results

class FastNelderMead:
    """
    Enhanced Nelder-Mead optimizer with batch evaluation support.
    MUTATION: Updated to use vectorized batch evaluator for performance.
    """

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha  # Reflection
        self.gamma = gamma  # Expansion
        self.rho = rho      # Contraction
        self.sigma = sigma  # Shrink

    def optimize(self, batch_evaluator: VectorizedBatchEvaluator, x0: np.ndarray,
                maxiter: int = 30, tolerance: float = 1e-3) -> Tuple[np.ndarray, float]:
        """Fast Nelder-Mead optimization with batch evaluation."""
        n = len(x0)

        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        # Create initial simplex with adaptive step size
        step_size = 0.05 * np.abs(x0)
        step_size = np.where(step_size < 0.01, 0.01, step_size)

        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1, i] += step_size[i]

        # Batch evaluate initial simplex - PERFORMANCE IMPROVEMENT
        f_values = np.array(batch_evaluator.evaluate_batch([x for x in simplex]))

        for iteration in range(maxiter):
            # Sort simplex by function values
            indices = np.argsort(f_values)
            simplex = simplex[indices]
            f_values = f_values[indices]

            # Check convergence
            if f_values[-1] - f_values[0] < tolerance:
                break

            # Centroid of n best points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            reflected = centroid + self.alpha * (centroid - simplex[-1])

            # Batch evaluation for all candidates in this iteration - PERFORMANCE IMPROVEMENT
            candidates = [reflected]
            candidate_labels = ['reflected']

            # Prepare expansion candidate
            expanded = centroid + self.gamma * (reflected - centroid)
            candidates.append(expanded)
            candidate_labels.append('expanded')

            # Prepare contraction candidates
            contracted_out = centroid + self.rho * (reflected - centroid)
            contracted_in = centroid + self.rho * (simplex[-1] - centroid)
            candidates.extend([contracted_out, contracted_in])
            candidate_labels.extend(['contracted_out', 'contracted_in'])

            # Batch evaluate all candidates at once
            candidate_values = batch_evaluator.evaluate_batch(candidates)
            f_reflected = candidate_values[0]
            f_expanded = candidate_values[1]
            f_contracted_out = candidate_values[2]
            f_contracted_in = candidate_values[3]

            if f_values[0] <= f_reflected < f_values[-2]:
                simplex[-1] = reflected
                f_values[-1] = f_reflected
                continue

            if f_reflected < f_values[0]:
                # Use expansion if better
                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    f_values[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    f_values[-1] = f_reflected
                continue

            # Choose best contraction
            if f_reflected < f_values[-1]:
                if f_contracted_out < min(f_reflected, f_values[-1]):
                    simplex[-1] = contracted_out
                    f_values[-1] = f_contracted_out
                    continue
            else:
                if f_contracted_in < min(f_reflected, f_values[-1]):
                    simplex[-1] = contracted_in
                    f_values[-1] = f_contracted_in
                    continue

            # Shrink - batch evaluate all shrunk points
            shrink_points = []
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])
                shrink_points.append(simplex[i])

            shrink_values = batch_evaluator.evaluate_batch(shrink_points)
            for i, val in enumerate(shrink_values):
                f_values[i + 1] = val

        # Return best point
        best_idx = np.argmin(f_values)
        return simplex[best_idx], f_values[best_idx]

class HybridCSTInitializer:
    """
    Hybrid CST initializer with vectorized performance optimizations.
    MUTATION: Integrated batch evaluation throughout the pipeline.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.batch_evaluator = VectorizedBatchEvaluator(max_workers=4)

        # Parameter bounds from parent solutions
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[Tuple[List[float], List[float]]]:
        """Convert Sobol points to CST parameters using vectorized operations."""
        n_points = sobol_points.shape[0]
        upper_bounds = np.array([b for b, _ in self.bounds['upper_coeffs']])
        upper_ranges = np.array([r - l for l, r in self.bounds['upper_coeffs']])
        lower_bounds = np.array([b for b, _ in self.bounds['lower_coeffs']])
        lower_ranges = np.array([r - l for l, r in self.bounds['lower_coeffs']])

        # Vectorized transformation
        upper_coeffs_all = upper_bounds + sobol_points[:, :5] * upper_ranges
        lower_coeffs_all = lower_bounds + sobol_points[:, 5:10] * lower_ranges

        return [(upper_coeffs_all[i].tolist(), lower_coeffs_all[i].tolist())
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 30) -> List[Airfoil]:
        """Generate Sobol-based quasi-random population from gen0_c."""
        sobol = FastSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_params = self.sobol_to_cst_params(points)

        population = [
            Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0)
            for upper, lower in cst_params
        ]

        return population

    def generate_diverse_gaussian_population(self, size: int = 20) -> List[Airfoil]:
        """Generate diverse Gaussian population using multi-strategy approach from gen0_a."""
        population = []

        # Strategy 1: Standard distribution
        n_coeffs = 5
        all_upper = self.rng.normal(0.15, 0.05, (size//4, n_coeffs))
        all_lower = self.rng.normal(-0.12, 0.04, (size//4, n_coeffs))
        all_upper = np.clip(all_upper, 0.05, 0.3)
        all_lower = np.clip(all_lower, -0.25, -0.05)

        for i in range(size//4):
            population.append(Airfoil(
                upper_coeffs=all_upper[i].tolist(),
                lower_coeffs=all_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 2: High-camber biased
        upper_biased = self.rng.normal(0.18, 0.06, (size//4, n_coeffs))
        lower_biased = self.rng.normal(-0.09, 0.03, (size//4, n_coeffs))
        upper_biased = np.clip(upper_biased, 0.05, 0.3)
        lower_biased = np.clip(lower_biased, -0.25, -0.05)

        for i in range(size//4):
            population.append(Airfoil(
                upper_coeffs=upper_biased[i].tolist(),
                lower_coeffs=lower_biased[i].tolist(),
                zte=0.0
            ))

        # Strategy 3: Thick airfoils
        thick_upper = self.rng.normal(0.2, 0.04, (size//4, n_coeffs))
        thick_lower = self.rng.normal(-0.15, 0.04, (size//4, n_coeffs))
        thick_upper = np.clip(thick_upper * 1.2, 0.05, 0.3)
        thick_lower = np.clip(thick_lower * 1.2, -0.3, -0.05)

        for i in range(size//4):
            population.append(Airfoil(
                upper_coeffs=thick_upper[i].tolist(),
                lower_coeffs=thick_lower[i].tolist(),
                zte=0.0
            ))

        # Strategy 4: Symmetric airfoils
        sym_upper = self.rng.normal(0.12, 0.03, (size//4, n_coeffs))
        sym_upper = np.clip(sym_upper, 0.05, 0.25)

        for i in range(size//4):
            upper_list = sym_upper[i].tolist()
            population.append(Airfoil(
                upper_coeffs=upper_list,
                lower_coeffs=[-c for c in upper_list],
                zte=0.0
            ))

        return population

    def optimize_promising_candidates(self, candidates: List[Airfoil], top_n: int = 10) -> List[Airfoil]:
        """
        Optimize promising candidates using batch-optimized Nelder-Mead.
        PERFORMANCE MUTATION: Use vectorized batch evaluation for selection and optimization.
        """
        # Convert candidates to coefficient vectors for batch evaluation
        coeffs_batch = [np.array(airfoil.upper_coeffs + airfoil.lower_coeffs) for airfoil in candidates]

        # VECTORIZED BATCH EVALUATION - Major performance improvement
        objective_values = self.batch_evaluator.evaluate_batch(coeffs_batch)

        # Convert to fitness values and pair with airfoils
        evaluated = []
        for i, (airfoil, obj_val) in enumerate(zip(candidates, objective_values)):
            fitness = -obj_val if obj_val < 1000 else 0.0  # Convert objective back to fitness
            evaluated.append((airfoil, fitness))

        # Sort and select top candidates
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [a for a, f in evaluated[:top_n] if f > 0]

        optimized = []
        optimizer = FastNelderMead()

        for airfoil in top_candidates:
            # Convert to coefficient vector
            coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)

            try:
                # Optimize using batch evaluator
                opt_coeffs, _ = optimizer.optimize(
                    self.batch_evaluator,
                    coeffs,
                    maxiter=25,
                    tolerance=1e-3
                )
                optimized.append(self.batch_evaluator.coeffs_to_airfoil(opt_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 15) -> List[Airfoil]:
        """Adaptive refinement around good points from gen0_c."""
        refined = []
        sobol = FastSobol(dimension=10)

        for airfoil in good_airfoils[:5]:  # Refine around top 5
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations
            local_points = sobol.generate_points(refinement_size // 5)
            scale = 0.08  # 8% neighborhood

            for point in local_points:
                # Perturb around center
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
    """Generate hybrid population combining all three strategies."""
    print("Generating hybrid Sobol-optimized multi-strategy population with vectorized evaluation...")

    initializer = HybridCSTInitializer(seed=424344)

    # Stage 1: Generate diverse base population
    sobol_pop = initializer.generate_sobol_population(30)  # Sobol quasi-random
    gaussian_pop = initializer.generate_diverse_gaussian_population(20)  # Multi-strategy Gaussian

    base_population = sobol_pop + gaussian_pop

    # Stage 2: Optimize promising candidates using batch evaluation
    optimized_pop = initializer.optimize_promising_candidates(base_population, top_n=10)

    # Stage 3: Adaptive refinement around best solutions
    refined_pop = initializer.adaptive_refinement(optimized_pop[:5], refinement_size=15)

    # Combine all populations
    final_population = base_population + optimized_pop + refined_pop

    print(f"Generated {len(final_population)} airfoils using vectorized hybrid approach")
    return final_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """
    Evaluate population fitness using vectorized batch evaluation.
    PERFORMANCE MUTATION: Batch evaluation for entire population.
    """
    batch_evaluator = VectorizedBatchEvaluator(max_workers=4)

    # Convert population to coefficient vectors for batch evaluation
    coeffs_batch = [np.array(airfoil.upper_coeffs + airfoil.lower_coeffs) for airfoil in population]

    # VECTORIZED BATCH EVALUATION - Major performance improvement
    objective_values = batch_evaluator.evaluate_batch(coeffs_batch)

    # Convert to fitness and create results
    results = []
    for airfoil, obj_val in zip(population, objective_values):
        fitness = -obj_val if obj_val < 1000 else 0.0  # Convert objective back to fitness
        results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid initialization with vectorized performance optimizations."""
    print("Generating hybrid population with vectorized batch evaluation optimizations...")

    # Generate population
    population = generate_population()

    # Evaluate fitness using batch evaluation
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:25]):
        filename = f"results/vectorized_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['vectorized_rank'] = i
            data['fitness'] = fitness
            data['generation'] = "5b"
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "vectorized_rank": i
        })

    # Calculate diversity metrics
    valid_fitnesses = [f for _, f in evaluated if f > 0]
    diversity_score = len(set(str(a.upper_coeffs + a.lower_coeffs) for a, _ in evaluated)) / len(evaluated)

    print(f"Generated {len(population)} vectorized hybrid airfoils")
    print(f"Valid designs: {len(valid_fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    if valid_fitnesses:
        print(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"Fitness std: {np.std(valid_fitnesses):.4f}")
    print(f"Population diversity: {diversity_score:.3f}")

    return best_results

if __name__ == "__main__":
    main()