#!/usr/bin/env python3
"""
Gen 2 Crossover Hybrid: Multi-Strategy Population Initializer
Combines Sobol space-filling, optimization bootstrap, and diverse strategies.
Integrates best features from parents C (Sobol), G (optimization), and A (diversity).
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastSobol:
    """Fast Sobol sequence generator from Parent C - optimal space coverage."""

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

class FastNelderMead:
    """Simplified Nelder-Mead optimizer from Parent G - for local refinement."""

    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def optimize(self, objective_func: Callable, x0: np.ndarray,
                maxiter: int = 30, tolerance: float = 1e-3) -> Tuple[np.ndarray, float]:
        """Fast Nelder-Mead optimization with early termination."""
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

        # Evaluate initial simplex
        f_values = np.array([objective_func(x) for x in simplex])

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

class HybridPopulationInitializer:
    """Hybrid initializer combining Sobol, optimization, and diverse strategies."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator_cache = {}

        # Bounds for CST coefficients
        self.bounds = {
            'upper_coeffs': [(0.05, 0.3), (0.04, 0.25), (0.03, 0.2), (0.02, 0.15), (0.01, 0.1)],
            'lower_coeffs': [(-0.25, -0.04), (-0.2, -0.03), (-0.15, -0.02), (-0.12, -0.01), (-0.1, -0.005)]
        }

    def coeffs_to_airfoil(self, coeffs: np.ndarray) -> Airfoil:
        """Convert coefficient vector to Airfoil object."""
        upper_coeffs = np.clip(coeffs[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(coeffs[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def objective_function(self, coeffs: np.ndarray) -> float:
        """Objective function for optimization (minimization, so negate fitness)."""
        coeffs_key = tuple(np.round(coeffs, 6))

        if coeffs_key in self.evaluator_cache:
            return self.evaluator_cache[coeffs_key]

        try:
            airfoil = self.coeffs_to_airfoil(coeffs)
            req = DesignRequirements(reynolds=200000, objective="max_ld")
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                fitness = result['fitness']
                # Add penalty for constraint violations (from Parent G)
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

    def sobol_to_cst_params(self, sobol_points: np.ndarray) -> List[np.ndarray]:
        """Convert Sobol points to CST parameters using vectorized operations (from Parent C)."""
        n_points = sobol_points.shape[0]
        upper_bounds = np.array([b for b, _ in self.bounds['upper_coeffs']])
        upper_ranges = np.array([r - l for l, r in self.bounds['upper_coeffs']])
        lower_bounds = np.array([b for b, _ in self.bounds['lower_coeffs']])
        lower_ranges = np.array([r - l for l, r in self.bounds['lower_coeffs']])

        # Vectorized transformation
        upper_coeffs_all = upper_bounds + sobol_points[:, :5] * upper_ranges
        lower_coeffs_all = lower_bounds + sobol_points[:, 5:10] * lower_ranges

        # Return as coefficient vectors for optimization
        return [np.concatenate([upper_coeffs_all[i], lower_coeffs_all[i]])
                for i in range(n_points)]

    def generate_sobol_population(self, size: int = 25) -> List[Airfoil]:
        """Generate Sobol-based quasi-random population (from Parent C)."""
        sobol = FastSobol(dimension=10)
        points = sobol.generate_points(size)
        cst_coeffs = self.sobol_to_cst_params(points)

        return [self.coeffs_to_airfoil(coeffs) for coeffs in cst_coeffs]

    def generate_diverse_population(self, size: int = 15) -> List[Airfoil]:
        """Generate diverse population with different strategies (from Parent A)."""
        airfoils = []

        # Standard Gaussian
        for _ in range(size // 3):
            upper = np.clip(self.rng.normal(0.15, 0.05, 5), 0.05, 0.25)
            lower = np.clip(self.rng.normal(-0.12, 0.04, 5), -0.25, -0.05)
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(),
                                  lower_coeffs=lower.tolist(), zte=0.0))

        # High-camber biased
        for _ in range(size // 3):
            upper = np.clip(self.rng.normal(0.18, 0.04, 5), 0.08, 0.28)
            lower = np.clip(self.rng.normal(-0.10, 0.03, 5), -0.20, -0.03)
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(),
                                  lower_coeffs=lower.tolist(), zte=0.0))

        # Thick airfoils
        for _ in range(size - len(airfoils)):
            upper = np.clip(self.rng.normal(0.20, 0.06, 5), 0.10, 0.30)
            lower = np.clip(self.rng.normal(-0.15, 0.05, 5), -0.30, -0.05)
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(),
                                  lower_coeffs=lower.tolist(), zte=0.0))

        return airfoils

    def optimize_promising_designs(self, airfoils: List[Airfoil], top_n: int = 10) -> List[Airfoil]:
        """Optimize top designs using Nelder-Mead (from Parent G)."""
        # Quick evaluation to find promising designs
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        for airfoil in airfoils:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            evaluated.append((airfoil, fitness))

        # Sort and select top designs
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_airfoils = [a for a, f in evaluated[:top_n] if f > 0]

        # Optimize each promising design
        optimized = []
        optimizer = FastNelderMead()

        for airfoil in top_airfoils:
            initial_coeffs = np.array(airfoil.upper_coeffs + airfoil.lower_coeffs)
            try:
                optimized_coeffs, _ = optimizer.optimize(
                    self.objective_function,
                    initial_coeffs,
                    maxiter=25
                )
                optimized.append(self.coeffs_to_airfoil(optimized_coeffs))
            except Exception:
                # Keep original if optimization fails
                optimized.append(airfoil)

        return optimized

    def adaptive_refinement(self, good_airfoils: List[Airfoil], refinement_size: int = 10) -> List[Airfoil]:
        """Adaptive refinement around good designs (from Parent C)."""
        refined = []

        for airfoil in good_airfoils[:5]:  # Top 5 to refine around
            center_upper = np.array(airfoil.upper_coeffs)
            center_lower = np.array(airfoil.lower_coeffs)

            # Generate local perturbations using multiple strategies
            for _ in range(refinement_size // 5):
                # Strategy 1: Small Gaussian perturbations
                if self.rng.random() < 0.5:
                    scale = 0.08
                    new_upper = center_upper + self.rng.normal(0, scale * np.abs(center_upper))
                    new_lower = center_lower + self.rng.normal(0, scale * np.abs(center_lower))
                else:
                    # Strategy 2: Uniform local sampling
                    scale = 0.1
                    new_upper = center_upper + (self.rng.uniform(-1, 1, 5) * scale * np.abs(center_upper))
                    new_lower = center_lower + (self.rng.uniform(-1, 1, 5) * scale * np.abs(center_lower))

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
    """Generate hybrid population combining all strategies."""
    initializer = HybridPopulationInitializer(seed=202425)

    # Stage 1: Sobol space-filling foundation (25 designs)
    sobol_population = initializer.generate_sobol_population(25)

    # Stage 2: Diverse strategy exploration (15 designs)
    diverse_population = initializer.generate_diverse_population(15)

    # Stage 3: Optimize promising designs from both populations
    combined_initial = sobol_population + diverse_population
    optimized_population = initializer.optimize_promising_designs(combined_initial, top_n=8)

    # Stage 4: Adaptive refinement around best optimized designs
    refined_population = initializer.adaptive_refinement(optimized_population, refinement_size=12)

    # Final population: combine all strategies
    final_population = sobol_population[:15] + diverse_population[:10] + optimized_population + refined_population

    return final_population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population using batch processing."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def main():
    """Main hybrid population initialization."""
    print("Generating hybrid multi-strategy airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):
        filename = f"results/hybrid_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['generation'] = 2
            data['method'] = 'hybrid_crossover'
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "rank": i,
            "method": "hybrid_crossover"
        })

    # Calculate population metrics
    fitnesses = [f for _, f in evaluated if f > 0]

    print(f"Generated {len(population)} hybrid airfoils")
    print(f"Valid designs: {len(fitnesses)}")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Mean fitness: {np.mean(fitnesses):.4f}" if fitnesses else "No valid designs")

    # Calculate diversity
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    diversity = np.mean(np.std(all_coeffs, axis=0))
    print(f"Population diversity: {diversity:.4f}")

    return best_results

if __name__ == "__main__":
    main()