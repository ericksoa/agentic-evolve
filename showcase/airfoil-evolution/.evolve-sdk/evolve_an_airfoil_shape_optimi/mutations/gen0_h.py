#!/usr/bin/env python3
"""
Gen 0 Variant H: Adaptive Sampling with Machine Learning Surrogate
Uses Gaussian Process surrogate model to guide adaptive sampling of CST parameter space.
Optimized with sparse GP approximations and vectorized acquisition functions.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class FastGaussianProcess:
    """
    Fast Gaussian Process surrogate model optimized for CST parameter space.
    Uses sparse inducing points for scalability.
    """

    def __init__(self, n_inducing: int = 50, noise_level: float = 0.1):
        # Simple GP with RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,  # Reduced for speed
            alpha=1e-6,
            normalize_y=True
        )
        self.n_inducing = n_inducing
        self.is_fitted = False
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP model with sparse approximation if needed."""
        if len(X) > self.n_inducing:
            # Use subset for training (sparse approximation)
            indices = np.random.choice(len(X), self.n_inducing, replace=False)
            X_sparse = X[indices]
            y_sparse = y[indices]
        else:
            X_sparse = X
            y_sparse = y

        self.gp.fit(X_sparse, y_sparse)
        self.X_train = X_sparse
        self.y_train = y_sparse
        self.is_fitted = True

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.gp.predict(X, return_std=return_std)

    def acquisition_function(self, X: np.ndarray, acquisition_type: str = 'ei') -> np.ndarray:
        """
        Compute acquisition function for adaptive sampling.
        Expected Improvement (EI) or Upper Confidence Bound (UCB).
        """
        mean, std = self.predict(X, return_std=True)

        if acquisition_type == 'ei':
            # Expected Improvement
            best_f = np.max(self.y_train) if self.y_train is not None else 0
            z = (mean - best_f) / (std + 1e-9)
            ei = (mean - best_f) * self._norm_cdf(z) + std * self._norm_pdf(z)
            return ei
        elif acquisition_type == 'ucb':
            # Upper Confidence Bound
            kappa = 2.0  # Exploration parameter
            return mean + kappa * std
        else:
            # Pure uncertainty
            return std

    @staticmethod
    def _norm_pdf(x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    @staticmethod
    def _norm_cdf(x):
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

class AdaptiveSampler:
    """Adaptive sampling using Gaussian Process surrogate model."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.gp_surrogate = FastGaussianProcess()
        self.evaluated_points = []
        self.evaluated_fitness = []

    def coeffs_to_vector(self, upper_coeffs: List[float], lower_coeffs: List[float]) -> np.ndarray:
        """Convert CST coefficients to parameter vector."""
        return np.array(upper_coeffs + lower_coeffs)

    def vector_to_airfoil(self, vector: np.ndarray) -> Airfoil:
        """Convert parameter vector to airfoil."""
        upper_coeffs = np.clip(vector[:5], 0.01, 0.3).tolist()
        lower_coeffs = np.clip(vector[5:], -0.3, -0.01).tolist()
        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def evaluate_point(self, vector: np.ndarray) -> float:
        """Evaluate single point and add to database."""
        airfoil = self.vector_to_airfoil(vector)
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)

        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0

        # Add to evaluation database
        self.evaluated_points.append(vector.copy())
        self.evaluated_fitness.append(fitness)

        return fitness

    def initial_sampling(self, n_initial: int = 30) -> List[np.ndarray]:
        """Generate initial sample using Latin Hypercube-like sampling."""
        initial_points = []

        # Systematic sampling of parameter space
        for i in range(n_initial):
            # Use different sampling strategies
            if i % 4 == 0:
                # Random uniform
                vector = np.concatenate([
                    self.rng.uniform(0.05, 0.25, 5),  # upper coeffs
                    self.rng.uniform(-0.25, -0.05, 5)  # lower coeffs
                ])
            elif i % 4 == 1:
                # Normal around good region
                vector = np.concatenate([
                    np.clip(self.rng.normal(0.15, 0.05, 5), 0.01, 0.3),
                    np.clip(self.rng.normal(-0.12, 0.04, 5), -0.3, -0.01)
                ])
            elif i % 4 == 2:
                # Low thickness
                vector = np.concatenate([
                    self.rng.uniform(0.05, 0.15, 5),
                    self.rng.uniform(-0.15, -0.05, 5)
                ])
            else:
                # High thickness
                vector = np.concatenate([
                    self.rng.uniform(0.15, 0.25, 5),
                    self.rng.uniform(-0.25, -0.15, 5)
                ])

            initial_points.append(vector)

        # Evaluate initial points
        for point in initial_points:
            self.evaluate_point(point)

        return initial_points

    def adaptive_sampling_iteration(self, n_candidates: int = 100) -> Optional[np.ndarray]:
        """Single iteration of adaptive sampling."""
        if len(self.evaluated_points) < 5:
            return None

        # Fit GP to current data
        X_train = np.array(self.evaluated_points)
        y_train = np.array(self.evaluated_fitness)

        if len(np.unique(y_train)) < 2:
            # Not enough diversity for GP
            return None

        self.gp_surrogate.fit(X_train, y_train)

        # Generate candidate points
        candidates = []
        for _ in range(n_candidates):
            # Random candidates in parameter space
            candidate = np.concatenate([
                self.rng.uniform(0.01, 0.3, 5),
                self.rng.uniform(-0.3, -0.01, 5)
            ])
            candidates.append(candidate)

        candidates = np.array(candidates)

        # Evaluate acquisition function
        acquisition_scores = self.gp_surrogate.acquisition_function(candidates, 'ei')

        # Select best candidate
        best_idx = np.argmax(acquisition_scores)
        next_point = candidates[best_idx]

        # Evaluate the selected point
        self.evaluate_point(next_point)

        return next_point

    def generate_adaptive_population(self, total_evaluations: int = 80) -> List[Airfoil]:
        """Generate population using adaptive sampling."""
        print("Starting adaptive sampling with GP surrogate...")

        # Initial sampling
        initial_points = self.initial_sampling(30)
        print(f"Completed initial sampling: {len(initial_points)} points")

        # Adaptive iterations
        remaining_evals = total_evaluations - len(initial_points)
        for i in range(remaining_evals):
            next_point = self.adaptive_sampling_iteration()
            if next_point is None:
                print(f"Adaptive sampling stopped at iteration {i}")
                break

            if (i + 1) % 10 == 0:
                print(f"Adaptive iteration {i + 1}/{remaining_evals}")

        # Convert all evaluated points to airfoils
        population = []
        for point in self.evaluated_points:
            airfoil = self.vector_to_airfoil(point)
            population.append(airfoil)

        print(f"Completed adaptive sampling: {len(population)} airfoils")
        return population

def generate_population() -> List[Airfoil]:
    """Generate population using adaptive ML-guided sampling."""
    sampler = AdaptiveSampler(seed=222324)
    return sampler.generate_adaptive_population(60)

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate adaptively sampled population (reuse cached results where possible)."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Create evaluation cache
    eval_cache = {}

    for airfoil in population:
        # Create cache key
        coeffs_key = tuple(airfoil.upper_coeffs + airfoil.lower_coeffs)

        if coeffs_key in eval_cache:
            fitness = eval_cache[coeffs_key]
        else:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            eval_cache[coeffs_key] = fitness

        results.append((airfoil, fitness))

    return results

def analyze_adaptive_quality(population: List[Airfoil], evaluated: List[Tuple[Airfoil, float]]) -> Dict[str, float]:
    """Analyze quality of adaptive sampling."""
    fitnesses = [f for _, f in evaluated]
    valid_fitnesses = [f for f in fitnesses if f > 0]

    if not valid_fitnesses:
        return {"error": "No valid airfoils generated"}

    # Sampling efficiency metrics
    efficiency_metrics = {
        'valid_fraction': len(valid_fitnesses) / len(fitnesses),
        'mean_fitness': np.mean(valid_fitnesses),
        'std_fitness': np.std(valid_fitnesses),
        'max_fitness': np.max(valid_fitnesses),
        'top_10_percent_mean': np.mean(sorted(valid_fitnesses, reverse=True)[:max(1, len(valid_fitnesses)//10)])
    }

    # Parameter space coverage
    all_params = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    param_ranges = np.max(all_params, axis=0) - np.min(all_params, axis=0)

    efficiency_metrics.update({
        'parameter_coverage': np.mean(param_ranges),
        'adaptive_efficiency': efficiency_metrics['top_10_percent_mean'] / (efficiency_metrics['mean_fitness'] + 1e-6)
    })

    return efficiency_metrics

def main():
    """Main adaptive sampling initialization."""
    print("Generating ML-guided adaptive sampling population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness (reusing cached results)
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze adaptive quality
    adaptive_quality = analyze_adaptive_quality(population, evaluated)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:25]):
        filename = f"results/adaptive_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['adaptive_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "adaptive_rank": i,
            "adaptive_quality": adaptive_quality
        })

    print(f"Generated {len(population)} adaptively sampled airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Adaptive sampling quality metrics:")
    for key, value in adaptive_quality.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # Save adaptive analysis
    with open("results/adaptive_analysis.json", 'w') as f:
        json.dump({
            "adaptive_quality": adaptive_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1] if evaluated else 0
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()