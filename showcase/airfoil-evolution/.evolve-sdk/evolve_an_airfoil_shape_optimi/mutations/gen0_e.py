#!/usr/bin/env python3
"""
Gen 0 Variant E: Ensemble Learning Initialization
Combines multiple initialization strategies and uses ensemble methods for robust starting population.
Optimized with parallel processing simulation and cached evaluations.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import concurrent.futures
from functools import lru_cache
import hashlib

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class CachedEvaluator:
    """Performance-optimized evaluator with caching and batch processing."""

    def __init__(self):
        self._cache = {}

    def _airfoil_hash(self, airfoil: Airfoil) -> str:
        """Create hash key for airfoil caching."""
        coeffs_str = str(airfoil.upper_coeffs + airfoil.lower_coeffs + [airfoil.zte])
        return hashlib.md5(coeffs_str.encode()).hexdigest()[:16]

    def evaluate_cached(self, airfoil: Airfoil, req: DesignRequirements) -> Dict:
        """Cached evaluation for performance."""
        cache_key = (self._airfoil_hash(airfoil), req.reynolds, req.objective)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        self._cache[cache_key] = result
        return result

class EnsembleInitializer:
    """Ensemble of different initialization strategies."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.evaluator = CachedEvaluator()
        self.strategies = self._setup_strategies()

    def _setup_strategies(self) -> Dict[str, Callable]:
        """Setup different initialization strategies."""
        return {
            'random_normal': self._generate_random_normal,
            'latin_hypercube': self._generate_latin_hypercube,
            'evolutionary_inspired': self._generate_evolutionary_inspired,
            'morphing_baseline': self._generate_morphing_baseline,
            'optimization_guided': self._generate_optimization_guided
        }

    def _generate_random_normal(self, size: int) -> List[Airfoil]:
        """Generate using normal distribution (baseline)."""
        airfoils = []
        for _ in range(size):
            upper = self.rng.normal(0.12, 0.04, 5).clip(0.01, 0.3).tolist()
            lower = self.rng.normal(-0.1, 0.03, 5).clip(-0.3, -0.01).tolist()
            airfoils.append(Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0))
        return airfoils

    def _generate_latin_hypercube(self, size: int) -> List[Airfoil]:
        """Generate using Latin Hypercube sampling for good space coverage."""
        # Simple LHS implementation (production would use scipy.stats)
        n_dims = 10  # 5 upper + 5 lower coeffs
        samples = np.zeros((size, n_dims))

        for dim in range(n_dims):
            # Create Latin hypercube for this dimension
            intervals = np.linspace(0, 1, size + 1)
            lower_bounds = intervals[:-1]
            upper_bounds = intervals[1:]

            # Random permutation for LHS
            perm = self.rng.permutation(size)
            for i, p in enumerate(perm):
                samples[i, dim] = self.rng.uniform(lower_bounds[p], upper_bounds[p])

        # Transform to airfoil parameter space
        airfoils = []
        for sample in samples:
            # Map [0,1] to coefficient ranges
            upper = 0.01 + sample[:5] * 0.25  # [0.01, 0.26]
            lower = -0.25 + sample[5:] * 0.2   # [-0.25, -0.05]
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist(), zte=0.0))

        return airfoils

    def _generate_evolutionary_inspired(self, size: int) -> List[Airfoil]:
        """Generate based on known good airfoil families."""
        # Start with known good designs and perturb
        base_designs = [
            ([0.17, 0.16, 0.14, 0.12, 0.10], [-0.17, -0.16, -0.14, -0.12, -0.10]),  # Symmetric
            ([0.20, 0.18, 0.16, 0.13, 0.10], [-0.14, -0.12, -0.10, -0.09, -0.08]),  # Cambered
            ([0.15, 0.14, 0.12, 0.10, 0.08], [-0.12, -0.10, -0.08, -0.06, -0.05]),  # Thinner
            ([0.22, 0.20, 0.18, 0.15, 0.12], [-0.16, -0.14, -0.12, -0.10, -0.08]),  # Thicker
        ]

        airfoils = []
        designs_per_base = size // len(base_designs)

        for upper_base, lower_base in base_designs:
            for _ in range(designs_per_base):
                # Add noise for diversity
                noise_scale = 0.02
                upper = np.array(upper_base) + self.rng.normal(0, noise_scale, 5)
                lower = np.array(lower_base) + self.rng.normal(0, noise_scale, 5)

                # Clamp to valid ranges
                upper = np.clip(upper, 0.01, 0.3)
                lower = np.clip(lower, -0.3, -0.01)

                airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist(), zte=0.0))

        # Fill remainder with random variations
        while len(airfoils) < size:
            base_upper, base_lower = self.rng.choice(len(base_designs))
            upper = np.array(base_designs[0][0]) + self.rng.normal(0, 0.03, 5)
            lower = np.array(base_designs[0][1]) + self.rng.normal(0, 0.03, 5)
            upper = np.clip(upper, 0.01, 0.3)
            lower = np.clip(lower, -0.3, -0.01)
            airfoils.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist(), zte=0.0))

        return airfoils

    def _generate_morphing_baseline(self, size: int) -> List[Airfoil]:
        """Generate by morphing between known good airfoils."""
        # Define endpoint airfoils
        endpoints = [
            Airfoil([0.18, 0.16, 0.14, 0.12, 0.09], [-0.18, -0.16, -0.14, -0.12, -0.09]),  # Thick symmetric
            Airfoil([0.15, 0.13, 0.11, 0.09, 0.07], [-0.10, -0.08, -0.06, -0.05, -0.04]),  # Thin cambered
            Airfoil([0.12, 0.11, 0.10, 0.08, 0.06], [-0.12, -0.11, -0.10, -0.08, -0.06]),  # Minimal
            Airfoil([0.25, 0.22, 0.19, 0.16, 0.13], [-0.20, -0.17, -0.14, -0.12, -0.10]),  # High-lift
        ]

        airfoils = []
        morphs_per_pair = size // (len(endpoints) * (len(endpoints) - 1) // 2)

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                for _ in range(morphs_per_pair):
                    # Linear interpolation with random weight
                    t = self.rng.beta(2, 2)  # Beta distribution for variety

                    upper = [(1-t) * a + t * b for a, b in
                            zip(endpoints[i].upper_coeffs, endpoints[j].upper_coeffs)]
                    lower = [(1-t) * a + t * b for a, b in
                            zip(endpoints[i].lower_coeffs, endpoints[j].lower_coeffs)]

                    airfoils.append(Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0))

        # Fill remainder
        while len(airfoils) < size:
            i, j = self.rng.choice(len(endpoints), 2, replace=False)
            t = self.rng.uniform(0, 1)
            upper = [(1-t) * a + t * b for a, b in
                    zip(endpoints[i].upper_coeffs, endpoints[j].upper_coeffs)]
            lower = [(1-t) * a + t * b for a, b in
                    zip(endpoints[i].lower_coeffs, endpoints[j].lower_coeffs)]
            airfoils.append(Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0))

        return airfoils

    def _generate_optimization_guided(self, size: int) -> List[Airfoil]:
        """Generate using simple optimization-guided sampling."""
        # Start with random samples and perform local optimization
        candidates = self._generate_random_normal(size * 3)  # Over-generate

        # Quick evaluation and ranking
        req = DesignRequirements(reynolds=200000, objective="max_ld")
        evaluated = []

        for airfoil in candidates:
            result = self.evaluator.evaluate_cached(airfoil, req)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            evaluated.append((airfoil, fitness))

        # Sort and select top performers
        evaluated.sort(key=lambda x: x[1], reverse=True)
        selected = [a for a, f in evaluated[:size]]

        return selected

    def generate_ensemble_population(self, total_size: int = 50) -> List[Tuple[Airfoil, str]]:
        """Generate population using ensemble of strategies."""
        # Allocate sizes to each strategy
        strategy_sizes = {
            'random_normal': total_size // 5,
            'latin_hypercube': total_size // 5,
            'evolutionary_inspired': total_size // 5,
            'morphing_baseline': total_size // 5,
            'optimization_guided': total_size // 5
        }

        # Handle remainder
        remainder = total_size - sum(strategy_sizes.values())
        strategy_sizes['optimization_guided'] += remainder

        ensemble_population = []

        for strategy_name, size in strategy_sizes.items():
            if size > 0:
                strategy_func = self.strategies[strategy_name]
                strategy_airfoils = strategy_func(size)
                for airfoil in strategy_airfoils:
                    ensemble_population.append((airfoil, strategy_name))

        return ensemble_population

def generate_population() -> List[Airfoil]:
    """Generate ensemble-based population."""
    initializer = EnsembleInitializer(seed=131415)
    ensemble_pop = initializer.generate_ensemble_population(50)

    # Extract airfoils and shuffle for diversity
    population = [airfoil for airfoil, _ in ensemble_pop]
    np.random.shuffle(population)

    return population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population with ensemble performance metrics."""
    evaluator = CachedEvaluator()
    req = DesignRequirements(reynolds=200000, objective="max_ld")

    results = []
    for airfoil in population:
        result = evaluator.evaluate_cached(airfoil, req)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def analyze_ensemble_diversity(population: List[Airfoil]) -> Dict[str, float]:
    """Analyze diversity metrics of ensemble population."""
    # Coefficient diversity
    all_upper = np.array([a.upper_coeffs for a in population])
    all_lower = np.array([a.lower_coeffs for a in population])

    upper_std = np.mean(np.std(all_upper, axis=0))
    lower_std = np.mean(np.std(all_lower, axis=0))

    # Geometric diversity
    thicknesses = [a.max_thickness()[0] for a in population]
    cambers = [abs(a.max_camber()[0]) for a in population]

    return {
        'coefficient_diversity': (upper_std + lower_std) / 2,
        'thickness_range': max(thicknesses) - min(thicknesses),
        'camber_range': max(cambers) - min(cambers),
        'valid_fraction': sum(1 for a in population if a.is_valid()[0]) / len(population)
    }

def main():
    """Main ensemble population initialization."""
    print("Generating ensemble-based airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze diversity
    diversity = analyze_ensemble_diversity(population)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):
        filename = f"results/ensemble_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['ensemble_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "ensemble_rank": i,
            "diversity_contrib": diversity
        })

    print(f"Generated {len(population)} ensemble airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Diversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.4f}")

    # Save ensemble analysis
    with open("results/ensemble_analysis.json", 'w') as f:
        json.dump({
            "diversity_metrics": diversity,
            "population_size": len(population),
            "best_fitness": evaluated[0][1],
            "fitness_std": np.std([f for _, f in evaluated])
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()