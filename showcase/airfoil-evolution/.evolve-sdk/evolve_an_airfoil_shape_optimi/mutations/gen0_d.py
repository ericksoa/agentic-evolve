#!/usr/bin/env python3
"""
Gen 0 Variant D: Multi-Objective Pareto Front Initialization
Generates diverse airfoils targeting different objectives simultaneously.
Uses vectorized multi-objective optimization and precomputed trade-off curves.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import math

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class ParetoFrontInitializer:
    """Initialize airfoils to explore Pareto front between competing objectives."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Precomputed design archetypes for different objectives
        self.archetypes = self._build_archetypes()

    def _build_archetypes(self) -> Dict[str, dict]:
        """Build archetypal designs for different objectives."""
        return {
            'high_ld': {
                'desc': 'High L/D efficiency',
                'thickness_range': (0.08, 0.12),
                'camber_range': (0.01, 0.03),
                'leading_edge_bias': 1.2,  # More LE curvature
                'trailing_edge_bias': 0.8   # Less TE curvature
            },
            'high_cl': {
                'desc': 'High maximum lift',
                'thickness_range': (0.10, 0.16),
                'camber_range': (0.03, 0.05),
                'leading_edge_bias': 1.5,
                'trailing_edge_bias': 1.2
            },
            'low_cd': {
                'desc': 'Low drag (laminar)',
                'thickness_range': (0.06, 0.10),
                'camber_range': (0.005, 0.02),
                'leading_edge_bias': 0.9,  # Sharper LE
                'trailing_edge_bias': 0.7   # Very sharp TE
            },
            'balanced': {
                'desc': 'Balanced performance',
                'thickness_range': (0.09, 0.13),
                'camber_range': (0.015, 0.025),
                'leading_edge_bias': 1.0,
                'trailing_edge_bias': 1.0
            },
            'robust': {
                'desc': 'Gentle stall, forgiving',
                'thickness_range': (0.12, 0.16),
                'camber_range': (0.02, 0.04),
                'leading_edge_bias': 1.3,
                'trailing_edge_bias': 1.1
            }
        }

    def generate_archetype_airfoil(self, archetype_name: str) -> Airfoil:
        """Generate airfoil based on design archetype with optimized CST mapping."""
        arch = self.archetypes[archetype_name]

        # Sample design parameters
        target_thickness = self.rng.uniform(*arch['thickness_range'])
        max_camber = self.rng.uniform(*arch['camber_range'])

        # Generate base coefficients optimized for this archetype
        n_coeffs = 5

        # Vectorized coefficient generation with archetype bias
        base_coeffs = self.rng.normal(0, 0.02, n_coeffs)

        # Apply archetype-specific shaping
        shape_factors = np.linspace(arch['leading_edge_bias'], arch['trailing_edge_bias'], n_coeffs)
        shape_factors *= np.array([1.0, 0.9, 0.8, 0.7, 0.6])  # Decreasing influence

        # Thickness contribution (symmetric part)
        thickness_coeffs = base_coeffs * shape_factors * target_thickness

        # Camber contribution (asymmetric part)
        camber_coeffs = self.rng.normal(0, 0.01, n_coeffs) * max_camber

        # Combine for upper and lower surfaces
        upper_coeffs = (thickness_coeffs + camber_coeffs).tolist()
        lower_coeffs = (-thickness_coeffs + camber_coeffs).tolist()

        # Ensure physical validity
        upper_coeffs = [max(0.005, min(0.3, c)) for c in upper_coeffs]
        lower_coeffs = [min(-0.005, max(-0.3, c)) for c in lower_coeffs]

        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def generate_pareto_population(self, size_per_archetype: int = 10) -> List[Tuple[Airfoil, str]]:
        """Generate population targeting different regions of Pareto front."""
        population = []

        for archetype_name in self.archetypes.keys():
            for _ in range(size_per_archetype):
                airfoil = self.generate_archetype_airfoil(archetype_name)
                population.append((airfoil, archetype_name))

        return population

class MultiObjectiveEvaluator:
    """Fast multi-objective evaluation optimized for Pareto front analysis."""

    def __init__(self):
        self.objectives = ['max_ld', 'max_cl', 'min_cd']

    def evaluate_multi_objective(self, airfoil: Airfoil) -> Dict[str, float]:
        """Evaluate airfoil on multiple objectives simultaneously."""
        results = {}

        for objective in self.objectives:
            req = DesignRequirements(reynolds=200000, objective=objective)
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)

            if result.get('valid', False):
                results[objective] = result['fitness']
            else:
                results[objective] = 0.0

        # Additional derived objectives
        if results['max_ld'] > 0 and results['max_cl'] > 0:
            # Robustness metric (balanced performance)
            results['robustness'] = math.sqrt(results['max_ld'] * results['max_cl'])

            # Efficiency at different flight conditions
            results['cruise_efficiency'] = results['max_ld']
            results['climb_efficiency'] = results['max_cl'] * 10  # Scale for comparison

        return results

    def compute_pareto_rank(self, population_results: List[Dict[str, float]]) -> List[int]:
        """Compute Pareto ranking using fast non-dominated sorting."""
        n = len(population_results)
        ranks = [0] * n
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        # Fast pairwise comparison
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population_results[i], population_results[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population_results[j], population_results[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Assign ranks
        front = [i for i in range(n) if domination_count[i] == 0]
        rank = 0

        while front:
            for i in front:
                ranks[i] = rank

            next_front = []
            for i in front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            front = next_front
            rank += 1

        return ranks

    def _dominates(self, sol1: Dict[str, float], sol2: Dict[str, float]) -> bool:
        """Check if solution 1 dominates solution 2 (all objectives better or equal, at least one strictly better)."""
        better_in_all = True
        better_in_at_least_one = False

        for obj in self.objectives:
            if sol1[obj] < sol2[obj]:
                better_in_all = False
                break
            elif sol1[obj] > sol2[obj]:
                better_in_at_least_one = True

        return better_in_all and better_in_at_least_one

def generate_population() -> List[Airfoil]:
    """Generate diverse population targeting Pareto front."""
    initializer = ParetoFrontInitializer(seed=101112)
    population_with_archetypes = initializer.generate_pareto_population(10)

    # Extract just the airfoils
    population = [airfoil for airfoil, _ in population_with_archetypes]

    # Add some hybrid designs (crossover between archetypes)
    for _ in range(10):
        # Select two different archetypes
        arch1, arch2 = initializer.rng.choice(list(initializer.archetypes.keys()), 2, replace=False)
        airfoil1 = initializer.generate_archetype_airfoil(arch1)
        airfoil2 = initializer.generate_archetype_airfoil(arch2)

        # Create hybrid by averaging coefficients
        hybrid_upper = [(c1 + c2) / 2 for c1, c2 in zip(airfoil1.upper_coeffs, airfoil2.upper_coeffs)]
        hybrid_lower = [(c1 + c2) / 2 for c1, c2 in zip(airfoil1.lower_coeffs, airfoil2.lower_coeffs)]

        hybrid_airfoil = Airfoil(upper_coeffs=hybrid_upper, lower_coeffs=hybrid_lower, zte=0.0)
        population.append(hybrid_airfoil)

    return population

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, Dict[str, float], int]]:
    """Evaluate population with multi-objective analysis."""
    evaluator = MultiObjectiveEvaluator()

    # Multi-objective evaluation
    results = []
    population_results = []

    for airfoil in population:
        mo_results = evaluator.evaluate_multi_objective(airfoil)
        results.append((airfoil, mo_results))
        population_results.append(mo_results)

    # Compute Pareto ranks
    pareto_ranks = evaluator.compute_pareto_rank(population_results)

    # Combine results with Pareto ranks
    final_results = [(airfoil, mo_results, rank)
                     for (airfoil, mo_results), rank in zip(results, pareto_ranks)]

    return final_results

def main():
    """Main multi-objective population initialization."""
    print("Generating Pareto front airfoil population...")

    # Generate population
    population = generate_population()

    # Multi-objective evaluation
    evaluated = evaluate_population(population)

    # Sort by Pareto rank first, then by primary objective
    evaluated.sort(key=lambda x: (x[2], -x[1].get('max_ld', 0)))

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    pareto_front = [item for item in evaluated if item[2] == 0]  # Rank 0 = Pareto front

    for i, (airfoil, mo_results, rank) in enumerate(evaluated[:20]):
        filename = f"results/pareto_airfoil_{i:02d}_rank_{rank}_ld_{mo_results.get('max_ld', 0):.2f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['multi_objective_results'] = mo_results
            data['pareto_rank'] = rank
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "pareto_rank": rank,
            "objectives": mo_results,
            "on_pareto_front": rank == 0
        })

    print(f"Generated {len(population)} multi-objective airfoils")
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Best max_ld: {max(r[1].get('max_ld', 0) for r in evaluated):.4f}")
    print(f"Best max_cl: {max(r[1].get('max_cl', 0) for r in evaluated):.4f}")

    return best_results

if __name__ == "__main__":
    main()