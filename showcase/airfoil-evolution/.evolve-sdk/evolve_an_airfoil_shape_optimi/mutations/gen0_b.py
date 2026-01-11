#!/usr/bin/env python3
"""
Gen 0 Variant B: Physics-Informed Initialization
Uses aerodynamic theory to generate realistic starting airfoils.
Optimized with lookup tables and precomputed basis functions.
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

class PhysicsInformedInitializer:
    """Initialize airfoils based on aerodynamic design principles."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Precomputed lookup tables for performance
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for fast coefficient computation."""
        # Target thickness distributions (precomputed)
        x_vals = np.linspace(0, 1, 101)

        # NACA 4-digit thickness distribution (optimized)
        self.naca_thick_lut = {}
        for t in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
            thick = self._naca_thickness_distribution(x_vals, t)
            # Convert to CST coefficients using least squares (cached)
            coeffs = self._fit_cst_thickness(x_vals, thick)
            self.naca_thick_lut[t] = coeffs

        # Camber line lookup table
        self.camber_lut = {}
        for m, p in [(0.0, 0.4), (0.02, 0.4), (0.04, 0.4), (0.02, 0.3), (0.02, 0.5)]:
            camber = self._naca_camber_line(x_vals, m, p)
            coeffs = self._fit_cst_camber(x_vals, camber)
            self.camber_lut[(m, p)] = coeffs

    def _naca_thickness_distribution(self, x: np.ndarray, t: float) -> np.ndarray:
        """NACA 4-digit thickness distribution (vectorized)."""
        # Using optimized coefficients for NACA
        a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        return t * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)

    def _naca_camber_line(self, x: np.ndarray, m: float, p: float) -> np.ndarray:
        """NACA 4-digit camber line (vectorized)."""
        if m == 0:
            return np.zeros_like(x)

        # Vectorized camber calculation
        camber = np.zeros_like(x)
        front_mask = x <= p
        back_mask = x > p

        # Front part
        if np.any(front_mask):
            camber[front_mask] = m * x[front_mask] * (2*p - x[front_mask]) / (p**2)

        # Back part
        if np.any(back_mask):
            camber[back_mask] = m * (1 - 2*p + 2*p*x[back_mask] - x[back_mask]**2) / ((1-p)**2)

        return camber

    def _fit_cst_thickness(self, x: np.ndarray, thickness: np.ndarray) -> np.ndarray:
        """Fit CST coefficients to thickness distribution using least squares."""
        # Build matrix of Bernstein polynomials (5 coefficients)
        n = 4  # degree
        A = np.zeros((len(x), n+1))
        for i in range(n+1):
            for j, xi in enumerate(x):
                if 0 < xi < 1:
                    class_val = (xi**0.5) * ((1-xi)**1.0)
                    bern_val = math.comb(n, i) * (xi**i) * ((1-xi)**(n-i))
                    A[j, i] = class_val * bern_val

        # Solve least squares (fast using NumPy)
        coeffs, _, _, _ = np.linalg.lstsq(A, thickness, rcond=None)
        return coeffs

    def _fit_cst_camber(self, x: np.ndarray, camber: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit CST coefficients to camber line."""
        coeffs = self._fit_cst_thickness(x, camber)
        # Separate upper and lower coefficients
        upper_coeffs = coeffs.copy()
        lower_coeffs = -coeffs.copy()  # Symmetric camber effect
        return upper_coeffs, lower_coeffs

    def generate_naca_inspired_airfoil(self, target_thickness: float,
                                     max_camber: float, camber_pos: float) -> Airfoil:
        """Generate airfoil inspired by NACA design principles."""
        # Use lookup table for base thickness coefficients
        base_thick = min(self.naca_thick_lut.keys(), key=lambda x: abs(x - target_thickness))
        thick_coeffs = self.naca_thick_lut[base_thick].copy()

        # Scale for exact thickness
        scale = target_thickness / base_thick
        thick_coeffs *= scale

        # Get camber coefficients
        camber_key = min(self.camber_lut.keys(),
                        key=lambda x: abs(x[0] - max_camber) + abs(x[1] - camber_pos))
        upper_camber, lower_camber = self.camber_lut[camber_key]

        # Combine thickness and camber
        upper_coeffs = (thick_coeffs + upper_camber * max_camber).tolist()
        lower_coeffs = (-thick_coeffs + lower_camber * max_camber).tolist()

        return Airfoil(upper_coeffs=upper_coeffs, lower_coeffs=lower_coeffs, zte=0.0)

    def generate_population(self, size: int = 50) -> List[Airfoil]:
        """Generate physics-informed population."""
        population = []

        # Define design space (optimized for drone/RC aircraft)
        thickness_range = np.linspace(0.06, 0.16, 6)
        camber_range = np.linspace(0.0, 0.04, 5)
        camber_pos_range = np.linspace(0.3, 0.5, 4)

        # Generate systematic combinations
        for i in range(size):
            # Sample from design space
            t = self.rng.choice(thickness_range)
            m = self.rng.choice(camber_range)
            p = self.rng.choice(camber_pos_range)

            # Add some noise for diversity
            t += self.rng.normal(0, 0.005)
            m += self.rng.normal(0, 0.002)
            p += self.rng.normal(0, 0.02)

            # Clamp to reasonable bounds
            t = np.clip(t, 0.06, 0.18)
            m = np.clip(m, 0.0, 0.05)
            p = np.clip(p, 0.25, 0.55)

            airfoil = self.generate_naca_inspired_airfoil(t, m, p)
            population.append(airfoil)

        return population

def generate_population() -> List[Airfoil]:
    """Generate diverse physics-informed population."""
    initializer = PhysicsInformedInitializer(seed=456)
    return initializer.generate_population(50)

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate population with caching for repeated designs."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    # Cache for identical designs (physics-based can generate duplicates)
    eval_cache = {}

    for airfoil in population:
        # Simple hash based on coefficients
        coeffs_tuple = (tuple(airfoil.upper_coeffs), tuple(airfoil.lower_coeffs))

        if coeffs_tuple in eval_cache:
            fitness = eval_cache[coeffs_tuple]
        else:
            result = evaluate_airfoil(airfoil, req, use_xfoil=False)
            fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
            eval_cache[coeffs_tuple] = fitness

        results.append((airfoil, fitness))

    return results

def main():
    """Main population initialization."""
    print("Generating physics-informed airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:15]):
        filename = f"results/physics_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            json.dump(airfoil.to_dict(), f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "max_thickness": airfoil.max_thickness()[0],
            "max_camber": airfoil.max_camber()[0]
        })

    print(f"Generated {len(population)} physics-informed airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Cache hits: {len(set((tuple(a.upper_coeffs), tuple(a.lower_coeffs)) for a, _ in evaluated))}")

    return best_results

if __name__ == "__main__":
    main()