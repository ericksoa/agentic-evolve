"""
Airfoil Representation using CST (Class Shape Transformation) Parameterization

CST is a modern parameterization that can represent virtually any airfoil shape
using a small number of parameters. It's widely used in aerospace optimization.

The airfoil is defined by:
- Upper surface: N_upper coefficients (typically 4-6)
- Lower surface: N_lower coefficients (typically 4-6)
- Leading edge radius (implicit from class function)
- Trailing edge thickness (optional)

Reference:
Kulfan, B.M., "Universal Parametric Geometry Representation Method",
Journal of Aircraft, Vol. 45, No. 1, 2008.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


def bernstein(n: int, k: int, t: float) -> float:
    """Bernstein polynomial basis function."""
    from math import comb
    return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))


def class_function(x: float, N1: float = 0.5, N2: float = 1.0) -> float:
    """
    CST class function C(x) = x^N1 * (1-x)^N2

    N1=0.5, N2=1.0 gives round leading edge, sharp trailing edge (typical airfoil)
    """
    if x <= 0:
        return 0
    if x >= 1:
        return 0
    return (x ** N1) * ((1 - x) ** N2)


def shape_function(x: float, coeffs: List[float]) -> float:
    """
    CST shape function S(x) using Bernstein polynomials.

    S(x) = Σ A_i * B_{i,n}(x)
    """
    n = len(coeffs) - 1
    S = 0.0
    for i, A in enumerate(coeffs):
        S += A * bernstein(n, i, x)
    return S


@dataclass
class Airfoil:
    """
    CST-parameterized airfoil.

    Parameters:
        upper_coeffs: CST coefficients for upper surface (typically 4-6 values)
        lower_coeffs: CST coefficients for lower surface (typically 4-6 values)
        zte: Trailing edge thickness (half-thickness, typically 0 for sharp TE)

    The coefficients typically range from -0.5 to 0.5 for reasonable airfoils.
    """
    upper_coeffs: List[float]
    lower_coeffs: List[float]
    zte: float = 0.0  # Trailing edge half-thickness

    # Class function exponents (fixed for typical airfoils)
    N1: float = 0.5  # Leading edge (0.5 = round)
    N2: float = 1.0  # Trailing edge (1.0 = sharp)

    def upper_surface(self, x: float) -> float:
        """Get y-coordinate of upper surface at x (0 to 1)."""
        C = class_function(x, self.N1, self.N2)
        S = shape_function(x, self.upper_coeffs)
        return C * S + x * self.zte

    def lower_surface(self, x: float) -> float:
        """Get y-coordinate of lower surface at x (0 to 1)."""
        C = class_function(x, self.N1, self.N2)
        S = shape_function(x, self.lower_coeffs)
        # Lower coeffs are already negative, so C * S gives negative y
        return C * S + x * self.zte

    def get_coordinates(self, n_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate airfoil coordinates in Selig format.

        Returns (x, y) starting from trailing edge, going around upper surface
        to leading edge, then back along lower surface to trailing edge.
        """
        # Cosine spacing for better resolution at LE and TE
        beta = np.linspace(0, np.pi, n_points)
        x_coords = 0.5 * (1 - np.cos(beta))

        # Upper surface (TE to LE)
        x_upper = list(reversed(x_coords))
        y_upper = [self.upper_surface(x) for x in x_upper]

        # Lower surface (LE to TE, skip LE point to avoid duplicate)
        x_lower = list(x_coords[1:])
        y_lower = [self.lower_surface(x) for x in x_lower]

        x = x_upper + x_lower
        y = y_upper + y_lower

        return x, y

    def thickness_at(self, x: float) -> float:
        """Get thickness at chord position x."""
        return self.upper_surface(x) - self.lower_surface(x)

    def max_thickness(self, n_samples: int = 100) -> Tuple[float, float]:
        """Get maximum thickness and its x-location."""
        x_vals = np.linspace(0.01, 0.99, n_samples)
        thicknesses = [self.thickness_at(x) for x in x_vals]
        max_idx = np.argmax(thicknesses)
        return thicknesses[max_idx], x_vals[max_idx]

    def max_camber(self, n_samples: int = 100) -> Tuple[float, float]:
        """Get maximum camber and its x-location."""
        x_vals = np.linspace(0.01, 0.99, n_samples)
        cambers = [(self.upper_surface(x) + self.lower_surface(x)) / 2 for x in x_vals]
        max_idx = np.argmax(np.abs(cambers))
        return cambers[max_idx], x_vals[max_idx]

    def is_valid(self) -> Tuple[bool, str]:
        """Check if airfoil geometry is physically valid."""
        # Check for self-intersection (upper below lower)
        for x in np.linspace(0.01, 0.99, 50):
            if self.upper_surface(x) < self.lower_surface(x):
                return False, f"Self-intersection at x={x:.3f}"

        # Check for reasonable thickness
        max_t, x_t = self.max_thickness()
        if max_t < 0.01:
            return False, f"Too thin: max thickness = {max_t:.4f}"
        if max_t > 0.5:
            return False, f"Too thick: max thickness = {max_t:.4f}"

        # Check for reasonable camber
        max_c, x_c = self.max_camber()
        if abs(max_c) > 0.15:
            return False, f"Excessive camber: {max_c:.4f}"

        return True, "Valid airfoil"

    def to_dat_string(self, name: str = "EVOLVED") -> str:
        """Export to Selig .dat format (for XFOIL)."""
        x, y = self.get_coordinates(n_points=80)
        lines = [name]
        for xi, yi in zip(x, y):
            lines.append(f"  {xi:.6f}  {yi:.6f}")
        return "\n".join(lines)

    def save_dat(self, filepath: str, name: str = "EVOLVED"):
        """Save to .dat file."""
        with open(filepath, 'w') as f:
            f.write(self.to_dat_string(name))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "upper_coeffs": self.upper_coeffs,
            "lower_coeffs": self.lower_coeffs,
            "zte": self.zte
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Airfoil':
        """Create from dictionary."""
        return cls(
            upper_coeffs=d["upper_coeffs"],
            lower_coeffs=d["lower_coeffs"],
            zte=d.get("zte", 0.0)
        )


# Well-known airfoil approximations using CST
def naca0012_approx() -> Airfoil:
    """Approximate NACA 0012 symmetric airfoil."""
    # Symmetric, so upper = -lower
    coeffs = [0.17, 0.16, 0.14, 0.12, 0.10]
    return Airfoil(
        upper_coeffs=coeffs,
        lower_coeffs=[-c for c in coeffs],
        zte=0.0
    )


def naca2412_approx() -> Airfoil:
    """Approximate NACA 2412 cambered airfoil."""
    return Airfoil(
        upper_coeffs=[0.20, 0.18, 0.16, 0.13, 0.10],
        lower_coeffs=[-0.14, -0.12, -0.10, -0.09, -0.08],
        zte=0.0
    )


def flat_plate() -> Airfoil:
    """Thin flat plate (baseline for comparison)."""
    return Airfoil(
        upper_coeffs=[0.01, 0.01, 0.01, 0.01],
        lower_coeffs=[-0.01, -0.01, -0.01, -0.01],
        zte=0.0
    )


def load_airfoil(filepath: str) -> Airfoil:
    """Load airfoil from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return Airfoil.from_dict(data)


def save_airfoil(airfoil: Airfoil, filepath: str):
    """Save airfoil to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(airfoil.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test with NACA 0012 approximation
    foil = naca0012_approx()

    valid, msg = foil.is_valid()
    print(f"NACA 0012 approximation: {msg}")

    max_t, x_t = foil.max_thickness()
    print(f"Max thickness: {max_t*100:.1f}% at x={x_t:.2f}")

    max_c, x_c = foil.max_camber()
    print(f"Max camber: {max_c*100:.2f}% at x={x_c:.2f}")

    # Save as .dat
    foil.save_dat("test_airfoil.dat", "NACA0012-CST")
    print("\nSaved to test_airfoil.dat")
