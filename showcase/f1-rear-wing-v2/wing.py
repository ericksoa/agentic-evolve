#!/usr/bin/env python3
"""
F1 Rear Wing Aerodynamics Module

Models a multi-element F1 rear wing using CST parameterization.
F1 rear wings typically have:
- Main plane (primary downforce element)
- Flap (adjustable, DRS-enabled upper element)
- Endplates (not modeled here, but affect 3D flow)

Key metrics:
- Cl (downforce coefficient) - higher = more grip
- Cd (drag coefficient) - lower = more speed
- L/D efficiency - balance for different circuits
- Cl² / Cd - "aero efficiency" metric used by teams

Regulations (simplified 2024):
- Max chord: 350mm main + 150mm flap
- Max height: 200mm above reference plane
- Angle of attack limits apply
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
    """CST class function for airfoil shapes."""
    if x <= 0 or x >= 1:
        return 0
    return (x ** N1) * ((1 - x) ** N2)


def shape_function(x: float, coeffs: List[float]) -> float:
    """CST shape function using Bernstein polynomials."""
    n = len(coeffs) - 1
    S = 0.0
    for i, A in enumerate(coeffs):
        S += A * bernstein(n, i, x)
    return S


@dataclass
class WingElement:
    """Single F1 wing element (main plane or flap)."""
    upper_coeffs: List[float]
    lower_coeffs: List[float]
    chord: float = 1.0  # Normalized chord length
    angle: float = 0.0  # Angle of attack (degrees)
    position: Tuple[float, float] = (0.0, 0.0)  # (x, y) offset

    def surface_y(self, x: float, upper: bool = True) -> float:
        """Get y-coordinate at x position."""
        C = class_function(x, 0.5, 1.0)
        coeffs = self.upper_coeffs if upper else self.lower_coeffs
        S = shape_function(x, coeffs)
        if upper:
            return C * S
        else:
            return C * S  # Lower coeffs are already negative

    def get_coordinates(self, n_points: int = 60) -> Tuple[List[float], List[float]]:
        """Get wing element coordinates."""
        beta = np.linspace(0, np.pi, n_points)
        x_coords = 0.5 * (1 - np.cos(beta))

        # Upper surface (TE to LE)
        x_upper = list(reversed(x_coords))
        y_upper = [self.surface_y(x, upper=True) for x in x_upper]

        # Lower surface (LE to TE)
        x_lower = list(x_coords[1:])
        y_lower = [self.surface_y(x, upper=False) for x in x_lower]

        x = x_upper + x_lower
        y = y_upper + y_lower

        # Apply rotation for angle of attack
        if self.angle != 0:
            angle_rad = math.radians(self.angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            x_rot = [xi * cos_a - yi * sin_a for xi, yi in zip(x, y)]
            y_rot = [xi * sin_a + yi * cos_a for xi, yi in zip(x, y)]
            x, y = x_rot, y_rot

        # Scale by chord and apply position offset
        x = [xi * self.chord + self.position[0] for xi in x]
        y = [yi * self.chord + self.position[1] for yi in y]

        return x, y

    def thickness_at(self, x: float) -> float:
        """Get thickness at chord position."""
        return self.surface_y(x, upper=True) - self.surface_y(x, upper=False)

    def max_thickness(self) -> Tuple[float, float]:
        """Get maximum thickness and location."""
        x_vals = np.linspace(0.01, 0.99, 50)
        thicknesses = [self.thickness_at(x) for x in x_vals]
        max_idx = np.argmax(thicknesses)
        return thicknesses[max_idx], x_vals[max_idx]

    def camber_at(self, x: float) -> float:
        """Get camber (mean line) at position."""
        return (self.surface_y(x, upper=True) + self.surface_y(x, upper=False)) / 2

    def max_camber(self) -> Tuple[float, float]:
        """Get maximum camber and location."""
        x_vals = np.linspace(0.01, 0.99, 50)
        cambers = [abs(self.camber_at(x)) for x in x_vals]
        max_idx = np.argmax(cambers)
        return cambers[max_idx], x_vals[max_idx]


@dataclass
class F1RearWing:
    """
    Complete F1 rear wing assembly.

    Parameters define the shape of main plane and flap elements.
    """
    # Main plane CST coefficients
    main_upper: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.18, 0.12])
    main_lower: List[float] = field(default_factory=lambda: [-0.08, -0.06, -0.04, -0.02])
    main_angle: float = 12.0  # degrees

    # Flap CST coefficients
    flap_upper: List[float] = field(default_factory=lambda: [0.12, 0.15, 0.12, 0.08])
    flap_lower: List[float] = field(default_factory=lambda: [-0.06, -0.04, -0.02, -0.01])
    flap_angle: float = 25.0  # degrees (high for Monaco, low for Monza)
    flap_gap: float = 0.03  # Gap between elements (slot)

    # Configuration
    drs_open: bool = False  # DRS flap position

    def get_main_element(self) -> WingElement:
        """Get main plane element."""
        return WingElement(
            upper_coeffs=self.main_upper,
            lower_coeffs=self.main_lower,
            chord=0.7,  # 70% of total
            angle=self.main_angle,
            position=(0.0, 0.0)
        )

    def get_flap_element(self) -> WingElement:
        """Get flap element."""
        # Position flap behind and above main element
        angle = self.flap_angle if not self.drs_open else self.flap_angle - 15
        return WingElement(
            upper_coeffs=self.flap_upper,
            lower_coeffs=self.flap_lower,
            chord=0.3,  # 30% of total
            angle=angle,
            position=(0.65, 0.08 + self.flap_gap)
        )

    def get_coordinates(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Get coordinates for both elements."""
        main = self.get_main_element()
        flap = self.get_flap_element()

        main_x, main_y = main.get_coordinates()
        flap_x, flap_y = flap.get_coordinates()

        return [main_x, flap_x], [main_y, flap_y]

    def is_valid(self) -> Tuple[bool, str]:
        """Check if wing configuration is valid."""
        main = self.get_main_element()
        flap = self.get_flap_element()

        # Check thickness constraints
        main_t, _ = main.max_thickness()
        if main_t < 0.02:
            return False, f"Main element too thin: {main_t:.3f}"
        if main_t > 0.25:
            return False, f"Main element too thick: {main_t:.3f}"

        flap_t, _ = flap.max_thickness()
        if flap_t < 0.01:
            return False, f"Flap too thin: {flap_t:.3f}"
        if flap_t > 0.20:
            return False, f"Flap too thick: {flap_t:.3f}"

        # Check angles
        if self.main_angle < 0 or self.main_angle > 25:
            return False, f"Main angle out of range: {self.main_angle}"
        if self.flap_angle < 10 or self.flap_angle > 45:
            return False, f"Flap angle out of range: {self.flap_angle}"

        # Check gap
        if self.flap_gap < 0.01 or self.flap_gap > 0.10:
            return False, f"Flap gap out of range: {self.flap_gap}"

        return True, "Valid configuration"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "main_upper": self.main_upper,
            "main_lower": self.main_lower,
            "main_angle": self.main_angle,
            "flap_upper": self.flap_upper,
            "flap_lower": self.flap_lower,
            "flap_angle": self.flap_angle,
            "flap_gap": self.flap_gap
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'F1RearWing':
        """Create from dictionary."""
        return cls(
            main_upper=d["main_upper"],
            main_lower=d["main_lower"],
            main_angle=d.get("main_angle", 12.0),
            flap_upper=d["flap_upper"],
            flap_lower=d["flap_lower"],
            flap_angle=d.get("flap_angle", 25.0),
            flap_gap=d.get("flap_gap", 0.03)
        )


def load_wing(filepath: str) -> F1RearWing:
    """Load wing from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return F1RearWing.from_dict(data)


def save_wing(wing: F1RearWing, filepath: str):
    """Save wing to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(wing.to_dict(), f, indent=2)


# Preset configurations for different circuits
def monaco_wing() -> F1RearWing:
    """High downforce configuration for Monaco."""
    return F1RearWing(
        main_upper=[0.18, 0.25, 0.22, 0.15],
        main_lower=[-0.10, -0.08, -0.05, -0.02],
        main_angle=15.0,
        flap_upper=[0.15, 0.20, 0.15, 0.10],
        flap_lower=[-0.08, -0.05, -0.03, -0.01],
        flap_angle=35.0,
        flap_gap=0.025
    )


def monza_wing() -> F1RearWing:
    """Low drag configuration for Monza."""
    return F1RearWing(
        main_upper=[0.10, 0.12, 0.10, 0.08],
        main_lower=[-0.05, -0.04, -0.03, -0.02],
        main_angle=8.0,
        flap_upper=[0.08, 0.10, 0.08, 0.05],
        flap_lower=[-0.04, -0.03, -0.02, -0.01],
        flap_angle=18.0,
        flap_gap=0.04
    )


def balanced_wing() -> F1RearWing:
    """Balanced configuration (baseline)."""
    return F1RearWing()


if __name__ == "__main__":
    # Test wing configurations
    wing = balanced_wing()
    valid, msg = wing.is_valid()
    print(f"Balanced wing: {msg}")

    main = wing.get_main_element()
    t, x = main.max_thickness()
    print(f"Main element thickness: {t*100:.1f}% at x={x:.2f}")

    monaco = monaco_wing()
    print(f"\nMonaco wing angle: main={monaco.main_angle}°, flap={monaco.flap_angle}°")

    monza = monza_wing()
    print(f"Monza wing angle: main={monza.main_angle}°, flap={monza.flap_angle}°")
