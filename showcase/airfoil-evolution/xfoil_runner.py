"""
XFOIL Runner - Interface to XFOIL for airfoil aerodynamic analysis

XFOIL is a free program for subsonic airfoil analysis developed by MIT.
It calculates:
- Lift coefficient (Cl)
- Drag coefficient (Cd)
- Moment coefficient (Cm)
- Pressure distributions
- Boundary layer properties

Installation:
- macOS: brew install xfoil
- Linux: sudo apt-get install xfoil
- Windows: Download from web.mit.edu/drela/Public/web/xfoil/

This module provides a Python wrapper that:
1. Writes the airfoil coordinates to a temp file
2. Creates XFOIL command script
3. Runs XFOIL and parses output
4. Returns aerodynamic coefficients
"""

import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

from airfoil import Airfoil


@dataclass
class XfoilResult:
    """Results from XFOIL analysis at a single angle of attack."""
    alpha: float      # Angle of attack (degrees)
    Cl: float         # Lift coefficient
    Cd: float         # Drag coefficient
    Cdp: float        # Pressure drag coefficient
    Cm: float         # Moment coefficient (about c/4)
    Top_Xtr: float    # Upper surface transition location
    Bot_Xtr: float    # Lower surface transition location
    converged: bool   # Whether XFOIL converged


@dataclass
class XfoilPolar:
    """Collection of results across angle of attack sweep."""
    reynolds: float
    mach: float
    results: List[XfoilResult]

    @property
    def Cl_max(self) -> float:
        """Maximum lift coefficient."""
        return max(r.Cl for r in self.results if r.converged)

    @property
    def Cd_min(self) -> float:
        """Minimum drag coefficient."""
        return min(r.Cd for r in self.results if r.converged)

    @property
    def L_D_max(self) -> Tuple[float, float]:
        """Maximum L/D and the Cl at which it occurs."""
        best = max((r for r in self.results if r.converged and r.Cd > 0),
                   key=lambda r: r.Cl / r.Cd)
        return best.Cl / best.Cd, best.Cl

    @property
    def alpha_stall(self) -> float:
        """Approximate stall angle (angle at Cl_max)."""
        best = max((r for r in self.results if r.converged), key=lambda r: r.Cl)
        return best.alpha


def find_xfoil() -> Optional[str]:
    """Find XFOIL executable."""
    # Check common locations
    candidates = [
        "xfoil",
        "/usr/local/bin/xfoil",
        "/opt/homebrew/bin/xfoil",
        "/usr/bin/xfoil",
        os.path.expanduser("~/bin/xfoil"),
    ]

    for path in candidates:
        try:
            result = subprocess.run([path], input=b"\nquit\n",
                                   capture_output=True, timeout=5)
            return path
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    return None


def run_xfoil(airfoil: Airfoil,
              reynolds: float = 100000,
              mach: float = 0.0,
              alpha_start: float = -5.0,
              alpha_end: float = 15.0,
              alpha_step: float = 0.5,
              n_crit: float = 9.0,
              max_iter: int = 100,
              xfoil_path: Optional[str] = None) -> Optional[XfoilPolar]:
    """
    Run XFOIL analysis on an airfoil.

    Args:
        airfoil: Airfoil object to analyze
        reynolds: Reynolds number (e.g., 100000 for small RC aircraft)
        mach: Mach number (0 for incompressible)
        alpha_start: Starting angle of attack
        alpha_end: Ending angle of attack
        alpha_step: Angle of attack increment
        n_crit: Critical amplification ratio (9 = clean wind tunnel)
        max_iter: Maximum iterations for convergence
        xfoil_path: Path to XFOIL executable (auto-detect if None)

    Returns:
        XfoilPolar with results, or None if XFOIL failed
    """
    if xfoil_path is None:
        xfoil_path = find_xfoil()
        if xfoil_path is None:
            print("XFOIL not found. Install with: brew install xfoil")
            return None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write airfoil coordinates
        dat_file = os.path.join(tmpdir, "airfoil.dat")
        airfoil.save_dat(dat_file, "EVOLVED")

        # Write polar output file path
        polar_file = os.path.join(tmpdir, "polar.txt")

        # Create XFOIL command script
        commands = f"""
LOAD {dat_file}
EVOLVED

PANE
OPER
VISC {reynolds}
MACH {mach}
ITER {max_iter}
VPAR
N {n_crit}

PACC
{polar_file}

ASEQ {alpha_start} {alpha_end} {alpha_step}

PACC

QUIT
"""

        # Run XFOIL
        try:
            result = subprocess.run(
                [xfoil_path],
                input=commands.encode(),
                capture_output=True,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            print("XFOIL timed out")
            return None
        except Exception as e:
            print(f"XFOIL error: {e}")
            return None

        # Parse polar file
        if not os.path.exists(polar_file):
            print("XFOIL produced no output (airfoil may be invalid)")
            return None

        return parse_polar_file(polar_file, reynolds, mach)


def parse_polar_file(filepath: str, reynolds: float, mach: float) -> XfoilPolar:
    """Parse XFOIL polar output file."""
    results = []

    with open(filepath) as f:
        lines = f.readlines()

    # Skip header lines, find data
    in_data = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Header line marker
        if "alpha" in line.lower() and "CL" in line:
            in_data = True
            continue

        if line.startswith("---"):
            in_data = True
            continue

        if in_data:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    result = XfoilResult(
                        alpha=float(parts[0]),
                        Cl=float(parts[1]),
                        Cd=float(parts[2]),
                        Cdp=float(parts[3]),
                        Cm=float(parts[4]),
                        Top_Xtr=float(parts[5]),
                        Bot_Xtr=float(parts[6]),
                        converged=True
                    )
                    results.append(result)
                except (ValueError, IndexError):
                    continue

    return XfoilPolar(reynolds=reynolds, mach=mach, results=results)


def analyze_single_alpha(airfoil: Airfoil,
                         alpha: float,
                         reynolds: float = 100000,
                         xfoil_path: Optional[str] = None) -> Optional[XfoilResult]:
    """Quick single-point analysis."""
    polar = run_xfoil(
        airfoil,
        reynolds=reynolds,
        alpha_start=alpha,
        alpha_end=alpha,
        alpha_step=1.0,
        xfoil_path=xfoil_path
    )

    if polar and polar.results:
        return polar.results[0]
    return None


# Fallback: Simple panel method for when XFOIL isn't available
def simple_lift_estimate(airfoil: Airfoil, alpha_deg: float) -> float:
    """
    Very rough lift coefficient estimate using thin airfoil theory.
    Only use as fallback when XFOIL isn't available.

    Cl ≈ 2π(α + α_L0)
    where α_L0 is the zero-lift angle (related to camber)
    """
    alpha_rad = math.radians(alpha_deg)

    # Estimate zero-lift angle from camber
    max_camber, _ = airfoil.max_camber()
    alpha_L0 = -2 * max_camber  # Rough approximation

    Cl = 2 * math.pi * (alpha_rad - alpha_L0)
    return Cl


def simple_drag_estimate(airfoil: Airfoil, Cl: float, reynolds: float) -> float:
    """
    Very rough drag coefficient estimate.
    Only use as fallback when XFOIL isn't available.

    Cd ≈ Cd0 + Cl²/(π*AR*e)  (induced drag not applicable for 2D)
    Cd ≈ Cd_friction + Cd_pressure

    For 2D airfoil, use empirical correlation.
    """
    max_t, _ = airfoil.max_thickness()

    # Skin friction (flat plate turbulent)
    Cf = 0.074 / (reynolds ** 0.2)

    # Form factor (depends on thickness)
    FF = 1 + 2.0 * max_t + 60 * max_t**4

    # Wetted area ratio (both surfaces)
    Swet_S = 2.0

    # Parasite drag
    Cd0 = Cf * FF * Swet_S

    # Add pressure drag estimate (very rough)
    Cd_pressure = 0.01 * max_t

    return Cd0 + Cd_pressure


import math

if __name__ == "__main__":
    from airfoil import naca0012_approx, naca2412_approx

    # Check if XFOIL is available
    xfoil = find_xfoil()
    if xfoil:
        print(f"Found XFOIL at: {xfoil}")

        # Test with NACA 0012
        foil = naca0012_approx()
        print("\nAnalyzing NACA 0012 approximation...")

        polar = run_xfoil(foil, reynolds=100000, alpha_start=-2, alpha_end=12, alpha_step=1)

        if polar and polar.results:
            print(f"\nResults at Re={polar.reynolds:.0f}:")
            print(f"{'Alpha':>6} {'Cl':>8} {'Cd':>8} {'L/D':>8}")
            print("-" * 34)
            for r in polar.results:
                ld = r.Cl / r.Cd if r.Cd > 0 else 0
                print(f"{r.alpha:>6.1f} {r.Cl:>8.4f} {r.Cd:>8.5f} {ld:>8.1f}")

            ld_max, cl_at_max = polar.L_D_max
            print(f"\nMax L/D: {ld_max:.1f} at Cl={cl_at_max:.3f}")
            print(f"Cl_max: {polar.Cl_max:.3f}")
            print(f"Cd_min: {polar.Cd_min:.5f}")
    else:
        print("XFOIL not found. Install with:")
        print("  macOS: brew install xfoil")
        print("  Linux: sudo apt-get install xfoil")
        print("\nUsing simple analytical estimates instead...")

        foil = naca0012_approx()
        for alpha in range(0, 10, 2):
            Cl = simple_lift_estimate(foil, alpha)
            Cd = simple_drag_estimate(foil, Cl, 100000)
            print(f"α={alpha:>2}°: Cl={Cl:.3f}, Cd={Cd:.4f}, L/D={Cl/Cd:.1f}")
