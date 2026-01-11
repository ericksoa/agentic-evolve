#!/usr/bin/env python3
"""
Airfoil Fitness Evaluator

Evaluates airfoil designs using XFOIL aerodynamic analysis.

Fitness can be optimized for different objectives:
- max_ld: Maximum lift-to-drag ratio (cruise efficiency)
- max_cl: Maximum lift coefficient (high-lift applications)
- min_cd: Minimum drag at target Cl (speed/racing)
- endurance: Cl^1.5 / Cd (maximum endurance for propeller aircraft)
- range: Cl / Cd (maximum range)

Design constraints:
- Minimum thickness for structural integrity
- Maximum thickness for drag
- Reasonable stall characteristics
- Smooth surface (no oscillations)
"""

import json
import sys
import math
from pathlib import Path
from typing import Optional

from airfoil import Airfoil, load_airfoil
from xfoil_runner import run_xfoil, find_xfoil, simple_lift_estimate, simple_drag_estimate


# Design requirements
class DesignRequirements:
    """Operating conditions and constraints."""
    def __init__(self,
                 reynolds: float = 200000,      # Typical for RC/drone
                 mach: float = 0.0,             # Incompressible
                 design_alpha: float = 4.0,     # Design angle of attack
                 design_cl: float = 0.6,        # Target lift coefficient
                 min_thickness: float = 0.06,   # Minimum t/c for structure
                 max_thickness: float = 0.18,   # Maximum t/c for drag
                 objective: str = "max_ld"):    # Optimization objective
        self.reynolds = reynolds
        self.mach = mach
        self.design_alpha = design_alpha
        self.design_cl = design_cl
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.objective = objective


def evaluate_airfoil(airfoil: Airfoil,
                     requirements: Optional[DesignRequirements] = None,
                     use_xfoil: bool = True) -> dict:
    """
    Evaluate an airfoil design.

    Returns dict with:
    - valid: bool
    - fitness: float (higher is better)
    - metrics: dict with Cl, Cd, L/D, etc.
    - error: str (if invalid)
    """
    if requirements is None:
        requirements = DesignRequirements()

    # Check geometric validity
    is_valid, msg = airfoil.is_valid()
    if not is_valid:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": f"Invalid geometry: {msg}"
        }

    # Check thickness constraints
    max_t, x_t = airfoil.max_thickness()
    if max_t < requirements.min_thickness:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": f"Too thin: {max_t:.3f} < {requirements.min_thickness:.3f}"
        }
    if max_t > requirements.max_thickness:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": f"Too thick: {max_t:.3f} > {requirements.max_thickness:.3f}"
        }

    # Aerodynamic analysis
    if use_xfoil and find_xfoil():
        return evaluate_with_xfoil(airfoil, requirements)
    else:
        return evaluate_analytical(airfoil, requirements)


def evaluate_with_xfoil(airfoil: Airfoil, req: DesignRequirements) -> dict:
    """Evaluate using XFOIL."""
    polar = run_xfoil(
        airfoil,
        reynolds=req.reynolds,
        mach=req.mach,
        alpha_start=-2,
        alpha_end=14,
        alpha_step=0.5
    )

    if polar is None or len(polar.results) < 5:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": "XFOIL analysis failed (unconverged or invalid shape)"
        }

    # Extract key metrics
    converged = [r for r in polar.results if r.converged]
    if len(converged) < 3:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": "Too few converged points"
        }

    # Find design point (closest to target Cl or alpha)
    design_point = min(converged, key=lambda r: abs(r.alpha - req.design_alpha))

    # Calculate metrics
    Cl_design = design_point.Cl
    Cd_design = design_point.Cd
    L_D_design = Cl_design / Cd_design if Cd_design > 0 else 0

    # Maximum L/D
    L_D_max, Cl_at_max_LD = polar.L_D_max

    # Cl_max and stall
    Cl_max = polar.Cl_max
    alpha_stall = polar.alpha_stall

    # Calculate fitness based on objective
    if req.objective == "max_ld":
        # Primary: max L/D, secondary: gentle stall
        fitness = L_D_max
        # Bonus for Cl_max > 1.0 (good stall margin)
        if Cl_max > 1.0:
            fitness *= 1 + 0.05 * (Cl_max - 1.0)

    elif req.objective == "max_cl":
        # Primary: maximum lift
        fitness = Cl_max * 100  # Scale for comparison
        # Penalty for very high drag at Cl_max
        cl_max_point = max(converged, key=lambda r: r.Cl)
        if cl_max_point.Cd > 0.05:
            fitness *= 0.8

    elif req.objective == "min_cd":
        # Minimum drag at target Cl
        # Find point closest to target Cl
        target_point = min(converged, key=lambda r: abs(r.Cl - req.design_cl))
        Cd_at_target = target_point.Cd
        fitness = 100 / (1 + Cd_at_target * 1000)  # Inverse, higher is better

    elif req.objective == "endurance":
        # Cl^1.5 / Cd for propeller aircraft endurance
        best_endurance = max((r.Cl**1.5 / r.Cd for r in converged if r.Cd > 0 and r.Cl > 0),
                             default=0)
        fitness = best_endurance * 10

    elif req.objective == "range":
        # Cl / Cd (same as L/D)
        fitness = L_D_max

    else:
        fitness = L_D_max  # Default

    max_t, x_t = airfoil.max_thickness()
    max_c, x_c = airfoil.max_camber()

    return {
        "valid": True,
        "fitness": round(fitness, 4),
        "metrics": {
            "L_D_max": round(L_D_max, 2),
            "L_D_design": round(L_D_design, 2),
            "Cl_max": round(Cl_max, 4),
            "Cl_design": round(Cl_design, 4),
            "Cd_design": round(Cd_design, 6),
            "alpha_stall": round(alpha_stall, 1),
            "max_thickness": round(max_t, 4),
            "max_thickness_x": round(x_t, 3),
            "max_camber": round(max_c, 4),
            "max_camber_x": round(x_c, 3),
            "converged_points": len(converged)
        },
        "objective": req.objective,
        "reynolds": req.reynolds
    }


def evaluate_analytical(airfoil: Airfoil, req: DesignRequirements) -> dict:
    """
    Evaluate using simple analytical estimates.
    Fallback when XFOIL is not available.
    """
    # Sample across angle range
    results = []
    for alpha in range(-2, 13):
        Cl = simple_lift_estimate(airfoil, alpha)
        Cd = simple_drag_estimate(airfoil, Cl, req.reynolds)
        if Cd > 0:
            results.append((alpha, Cl, Cd, Cl/Cd))

    if not results:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": "Analytical evaluation failed"
        }

    # Find max L/D
    best = max(results, key=lambda x: x[3])
    L_D_max = best[3]
    Cl_at_max = best[1]

    # Estimate Cl_max (thin airfoil theory breaks down)
    Cl_max = max(r[1] for r in results)

    max_t, x_t = airfoil.max_thickness()
    max_c, x_c = airfoil.max_camber()

    # Fitness
    if req.objective == "max_ld":
        fitness = L_D_max
    elif req.objective == "max_cl":
        fitness = Cl_max * 100
    else:
        fitness = L_D_max

    return {
        "valid": True,
        "fitness": round(fitness, 4),
        "metrics": {
            "L_D_max": round(L_D_max, 2),
            "Cl_max": round(Cl_max, 4),
            "max_thickness": round(max_t, 4),
            "max_camber": round(max_c, 4),
        },
        "objective": req.objective,
        "reynolds": req.reynolds,
        "method": "analytical_estimate"
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <airfoil.json> [--json] [--objective=max_ld]")
        print("\nObjectives: max_ld, max_cl, min_cd, endurance, range")
        print("Reynolds: Set via --reynolds=N (default: 200000)")
        sys.exit(1)

    airfoil_path = sys.argv[1]
    json_output = "--json" in sys.argv

    # Parse options
    objective = "max_ld"
    reynolds = 200000
    for arg in sys.argv:
        if arg.startswith("--objective="):
            objective = arg.split("=")[1]
        if arg.startswith("--reynolds="):
            reynolds = float(arg.split("=")[1])

    try:
        airfoil = load_airfoil(airfoil_path)
    except Exception as e:
        result = {"valid": False, "fitness": 0.0, "error": f"Failed to load: {e}"}
        if json_output:
            print(json.dumps(result))
        else:
            print(f"ERROR: {result['error']}")
        sys.exit(1)

    req = DesignRequirements(reynolds=reynolds, objective=objective)
    result = evaluate_airfoil(airfoil, req)

    if json_output:
        print(json.dumps(result))
    else:
        if result["valid"]:
            print(f"Airfoil Evaluation (objective: {objective})")
            print(f"  Fitness: {result['fitness']:.4f}")
            print(f"\nMetrics:")
            for key, value in result.get("metrics", {}).items():
                print(f"  {key}: {value}")
            if "method" in result:
                print(f"\nNote: Using {result['method']} (XFOIL not available)")
        else:
            print(f"INVALID: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
