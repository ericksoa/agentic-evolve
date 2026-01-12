#!/usr/bin/env python3
"""
F1 Rear Wing Aerodynamic Evaluation - Physics-Based

Uses AeroSandbox/NeuralFoil for real aerodynamic coefficients.
This replaces the simplified thin airfoil model with trained neural
networks validated against XFOIL and experimental data.

Reynolds number for F1 rear wing: ~2-3 million at racing speeds
Mach number: ~0.2-0.3 (60-100 m/s)
"""

import json
import sys
from math import factorial
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

try:
    import aerosandbox as asb
    HAS_AEROSANDBOX = True
except ImportError:
    HAS_AEROSANDBOX = False
    print("Warning: aerosandbox not installed. Install with: pip install aerosandbox")


@dataclass
class CircuitProfile:
    """Circuit characteristics for optimization."""
    name: str
    downforce_weight: float
    drag_weight: float
    avg_speed_kmh: float
    top_speed_kmh: float


CIRCUITS = {
    "monaco": CircuitProfile("Monaco", 0.9, 0.1, 160, 290),
    "monza": CircuitProfile("Monza", 0.2, 0.8, 260, 365),
    "silverstone": CircuitProfile("Silverstone", 0.6, 0.4, 230, 340),
    "spa": CircuitProfile("Spa-Francorchamps", 0.5, 0.5, 235, 345),
    "suzuka": CircuitProfile("Suzuka", 0.7, 0.3, 220, 325),
}


def cst_to_coordinates(upper_coeffs: list, lower_coeffs: list, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate airfoil coordinates from CST (Class Shape Transformation) coefficients.

    CST parameterization: y = C(x) * S(x) + x * zte
    where C(x) = x^N1 * (1-x)^N2 is the class function (N1=0.5, N2=1.0 for airfoils)
    and S(x) is a shape function using Bernstein polynomials.
    """
    # Cosine spacing for better leading edge resolution
    x = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2

    # Class function: sqrt(x) * (1-x) for typical airfoil
    class_func = np.sqrt(x) * (1 - x)

    def shape_function(coeffs):
        """Compute Bernstein polynomial shape function."""
        n = len(coeffs) - 1
        S = np.zeros_like(x)
        for i, c in enumerate(coeffs):
            # Binomial coefficient
            binom = factorial(n) // (factorial(i) * factorial(n - i))
            S += c * binom * (x ** i) * ((1 - x) ** (n - i))
        return S

    y_upper = class_func * shape_function(upper_coeffs)

    # Lower surface - coefficients stored as negative, use absolute value
    lower_abs = [abs(c) for c in lower_coeffs]
    y_lower = -class_func * shape_function(lower_abs)

    # Combine: upper surface from TE to LE, then lower surface from LE to TE
    x_coords = np.concatenate([x[::-1], x[1:]])
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])

    return x_coords, y_coords


def analyze_element(upper_coeffs: list, lower_coeffs: list, angle: float,
                    Re: float = 2e6, mach: float = 0.3) -> dict:
    """
    Analyze a single wing element using NeuralFoil.

    Returns Cl, Cd, and confidence level.
    """
    if not HAS_AEROSANDBOX:
        return {"valid": False, "error": "aerosandbox not installed"}

    try:
        x, y = cst_to_coordinates(upper_coeffs, lower_coeffs)
        coords = np.column_stack([x, y])

        # Create airfoil
        af = asb.Airfoil(name="wing_element", coordinates=coords)

        # Run NeuralFoil analysis
        aero = af.get_aero_from_neuralfoil(
            alpha=angle,
            Re=Re,
            mach=mach
        )

        cl = float(aero['CL'][0])
        cd = float(aero['CD'][0])
        confidence = float(aero['analysis_confidence'][0])

        return {
            "valid": True,
            "Cl": cl,
            "Cd": cd,
            "confidence": confidence,
            "L_D": cl / cd if cd > 0 else 0
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


def analyze_multi_element_wing(wing_config: dict, Re: float = 2e6, mach: float = 0.3) -> dict:
    """
    Analyze multi-element F1 rear wing.

    For multi-element wings, we analyze each element and apply interaction corrections.
    Real multi-element analysis would require panel methods or CFD, but this gives
    a reasonable approximation based on aerodynamic literature.

    Slot effect corrections:
    - Gap too small (<2%): interference drag, reduced lift
    - Gap optimal (2-4%): slot energizes boundary layer, 20-30% more lift
    - Gap too large (>5%): elements act independently
    """
    # Analyze main element
    main_result = analyze_element(
        wing_config["main_upper"],
        wing_config["main_lower"],
        wing_config["main_angle"],
        Re=Re, mach=mach
    )

    if not main_result["valid"]:
        return {"valid": False, "error": f"Main element: {main_result.get('error', 'unknown')}"}

    # Analyze flap element (sees disturbed flow from main element)
    # Flap operates at lower effective Re due to wake
    flap_Re = Re * 0.6  # Reduced Re in wake
    flap_result = analyze_element(
        wing_config["flap_upper"],
        wing_config["flap_lower"],
        wing_config["flap_angle"],
        Re=flap_Re, mach=mach
    )

    if not flap_result["valid"]:
        return {"valid": False, "error": f"Flap element: {flap_result.get('error', 'unknown')}"}

    # Apply multi-element interaction corrections
    gap = wing_config.get("flap_gap", 0.03)

    # Slot effectiveness
    if gap < 0.02:
        # Gap too small - interference
        slot_cl_factor = 0.85
        slot_cd_penalty = 0.02
    elif gap < 0.05:
        # Optimal gap - slot effect working
        slot_cl_factor = 1.25  # Boundary layer energization
        slot_cd_penalty = 0.005
    else:
        # Gap too large - elements more independent
        slot_cl_factor = 1.05
        slot_cd_penalty = 0.01

    # Flap operates in downwash from main element
    # Effective angle is reduced
    downwash_angle = main_result["Cl"] * 2  # Approximate downwash in degrees

    # Combined lift (main + modified flap contribution)
    # Flap sees reduced dynamic pressure in wake (~70%)
    flap_dp_factor = 0.7

    Cl_total = main_result["Cl"] + flap_result["Cl"] * flap_dp_factor * slot_cl_factor

    # Drag adds directly plus interaction terms
    Cd_total = main_result["Cd"] + flap_result["Cd"] * flap_dp_factor + slot_cd_penalty

    # Induced drag from total lift (F1 wing AR ~2.5)
    aspect_ratio = 2.5
    oswald_efficiency = 0.75
    Cd_induced = (Cl_total ** 2) / (np.pi * aspect_ratio * oswald_efficiency)
    Cd_total += Cd_induced

    # Confidence is minimum of both elements
    confidence = min(main_result["confidence"], flap_result["confidence"])

    return {
        "valid": True,
        "Cl": round(Cl_total, 4),
        "Cd": round(Cd_total, 4),
        "L_D": round(Cl_total / Cd_total, 2),
        "Cl2_Cd": round(Cl_total**2 / Cd_total, 2),
        "confidence": round(confidence, 3),
        "main": {
            "Cl": round(main_result["Cl"], 4),
            "Cd": round(main_result["Cd"], 4),
            "L_D": round(main_result["L_D"], 1)
        },
        "flap": {
            "Cl": round(flap_result["Cl"], 4),
            "Cd": round(flap_result["Cd"], 4),
            "L_D": round(flap_result["L_D"], 1)
        }
    }


def evaluate_wing(wing_config: dict, circuit: str = "silverstone") -> dict:
    """
    Full evaluation of wing configuration with physics-based aerodynamics.
    """
    profile = CIRCUITS.get(circuit, CIRCUITS["silverstone"])

    # Analyze wing aerodynamics
    aero = analyze_multi_element_wing(wing_config)

    if not aero["valid"]:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": aero.get("error", "Analysis failed")
        }

    Cl = aero["Cl"]
    Cd = aero["Cd"]
    L_D = aero["L_D"]

    # Fitness function weighted by circuit requirements
    # Higher downforce weight = prefer higher Cl
    # Higher drag weight = prefer lower Cd

    fitness = (
        profile.downforce_weight * Cl * 30 +      # Downforce contribution
        profile.drag_weight * (1/Cd) * 15 +       # Low drag contribution
        L_D * 1.0 +                                # Efficiency bonus
        (Cl**2 / Cd) * 0.5                         # Aero efficiency
    )

    # Penalize low confidence results
    fitness *= aero["confidence"]

    return {
        "valid": True,
        "fitness": round(fitness, 4),
        "metrics": {
            "Cl": Cl,
            "Cd": Cd,
            "L_D": L_D,
            "Cl2_Cd": aero["Cl2_Cd"],
            "confidence": aero["confidence"],
            "main_angle": wing_config.get("main_angle"),
            "flap_angle": wing_config.get("flap_angle"),
            "flap_gap": wing_config.get("flap_gap")
        },
        "breakdown": aero,
        "circuit": profile.name
    }


def load_wing(path: str) -> dict:
    """Load wing configuration from JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_physics.py <wing.json> [--json] [--circuit=X]")
        print("\nCircuits: monaco, monza, silverstone, spa, suzuka")
        print("\nThis uses AeroSandbox/NeuralFoil for physics-based aerodynamics.")
        sys.exit(1)

    wing_path = sys.argv[1]
    json_output = "--json" in sys.argv

    # Parse circuit
    circuit = "silverstone"
    for arg in sys.argv:
        if arg.startswith("--circuit="):
            circuit = arg.split("=")[1].lower()

    try:
        wing = load_wing(wing_path)
    except Exception as e:
        result = {"valid": False, "fitness": 0.0, "error": f"Failed to load: {e}"}
        if json_output:
            print(json.dumps(result))
        else:
            print(f"ERROR: {result['error']}")
        sys.exit(1)

    result = evaluate_wing(wing, circuit)

    if json_output:
        print(json.dumps(result))
    else:
        if result["valid"]:
            m = result["metrics"]
            b = result["breakdown"]
            print(f"F1 Rear Wing - Physics-Based Evaluation")
            print(f"Circuit: {result['circuit']}")
            print(f"{'=' * 50}")
            print(f"  Fitness: {result['fitness']:.2f}")
            print(f"  Confidence: {m['confidence']:.1%}")
            print(f"\nTotal Aerodynamics:")
            print(f"  Cl (downforce): {m['Cl']:.4f}")
            print(f"  Cd (drag):      {m['Cd']:.4f}")
            print(f"  L/D:            {m['L_D']:.1f}")
            print(f"  Cl²/Cd:         {m['Cl2_Cd']:.1f}")
            print(f"\nElement Breakdown:")
            print(f"  Main @ {m['main_angle']}°:  Cl={b['main']['Cl']:.3f}, Cd={b['main']['Cd']:.4f}, L/D={b['main']['L_D']:.0f}")
            print(f"  Flap @ {m['flap_angle']}°:  Cl={b['flap']['Cl']:.3f}, Cd={b['flap']['Cd']:.4f}, L/D={b['flap']['L_D']:.0f}")
            print(f"\nConfiguration:")
            print(f"  Slot gap: {m['flap_gap']*100:.1f}%")
        else:
            print(f"INVALID: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
