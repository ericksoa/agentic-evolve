#!/usr/bin/env python3
"""
F1 Rear Wing Aerodynamic Evaluation

Evaluates wing configurations for:
- Downforce (Cl) - grip for cornering
- Drag (Cd) - resistance limiting top speed
- Efficiency (Cl/Cd or Cl²/Cd) - balance metric
- Lap time impact estimation

Uses simplified aerodynamic models calibrated to F1 data.
Real teams use CFD (millions of cells) and wind tunnels.

Circuit-specific optimization:
- Monaco: Max downforce (tight corners, low speed)
- Monza: Min drag (long straights, high speed)
- Silverstone: Balanced (mix of high-speed and technical)
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from wing import F1RearWing, load_wing


@dataclass
class CircuitProfile:
    """Circuit characteristics for optimization."""
    name: str
    downforce_weight: float  # 0-1, importance of downforce
    drag_weight: float       # 0-1, importance of low drag
    avg_speed_kmh: float     # Average speed
    top_speed_kmh: float     # Top speed
    description: str


# Circuit profiles
CIRCUITS = {
    "monaco": CircuitProfile(
        name="Monaco",
        downforce_weight=0.9,
        drag_weight=0.1,
        avg_speed_kmh=160,
        top_speed_kmh=290,
        description="Tight streets, max downforce"
    ),
    "monza": CircuitProfile(
        name="Monza",
        downforce_weight=0.2,
        drag_weight=0.8,
        avg_speed_kmh=260,
        top_speed_kmh=365,
        description="Temple of Speed, min drag"
    ),
    "silverstone": CircuitProfile(
        name="Silverstone",
        downforce_weight=0.6,
        drag_weight=0.4,
        avg_speed_kmh=230,
        top_speed_kmh=340,
        description="High-speed corners, balanced"
    ),
    "spa": CircuitProfile(
        name="Spa-Francorchamps",
        downforce_weight=0.5,
        drag_weight=0.5,
        avg_speed_kmh=235,
        top_speed_kmh=345,
        description="Power circuit, efficiency matters"
    ),
    "suzuka": CircuitProfile(
        name="Suzuka",
        downforce_weight=0.7,
        drag_weight=0.3,
        avg_speed_kmh=220,
        top_speed_kmh=325,
        description="Technical, needs grip"
    ),
}


def estimate_cl(wing: F1RearWing) -> float:
    """
    Estimate downforce coefficient (Cl).

    Based on:
    - Element angles (more angle = more lift, up to stall)
    - Camber (more camber = more lift)
    - Multi-element interaction (slot effect)

    F1 rear wings typically achieve Cl = 1.5-2.5
    """
    main = wing.get_main_element()
    flap = wing.get_flap_element()

    # Base lift from thin airfoil theory: Cl ≈ 2π × sin(α)
    # Modified for high-lift multi-element wings

    # Main element contribution
    main_alpha = math.radians(wing.main_angle)
    main_camber, _ = main.max_camber()
    cl_main = 2 * math.pi * math.sin(main_alpha) * (1 + 2 * main_camber)

    # Account for high angle stall reduction
    if wing.main_angle > 18:
        stall_factor = 1 - 0.05 * (wing.main_angle - 18)
        cl_main *= max(0.5, stall_factor)

    # Flap contribution (enhanced by slot effect)
    flap_alpha = math.radians(wing.flap_angle)
    flap_camber, _ = flap.max_camber()
    cl_flap = 2 * math.pi * math.sin(flap_alpha) * (1 + 2 * flap_camber)

    # Slot effect: gap energizes boundary layer, delays separation
    # Optimal gap around 2-4% chord
    gap_efficiency = 1.0
    if wing.flap_gap < 0.02:
        gap_efficiency = 0.8  # Too small, interference
    elif wing.flap_gap > 0.05:
        gap_efficiency = 0.9  # Too large, loses slot effect

    cl_flap *= gap_efficiency * 0.7  # Flap sees reduced dynamic pressure

    # Stall reduction for high flap angles
    if wing.flap_angle > 30:
        stall_factor = 1 - 0.03 * (wing.flap_angle - 30)
        cl_flap *= max(0.6, stall_factor)

    # Total with interaction effects
    cl_total = (cl_main + cl_flap) * 1.15  # Multi-element bonus

    # Clamp to realistic F1 range
    return min(max(cl_total, 0.5), 3.0)


def estimate_cd(wing: F1RearWing, cl: float) -> float:
    """
    Estimate drag coefficient (Cd).

    Components:
    - Profile drag (friction + pressure)
    - Induced drag (due to lift)
    - Interference drag (element interaction)

    F1 rear wings typically have Cd = 0.3-0.8
    """
    main = wing.get_main_element()
    flap = wing.get_flap_element()

    # Profile drag (depends on thickness and angle)
    main_t, _ = main.max_thickness()
    flap_t, _ = flap.max_thickness()

    cd_profile_main = 0.01 + 0.1 * main_t + 0.002 * wing.main_angle
    cd_profile_flap = 0.008 + 0.08 * flap_t + 0.002 * wing.flap_angle

    # Induced drag: Cd_i = Cl² / (π × AR × e)
    # F1 wings have low aspect ratio (~2-3), so high induced drag
    aspect_ratio = 2.5
    oswald_efficiency = 0.75
    cd_induced = (cl ** 2) / (math.pi * aspect_ratio * oswald_efficiency)

    # Interference drag from multi-element
    cd_interference = 0.02 * (1 - wing.flap_gap / 0.05)

    # Total drag
    cd_total = cd_profile_main + cd_profile_flap + cd_induced + cd_interference

    return max(cd_total, 0.1)


def estimate_lap_time_delta(wing: F1RearWing, circuit: CircuitProfile,
                            baseline_cl: float = 1.8, baseline_cd: float = 0.5) -> float:
    """
    Estimate lap time difference vs baseline (seconds).

    Simplified model:
    - More downforce = faster corners
    - More drag = slower straights

    Negative = faster than baseline
    """
    cl = estimate_cl(wing)
    cd = estimate_cd(wing, cl)

    # Downforce effect on corner speed
    # More Cl = higher corner speed = lower time
    downforce_ratio = cl / baseline_cl
    corner_time_factor = 1 / math.sqrt(downforce_ratio)  # v ∝ √(downforce)

    # Drag effect on straight speed
    # More Cd = lower top speed = higher time
    drag_ratio = cd / baseline_cd
    straight_time_factor = drag_ratio ** 0.3  # Simplified

    # Weight by circuit characteristics
    corner_delta = (corner_time_factor - 1) * 30 * circuit.downforce_weight  # ~30s in corners
    straight_delta = (straight_time_factor - 1) * 20 * circuit.drag_weight   # ~20s on straights

    return corner_delta + straight_delta


def evaluate_wing(wing: F1RearWing, circuit: str = "silverstone") -> dict:
    """
    Full evaluation of wing configuration.

    Returns dict with:
    - valid: bool
    - fitness: float (higher is better)
    - metrics: detailed performance data
    """
    # Validate geometry
    valid, msg = wing.is_valid()
    if not valid:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": msg
        }

    # Get circuit profile
    profile = CIRCUITS.get(circuit, CIRCUITS["silverstone"])

    # Calculate aero coefficients
    cl = estimate_cl(wing)
    cd = estimate_cd(wing, cl)
    efficiency = cl / cd
    aero_efficiency = (cl ** 2) / cd  # F1 metric

    # Lap time impact
    lap_delta = estimate_lap_time_delta(wing, profile)

    # Fitness function
    # Weighted combination based on circuit
    fitness = (
        profile.downforce_weight * cl * 40 +      # Downforce contribution
        profile.drag_weight * (1/cd) * 20 +       # Low drag contribution
        efficiency * 5 +                           # Efficiency bonus
        max(0, -lap_delta) * 10                   # Lap time bonus
    )

    return {
        "valid": True,
        "fitness": round(fitness, 4),
        "metrics": {
            "Cl": round(cl, 4),
            "Cd": round(cd, 4),
            "L_D": round(efficiency, 2),
            "Cl2_Cd": round(aero_efficiency, 2),
            "lap_delta_s": round(lap_delta, 3),
            "main_angle": wing.main_angle,
            "flap_angle": wing.flap_angle,
            "flap_gap": round(wing.flap_gap, 3)
        },
        "circuit": profile.name,
        "circuit_description": profile.description
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <wing.json> [--json] [--circuit=X]")
        print("\nCircuits: monaco, monza, silverstone, spa, suzuka")
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
            print(f"F1 Rear Wing Evaluation - {result['circuit']}")
            print(f"{'=' * 50}")
            print(f"  Fitness: {result['fitness']:.2f}")
            print(f"\nAerodynamics:")
            print(f"  Downforce (Cl):     {result['metrics']['Cl']:.3f}")
            print(f"  Drag (Cd):          {result['metrics']['Cd']:.3f}")
            print(f"  Efficiency (L/D):   {result['metrics']['L_D']:.1f}")
            print(f"  Aero Eff (Cl²/Cd):  {result['metrics']['Cl2_Cd']:.1f}")
            print(f"\nConfiguration:")
            print(f"  Main angle:  {result['metrics']['main_angle']}°")
            print(f"  Flap angle:  {result['metrics']['flap_angle']}°")
            print(f"  Slot gap:    {result['metrics']['flap_gap']*100:.1f}%")
            print(f"\nPerformance:")
            delta = result['metrics']['lap_delta_s']
            sign = "+" if delta > 0 else ""
            print(f"  Lap time vs baseline: {sign}{delta:.3f}s")
        else:
            print(f"INVALID: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
