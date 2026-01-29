#!/usr/bin/env python3
"""
Conservative Baseline Wing
Strategy: Safe, proven configuration as evolutionary foundation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Reliable baseline wing configuration for evolutionary starting point."""

    # Based on proven NACA-style airfoils
    # Safe margins from stall limits

    return F1RearWing(
        # Main element: Classic design (similar to original baseline)
        main_upper=[0.15, 0.19, 0.17, 0.11],  # Gentle camber curve
        main_lower=[-0.07, -0.05, -0.04, -0.02],  # Moderate lower surface
        main_angle=12.5,  # Safe angle

        # Flap: Well-proven design
        flap_upper=[0.13, 0.16, 0.13, 0.09],  # Moderate camber
        flap_lower=[-0.06, -0.04, -0.03, -0.01],  # Gentle lower curve
        flap_angle=26.0,  # Proven effective angle
        flap_gap=0.030,  # Standard gap

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Conservative baseline wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Reliable Cl: {result['metrics']['Cl']:.3f}, Safety: HIGH")
    else:
        print(f"Invalid: {result['error']}")