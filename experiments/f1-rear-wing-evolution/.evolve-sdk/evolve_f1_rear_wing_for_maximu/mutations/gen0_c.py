#!/usr/bin/env python3
"""
Efficiency-Optimized Wing
Strategy: Maximize L/D ratio with optimized slot gap and moderate camber
"""
import sys
import os
import math

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Wing optimized for maximum aerodynamic efficiency."""
    # Use golden ratio principles for camber distribution
    phi = 1.618

    return F1RearWing(
        # Main element: Carefully tuned camber progression
        main_upper=[0.12, 0.16, 0.14, 0.10],  # Smooth camber curve
        main_lower=[-0.06, -0.05, -0.04, -0.03],  # Gradual lower surface
        main_angle=10.0,  # Moderate angle for efficiency

        # Flap: Optimized for slot interaction
        flap_upper=[0.10, 0.13, 0.11, 0.08],  # Balanced camber
        flap_lower=[-0.05, -0.04, -0.03, -0.02],  # Smooth lower
        flap_angle=22.0,  # Near-optimal angle
        flap_gap=0.032,  # Optimal slot gap (3.2% chord)

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Efficiency-optimized wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"L/D: {result['metrics']['L_D']:.2f}, Cl²/Cd: {result['metrics']['Cl2_Cd']:.2f}")
    else:
        print(f"Invalid: {result['error']}")