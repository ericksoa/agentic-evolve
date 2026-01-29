#!/usr/bin/env python3
"""
Bioinspired Wing Design
Strategy: Bird wing camber distribution with adaptive angles
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
    """Wing inspired by bird flight mechanics and natural airfoil shapes."""

    # Bird-like camber: high at front, tapering toward trailing edge
    # Based on raptor wing cross-sections
    bird_upper = [0.18, 0.22, 0.16, 0.08]  # Peak at 25% chord like birds
    bird_lower = [-0.04, -0.06, -0.05, -0.02]  # Gentle lower curve

    return F1RearWing(
        # Main element: Bird wing camber
        main_upper=bird_upper,
        main_lower=bird_lower,
        main_angle=13.0,  # Natural angle of attack

        # Flap: Secondary feather inspired
        flap_upper=[0.14, 0.18, 0.14, 0.09],  # Similar distribution scaled
        flap_lower=[-0.07, -0.05, -0.04, -0.02],  # Smooth pressure recovery
        flap_angle=28.0,  # Biomimetic angle
        flap_gap=0.035,  # Natural spacing

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Bioinspired wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Bird-like Cl: {result['metrics']['Cl']:.3f}, Efficiency: {result['metrics']['L_D']:.2f}")
    else:
        print(f"Invalid: {result['error']}")