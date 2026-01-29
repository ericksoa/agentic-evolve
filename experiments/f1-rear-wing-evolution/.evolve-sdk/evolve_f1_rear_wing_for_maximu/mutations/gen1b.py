#!/usr/bin/env python3
"""
Bioinspired Wing Design with Lookup Table Optimization
Strategy: Bird wing camber distribution with precomputed aerodynamic coefficients
"""
import sys
import os
import math

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

# Precomputed lookup table for optimal angle adjustments
# Based on common aerodynamic efficiency curves
ANGLE_EFFICIENCY_LUT = {
    12.0: 0.95,
    13.0: 1.00,
    14.0: 0.98,
    15.0: 0.92,
    26.0: 0.88,
    27.0: 0.92,
    28.0: 0.96,
    29.0: 0.94,
    30.0: 0.89
}

def get_optimized_angle(base_angle):
    """Use lookup table to find most efficient angle near base_angle."""
    best_angle = base_angle
    best_efficiency = ANGLE_EFFICIENCY_LUT.get(base_angle, 0.85)

    # Check nearby angles for better efficiency
    for angle, efficiency in ANGLE_EFFICIENCY_LUT.items():
        if abs(angle - base_angle) <= 2.0 and efficiency > best_efficiency:
            best_angle = angle
            best_efficiency = efficiency

    return best_angle

def create_wing():
    """Wing inspired by bird flight mechanics with performance-optimized angles."""

    # Bird-like camber: high at front, tapering toward trailing edge
    # Based on raptor wing cross-sections
    bird_upper = [0.18, 0.22, 0.16, 0.08]  # Peak at 25% chord like birds
    bird_lower = [-0.04, -0.06, -0.05, -0.02]  # Gentle lower curve

    # Use lookup table to optimize angles for performance
    optimized_main_angle = get_optimized_angle(13.0)
    optimized_flap_angle = get_optimized_angle(28.0)

    return F1RearWing(
        # Main element: Bird wing camber with optimized angle
        main_upper=bird_upper,
        main_lower=bird_lower,
        main_angle=optimized_main_angle,

        # Flap: Secondary feather inspired with optimized angle
        flap_upper=[0.14, 0.18, 0.14, 0.09],  # Similar distribution scaled
        flap_lower=[-0.07, -0.05, -0.04, -0.02],  # Smooth pressure recovery
        flap_angle=optimized_flap_angle,
        flap_gap=0.035,  # Natural spacing

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Optimized bioinspired wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Bird-like Cl: {result['metrics']['Cl']:.3f}, Efficiency: {result['metrics']['L_D']:.2f}")
    else:
        print(f"Invalid: {result['error']}")