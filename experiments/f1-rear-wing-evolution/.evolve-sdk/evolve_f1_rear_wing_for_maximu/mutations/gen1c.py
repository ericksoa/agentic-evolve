#!/usr/bin/env python3
"""
Aggressive Slot-Effect Wing - Lookup Table Optimized
Strategy: Use precomputed lookup table to eliminate runtime trigonometric calculations
"""
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

# Precomputed sine values for performance optimization
# sin(i * π/3) for i in [0, 1, 2, 3]
SINE_LUT = [0.0, 0.8660254037844387, 0.8660254037844386, 1.2246467991473532e-16]
# sin(i * π/3 + π) for i in [0, 1, 2, 3]
SINE_LUT_PI = [-1.2246467991473532e-16, -0.8660254037844386, -0.8660254037844387, -0.0]

def create_wing():
    """Wing designed to maximize slot effect and circulation control."""

    # Use precomputed lookup table for camber (no runtime trig calculations)
    def lut_camber(base, amplitude, use_pi_phase=False):
        lut = SINE_LUT_PI if use_pi_phase else SINE_LUT
        return [base + amplitude * lut[i] for i in range(4)]

    return F1RearWing(
        # Main element: Lookup table camber distribution
        main_upper=lut_camber(0.14, 0.04),  # Wavy upper surface
        main_lower=[-x for x in lut_camber(0.06, 0.02, True)],  # Inverted lower
        main_angle=14.0,  # Moderate-high angle

        # Flap: High-energy capture design
        flap_upper=[0.16, 0.20, 0.18, 0.12],  # High front loading
        flap_lower=[-0.08, -0.06, -0.04, -0.02],  # Strong pressure side
        flap_angle=32.0,  # High angle to capture slot energy
        flap_gap=0.020,  # Very tight slot for high energy

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Slot-effect wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Gap: {result['metrics']['flap_gap']*100:.1f}%")
    else:
        print(f"Invalid: {result['error']}")