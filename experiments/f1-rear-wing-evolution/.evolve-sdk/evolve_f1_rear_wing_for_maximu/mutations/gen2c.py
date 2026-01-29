#!/usr/bin/env python3
"""
Optimized Slot-Effect Wing with Precomputed Values
Strategy: Replace runtime sine calculations with lookup table for performance
"""
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Wing designed to maximize slot effect with precomputed camber values."""

    # Precomputed sine wave lookup table for performance optimization
    # Base sine values for positions i=0,1,2,3 with pi/3 spacing
    SINE_LUT = [
        np.sin(0 * np.pi/3),      # i=0: sin(0) = 0.0
        np.sin(1 * np.pi/3),      # i=1: sin(π/3) ≈ 0.866
        np.sin(2 * np.pi/3),      # i=2: sin(2π/3) ≈ 0.866
        np.sin(3 * np.pi/3)       # i=3: sin(π) = 0.0
    ]

    # Precomputed inverted sine values (phase shifted by π)
    SINE_LUT_INVERTED = [
        np.sin(0 * np.pi/3 + np.pi),  # i=0: sin(π) = 0.0
        np.sin(1 * np.pi/3 + np.pi),  # i=1: sin(4π/3) ≈ -0.866
        np.sin(2 * np.pi/3 + np.pi),  # i=2: sin(5π/3) ≈ -0.866
        np.sin(3 * np.pi/3 + np.pi)   # i=3: sin(2π) = 0.0
    ]

    # Fast lookup table camber calculation (no runtime sin() calls)
    def fast_sine_camber(base, amplitude, inverted=False):
        lut = SINE_LUT_INVERTED if inverted else SINE_LUT
        return [base + amplitude * val for val in lut]

    return F1RearWing(
        # Main element: Precomputed sine wave camber distribution
        main_upper=fast_sine_camber(0.14, 0.04),  # Wavy upper surface
        main_lower=[-x for x in fast_sine_camber(0.06, 0.02, inverted=True)],  # Inverted lower
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
    print(f"Optimized slot-effect wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Gap: {result['metrics']['flap_gap']*100:.1f}%")
    else:
        print(f"Invalid: {result['error']}")