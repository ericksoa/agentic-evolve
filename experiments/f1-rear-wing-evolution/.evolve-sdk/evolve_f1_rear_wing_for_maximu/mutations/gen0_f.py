#!/usr/bin/env python3
"""
Supercritical Airfoil Inspired Wing
Strategy: Flat-top design with rear loading for shock-free operation
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Wing based on supercritical airfoil concepts for high-speed efficiency."""

    # Supercritical design: flat top, rear camber loading
    # Delays shock formation and reduces pressure drag

    return F1RearWing(
        # Main element: Supercritical profile
        main_upper=[0.08, 0.10, 0.15, 0.18],  # Rear-loaded camber
        main_lower=[-0.05, -0.07, -0.06, -0.03],  # Strong rear pressure recovery
        main_angle=11.0,  # Moderate angle

        # Flap: Complementary rear loading
        flap_upper=[0.09, 0.11, 0.14, 0.16],  # Progressive rear loading
        flap_lower=[-0.06, -0.05, -0.03, -0.01],  # Smooth pressure recovery
        flap_angle=24.0,  # Balanced angle
        flap_gap=0.038,  # Moderate gap

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Supercritical wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"High-speed Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
    else:
        print(f"Invalid: {result['error']}")