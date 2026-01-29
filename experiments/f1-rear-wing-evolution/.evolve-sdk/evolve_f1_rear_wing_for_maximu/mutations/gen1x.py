#!/usr/bin/env python3
"""
Hybrid High-Performance Wing
Strategy: Combines aggressive downforce, bioinspired distribution, and slot optimization
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
    """
    Hybrid wing combining:
    - gen0_a: Aggressive angles and high camber for maximum downforce
    - gen0_e: Bioinspired camber distribution for efficiency
    - gen0_d: Mathematical optimization and ultra-tight slot gap
    """

    # From gen0_d: Mathematical sine wave function for optimized distribution
    def sine_camber(base, amplitude, phase=0):
        return [base + amplitude * np.sin(i * np.pi/3 + phase) for i in range(4)]

    # From gen0_e: Bird-inspired peak at 25% chord, but enhanced with gen0_a's aggression
    # From gen0_a: High base camber values for maximum downforce
    hybrid_upper = [0.19, 0.26, 0.22, 0.14]  # Bird distribution + Monaco aggression
    hybrid_lower = [-0.08, -0.09, -0.07, -0.03]  # Balanced from all parents

    # From gen0_d: Sine-enhanced flap camber for mathematical optimization
    flap_upper_base = sine_camber(0.16, 0.03)  # Mathematical enhancement
    flap_lower_enhanced = [-0.09, -0.07, -0.05, -0.02]  # From gen0_a's strength

    return F1RearWing(
        # Main element: Hybrid of bioinspired distribution + aggressive camber
        main_upper=hybrid_upper,
        main_lower=hybrid_lower,
        main_angle=16.0,  # Between gen0_e's natural 13° and gen0_a's aggressive 18°

        # Flap: Mathematical optimization + aggressive pressure differential
        flap_upper=flap_upper_base,
        flap_lower=flap_lower_enhanced,
        flap_angle=35.0,  # Between gen0_e's 28° and gen0_a's 40°
        flap_gap=0.022,  # Average of gen0_d's ultra-tight 0.020 and gen0_a's 0.025

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Hybrid wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Hybrid Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
        print(f"Efficiency L/D: {result['metrics']['L_D']:.2f}, Gap: {result['metrics']['flap_gap']*100:.1f}%")
    else:
        print(f"Invalid: {result['error']}")