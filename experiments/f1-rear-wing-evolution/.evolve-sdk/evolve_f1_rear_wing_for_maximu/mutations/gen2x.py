#!/usr/bin/env python3
"""
Hybrid High-Performance Wing (Gen 2 Crossover)
Strategy: Combines aggressive downforce, bioinspired efficiency, and slot optimization
Parents: gen0_a (high downforce), gen0_e (bioinspired), gen0_d (slot effect)
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
    - High camber strategy from gen0_a for maximum downforce
    - Front-loading distribution from gen0_e for natural efficiency
    - Optimized slot gap from gen0_d for circulation control
    """

    # From gen0_e: Use bioinspired front-loading but with gen0_a's aggressive camber levels
    # Peak at 25% chord like birds, but with high-downforce magnitudes from gen0_a
    hybrid_upper = [0.19, 0.26, 0.22, 0.14]  # Bird-like distribution with Monaco aggression
    hybrid_lower = [-0.08, -0.09, -0.07, -0.03]  # Balanced lower surface

    # From gen0_d: Mathematical optimization of flap camber using scaled sine approach
    base_flap_upper = 0.16
    flap_amplitude = 0.04
    flap_upper_hybrid = [
        base_flap_upper + flap_amplitude * np.sin(i * np.pi/3) for i in range(4)
    ]

    return F1RearWing(
        # Main element: Hybrid of bioinspired distribution with aggressive camber
        main_upper=hybrid_upper,
        main_lower=hybrid_lower,
        main_angle=15.5,  # Balance between gen0_e (13°) and gen0_a (18°)

        # Flap: Mathematical precision from gen0_d with high angles from gen0_a
        flap_upper=[round(x, 3) for x in flap_upper_hybrid],  # Sine-optimized camber
        flap_lower=[-0.09, -0.07, -0.05, -0.02],  # Strong pressure gradient from gen0_a
        flap_angle=35.0,  # Between gen0_d (32°) and gen0_a (40°) for optimal performance
        flap_gap=0.023,  # Optimized between gen0_d (0.020) and gen0_a (0.025)

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Hybrid crossover wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
        print(f"L/D ratio: {result['metrics']['L_D']:.2f}")
        print(f"Gap: {result['metrics']['flap_gap']*100:.1f}%")
    else:
        print(f"Invalid: {result['error']}")