#!/usr/bin/env python3
"""
Extreme Performance Wing
Strategy: Push boundaries of regulations for maximum performance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Wing that pushes regulatory limits for maximum performance."""

    # Maximum allowable configurations
    # Right at the edge of what's legal

    return F1RearWing(
        # Main element: Maximum thickness, high camber
        main_upper=[0.22, 0.24, 0.20, 0.16],  # Very high camber near limit
        main_lower=[-0.11, -0.09, -0.07, -0.04],  # Strong lower surface
        main_angle=22.0,  # Just below stall

        # Flap: Maximum effective angle
        flap_upper=[0.20, 0.23, 0.19, 0.14],  # High camber flap
        flap_lower=[-0.09, -0.07, -0.05, -0.02],  # Deep lower surface
        flap_angle=42.0,  # Near maximum before major stall
        flap_gap=0.018,  # Very tight for maximum energy

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Extreme performance wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Extreme Cl: {result['metrics']['Cl']:.3f}, Risk level: HIGH")
    else:
        print(f"Invalid: {result['error']}")