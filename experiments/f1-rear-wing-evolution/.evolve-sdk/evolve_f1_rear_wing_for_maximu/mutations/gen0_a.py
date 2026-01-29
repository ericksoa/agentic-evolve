#!/usr/bin/env python3
"""
Monaco High-Downforce Optimized Wing
Strategy: Maximum downforce with aggressive angles and high camber
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """High-downforce wing optimized for Monaco-style circuits."""
    return F1RearWing(
        # Main element: High camber, aggressive angle
        main_upper=[0.20, 0.28, 0.26, 0.18],  # Very high camber
        main_lower=[-0.12, -0.10, -0.08, -0.04],  # Strong lower surface
        main_angle=18.0,  # Near stall limit

        # Flap: Maximum legal angle with high camber
        flap_upper=[0.18, 0.25, 0.20, 0.15],  # High camber flap
        flap_lower=[-0.10, -0.08, -0.05, -0.02],  # Deep lower surface
        flap_angle=40.0,  # Aggressive angle
        flap_gap=0.025,  # Tight slot for energy

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Monaco-optimized wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
    else:
        print(f"Invalid: {result['error']}")