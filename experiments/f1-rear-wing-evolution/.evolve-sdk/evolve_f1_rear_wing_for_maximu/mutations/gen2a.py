#!/usr/bin/env python3
"""
Silverstone Balanced Efficiency Wing
Strategy: Optimized for 60/40 downforce/drag weighting with efficiency focus
Algorithm Change: Monaco aggressive → Silverstone balanced approach
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Silverstone-optimized wing with balanced downforce/efficiency."""
    return F1RearWing(
        # Main element: Moderate camber, efficient angle
        main_upper=[0.16, 0.22, 0.20, 0.14],  # Reduced peak camber for efficiency
        main_lower=[-0.09, -0.08, -0.06, -0.03],  # Moderate lower surface
        main_angle=14.0,  # Below stall threshold for clean flow

        # Flap: Balanced for efficiency with good downforce
        flap_upper=[0.14, 0.19, 0.16, 0.12],  # Moderate camber for efficiency
        flap_lower=[-0.08, -0.06, -0.04, -0.02],  # Clean lower surface
        flap_angle=28.0,  # Reduced from 40° for better L/D
        flap_gap=0.035,  # Slightly larger gap for better slot flow

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Silverstone-optimized wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
    else:
        print(f"Invalid: {result['error']}")