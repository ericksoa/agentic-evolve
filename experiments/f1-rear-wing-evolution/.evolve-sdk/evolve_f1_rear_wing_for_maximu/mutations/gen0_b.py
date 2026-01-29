#!/usr/bin/env python3
"""
Monza Low-Drag Optimized Wing
Strategy: Minimal drag with flat profiles and conservative angles
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

def create_wing():
    """Low-drag wing optimized for Monza-style power circuits."""
    return F1RearWing(
        # Main element: Thin, low camber
        main_upper=[0.08, 0.10, 0.09, 0.06],  # Very low camber
        main_lower=[-0.03, -0.02, -0.02, -0.01],  # Minimal lower curvature
        main_angle=6.0,  # Conservative angle

        # Flap: Minimal but still effective
        flap_upper=[0.06, 0.08, 0.07, 0.04],  # Low camber flap
        flap_lower=[-0.02, -0.02, -0.01, -0.01],  # Flat lower surface
        flap_angle=15.0,  # Minimum effective angle
        flap_gap=0.045,  # Wide gap reduces interference

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Monza-optimized wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cl: {result['metrics']['Cl']:.3f}, Cd: {result['metrics']['Cd']:.3f}")
    else:
        print(f"Invalid: {result['error']}")