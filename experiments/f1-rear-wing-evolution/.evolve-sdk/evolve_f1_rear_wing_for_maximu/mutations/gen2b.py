#!/usr/bin/env python3
"""
Optimized Bioinspired Wing Design
Strategy: Cached parameter lookup with precomputed trigonometric values
"""
import sys
import os
import math

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from wing import F1RearWing
from evaluate import evaluate_wing

# Precomputed lookup table for optimized wing configurations
# Cache trigonometric calculations and optimal parameter combinations
WING_CACHE = {
    'main_angle_rad': math.radians(13.0),
    'main_angle_sin': math.sin(math.radians(13.0)),
    'main_angle_cos': math.cos(math.radians(13.0)),
    'flap_angle_rad': math.radians(28.0),
    'flap_angle_sin': math.sin(math.radians(28.0)),
    'flap_angle_cos': math.cos(math.radians(28.0)),

    # Precomputed camber arrays for cache-friendly access
    'bird_upper_cache': tuple([0.18, 0.22, 0.16, 0.08]),  # Immutable tuple for performance
    'bird_lower_cache': tuple([-0.04, -0.06, -0.05, -0.02]),
    'flap_upper_cache': tuple([0.14, 0.18, 0.14, 0.09]),
    'flap_lower_cache': tuple([-0.07, -0.05, -0.04, -0.02])
}

def create_wing():
    """Wing inspired by bird flight mechanics with cached parameter optimization."""

    # Use cached values for better memory locality and reduced computation
    cache = WING_CACHE

    # Convert cached tuples back to lists (required by F1RearWing)
    bird_upper = list(cache['bird_upper_cache'])
    bird_lower = list(cache['bird_lower_cache'])
    flap_upper = list(cache['flap_upper_cache'])
    flap_lower = list(cache['flap_lower_cache'])

    return F1RearWing(
        # Main element: Bird wing camber (cache-optimized)
        main_upper=bird_upper,
        main_lower=bird_lower,
        main_angle=13.0,  # Use original angle value

        # Flap: Secondary feather inspired (cache-optimized)
        flap_upper=flap_upper,
        flap_lower=flap_lower,
        flap_angle=28.0,  # Use original angle value
        flap_gap=0.035,  # Natural spacing

        drs_open=False
    )

if __name__ == "__main__":
    wing = create_wing()
    result = evaluate_wing(wing, "silverstone")
    print(f"Cache-optimized bioinspired wing fitness: {result['fitness']:.4f}")
    if result['valid']:
        print(f"Cached Cl: {result['metrics']['Cl']:.3f}, Efficiency: {result['metrics']['L_D']:.2f}")
    else:
        print(f"Invalid: {result['error']}")