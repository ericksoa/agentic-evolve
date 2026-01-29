#!/usr/bin/env python3
"""
F1 Rear Wing Evolution v2 - Generation 1 Mutation A: Aggressive Downforce
Approach: High-downforce configuration optimized for Silverstone's 60/40 downforce/drag weighting
Performance Focus: Maximize Cl while maintaining reasonable L/D for circuit-specific optimization
"""

import sys
import os
import json
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from evaluate_physics import evaluate_wing

def get_solution():
    """
    Aggressive High-Downforce Configuration
    - Increased main wing angle from 10° to 15° for more downforce
    - More aggressive flap angle from 12° to 18° for maximum Cl
    - Higher camber CST coefficients for both elements
    - Reduced gap to 2.5% for stronger slot effect
    - Optimized for Silverstone's downforce-biased weighting (60% downforce vs 40% drag)
    """
    return {
        "main_upper": [0.22, 0.28, 0.25, 0.18],  # Increased camber for higher Cl
        "main_lower": [-0.12, -0.09, -0.06, -0.03],  # More negative camber
        "main_angle": 15.0,  # Increased from 10° for more downforce
        "flap_upper": [0.18, 0.22, 0.18, 0.12],  # Higher camber flap
        "flap_lower": [-0.08, -0.06, -0.04, -0.02],  # More aggressive lower surface
        "flap_angle": 18.0,  # Increased from 12° for maximum downforce
        "flap_gap": 0.025  # Reduced gap for stronger slot effect
    }

def evaluate():
    """Evaluate this solution using physics-based fitness"""
    config = get_solution()
    result = evaluate_wing(config, circuit="silverstone")
    return result["fitness"]

def get_info():
    """Return approach description"""
    return {
        "approach": "Aggressive high-downforce configuration optimized for Silverstone",
        "algorithm_family": "high_downforce_aggressive",
        "performance_focus": "maximum_downforce_silverstone",
        "key_features": [
            "Aggressive angles (main=15°, flap=18°)",
            "High-camber CST coefficient distribution",
            "Reduced gap (2.5% chord) for slot effect",
            "Circuit-specific optimization for downforce priority"
        ]
    }

if __name__ == "__main__":
    config = get_solution()
    fitness = evaluate()
    info = get_info()

    print(f"Mutation 1A - {info['approach']}")
    print(f"Fitness: {fitness:.2f}")
    print(f"Config: {json.dumps(config, indent=2)}")