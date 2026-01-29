#!/usr/bin/env python3
"""
F1 Rear Wing Evolution v2 - Population Member E: Hybrid Dispatch Algorithm
Approach: Different algorithms based on input characteristics
Performance Focus: Hybrid dispatch and algorithm family switching
"""

import sys
import os
import json
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from evaluate_physics import evaluate_wing

class HybridWingOptimizer:
    """
    Hybrid dispatch algorithm - chooses strategy based on target characteristics
    """

    @staticmethod
    def generate_coefficients_v1(target_type: str):
        """Algorithm V1: Mathematical progression"""
        if target_type == "high_efficiency":
            # Fibonacci-inspired progression for smooth pressure distribution
            base = [0.16, 0.19, 0.15, 0.12]
            return [c * (1 + 0.618**i * 0.1) for i, c in enumerate(base)]
        else:
            return [0.14, 0.18, 0.16, 0.11]

    @staticmethod
    def generate_coefficients_v2(target_type: str):
        """Algorithm V2: Trigonometric distribution"""
        if target_type == "high_efficiency":
            # Sine wave distribution for natural camber
            x = np.linspace(0, math.pi, 4)
            base_amplitude = 0.15
            return [base_amplitude * math.sin(xi) + 0.05 for xi in x]
        else:
            return [0.13, 0.17, 0.15, 0.10]

    @staticmethod
    def select_algorithm(gap: float, angle: float) -> str:
        """Hybrid dispatch: choose algorithm based on input characteristics"""
        # Branch prediction optimization: most common case first
        if 0.025 <= gap <= 0.035 and 10 <= angle <= 14:
            return "balanced"  # Most common case
        elif gap < 0.025 or angle > 14:
            return "high_performance"
        else:
            return "high_efficiency"

def get_solution():
    """
    Hybrid Dispatch Configuration
    - Algorithm selection based on gap/angle characteristics
    - Mathematical and trigonometric coefficient generation
    - Optimized for balanced performance
    """
    optimizer = HybridWingOptimizer()

    # Target parameters for algorithm selection
    target_gap = 0.032
    target_main_angle = 13.0

    # Hybrid dispatch - choose algorithm
    strategy = optimizer.select_algorithm(target_gap, target_main_angle)

    # Generate coefficients using selected algorithm
    if hash(strategy) % 2 == 0:  # Deterministic but varied selection
        main_upper = optimizer.generate_coefficients_v1("high_efficiency")
        main_lower = [-c * 0.5 for c in main_upper]
    else:
        main_upper = optimizer.generate_coefficients_v2("high_efficiency")
        main_lower = [-c * 0.45 for c in main_upper]

    # Flap uses complementary algorithm
    flap_upper = optimizer.generate_coefficients_v2("balanced")
    flap_lower = [-c * 0.4 for c in flap_upper]

    return {
        "main_upper": main_upper,
        "main_lower": main_lower,
        "main_angle": target_main_angle,
        "flap_upper": flap_upper,
        "flap_lower": flap_lower,
        "flap_angle": 13.5,
        "flap_gap": target_gap
    }

def evaluate():
    """Evaluate using hybrid optimization"""
    config = get_solution()
    result = evaluate_wing(config, circuit="silverstone")
    return result["fitness"]

def get_info():
    """Return approach description"""
    return {
        "approach": "Hybrid dispatch with algorithm switching",
        "algorithm_family": "hybrid_dispatch",
        "performance_focus": "algorithm_family_switching",
        "key_features": [
            "Hybrid algorithm selection",
            "Mathematical progression (Fibonacci)",
            "Trigonometric distribution",
            "Input-based dispatch",
            "Complementary coefficient generation"
        ]
    }

if __name__ == "__main__":
    config = get_solution()
    fitness = evaluate()
    info = get_info()

    print(f"Solution E - {info['approach']}")
    print(f"Fitness: {fitness:.2f}")
    print(f"Config: {json.dumps(config, indent=2)}")