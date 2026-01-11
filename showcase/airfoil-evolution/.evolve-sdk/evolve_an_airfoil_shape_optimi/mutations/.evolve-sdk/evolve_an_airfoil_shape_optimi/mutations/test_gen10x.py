#!/usr/bin/env python3
"""Quick test for gen10x hybrid"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    # Test core imports
    from airfoil import Airfoil
    from evaluate import evaluate_airfoil, DesignRequirements
    print("✓ Core imports successful")

    # Test hybrid implementation imports
    from gen10x import SupremeHybridSobol, SupremeNelderMead, SupremeCSTInitializer
    print("✓ Hybrid class imports successful")

    # Test Sobol generation
    sobol = SupremeHybridSobol(dimension=10)
    points = sobol.generate_points(5)
    assert points.shape == (5, 10), f"Expected (5, 10), got {points.shape}"
    assert np.all((points >= 0) & (points <= 1)), "Points should be in [0, 1]"
    print("✓ Sobol generation works")

    # Test CST initializer
    initializer = SupremeCSTInitializer(seed=42)

    # Test coefficient conversion
    coeffs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
    airfoil = initializer.coeffs_to_airfoil(coeffs)
    assert len(airfoil.upper_coeffs) == 5, "Should have 5 upper coeffs"
    assert len(airfoil.lower_coeffs) == 5, "Should have 5 lower coeffs"
    print("✓ Coefficient conversion works")

    # Test small population generation
    small_pop = initializer.generate_sobol_population(3)
    assert len(small_pop) == 3, f"Expected 3 airfoils, got {len(small_pop)}"
    print("✓ Small population generation works")

    print("\n✅ All basic tests passed! Hybrid implementation appears functional.")

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()