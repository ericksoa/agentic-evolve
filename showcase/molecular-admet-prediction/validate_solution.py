#!/usr/bin/env python3
"""
Quick validation script to test if a solution runs without errors.
Used by evolution to pre-filter broken solutions.

Usage:
    python validate_solution.py solution.py

Exit codes:
    0 = Valid (runs without errors)
    1 = Invalid (import/syntax/runtime error)
"""

import argparse
import importlib.util
import sys
import warnings
warnings.filterwarnings('ignore')

# Test SMILES - simple molecules that should work with any RDKit function
TEST_SMILES = [
    'CCO',  # Ethanol - simple
    'c1ccccc1',  # Benzene - aromatic
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin - drug-like
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine - heterocyclic
]
TEST_LABELS = [0, 1, 0, 1]


def validate_solution(solution_path: str, verbose: bool = False) -> tuple[bool, str]:
    """
    Validate a solution by:
    1. Loading the module (catches import/syntax errors)
    2. Instantiating HERGPredictor (catches __init__ errors)
    3. Calling fit() on small data (catches training errors)
    4. Calling predict_proba() (catches inference errors)

    Returns (is_valid, error_message)
    """
    try:
        # 1. Load module
        if verbose:
            print(f"Loading {solution_path}...")
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 2. Check for HERGPredictor class
        if not hasattr(module, 'HERGPredictor'):
            return False, "Missing HERGPredictor class"

        # 3. Instantiate
        if verbose:
            print("Instantiating HERGPredictor...")
        predictor = module.HERGPredictor()

        # 4. Check required methods
        if not hasattr(predictor, 'fit'):
            return False, "Missing fit() method"
        if not hasattr(predictor, 'predict_proba'):
            return False, "Missing predict_proba() method"

        # 5. Test fit
        if verbose:
            print("Testing fit()...")
        predictor.fit(TEST_SMILES * 5, TEST_LABELS * 5)  # 20 samples

        # 6. Test predict_proba
        if verbose:
            print("Testing predict_proba()...")
        proba = predictor.predict_proba(TEST_SMILES)

        # 7. Validate output shape
        if len(proba) != len(TEST_SMILES):
            return False, f"predict_proba returned {len(proba)} values, expected {len(TEST_SMILES)}"

        # 8. Validate output range
        proba_arr = list(proba)
        if any(p < 0 or p > 1 for p in proba_arr):
            return False, f"predict_proba values out of [0,1] range"

        if verbose:
            print(f"Predictions: {[f'{p:.3f}' for p in proba_arr]}")

        return True, "OK"

    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Validate hERG solution')
    parser.add_argument('solution', type=str, help='Path to solution Python file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    is_valid, message = validate_solution(args.solution, args.verbose)

    if is_valid:
        print(f"VALID: {args.solution}")
        return 0
    else:
        print(f"INVALID: {args.solution}")
        print(f"  Error: {message}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
