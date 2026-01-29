#!/usr/bin/env python3
"""
SDK-compatible benchmark harness for MALLORN TDE classification.

This module provides evaluation for the /evolve-sdk skill with:
- Holdout validation to detect overfitting
- JSON output for SDK parsing
- Acceptance criteria checking

Usage:
    python evolve_benchmark.py --evaluate path/to/solution.py --json
    python evolve_benchmark.py --evaluate path/to/solution.py --class-name Gen11_Candidate
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_batch
from data_loader import load_single_split

# =============================================================================
# SPLIT CONFIGURATION (from evolve_config.json)
# =============================================================================

CV_SPLITS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
HOLDOUT_SPLITS = [14, 17, 20]

# Acceptance criteria
MIN_HOLDOUT_F1 = 0.40
MAX_CV_HOLDOUT_GAP = 0.10


# =============================================================================
# SOLUTION LOADING
# =============================================================================

def load_solution_class(solution_path: str, class_name: str | None = None) -> Any:
    """
    Dynamically load a classifier class from a solution file.

    Args:
        solution_path: Path to the Python file containing the classifier
        class_name: Name of the class to load (auto-detects if None)

    Returns:
        The classifier class (not an instance)
    """
    path = Path(solution_path)
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)

    # Find the classifier class
    if class_name:
        if not hasattr(module, class_name):
            raise ValueError(f"Class {class_name} not found in {solution_path}")
        return getattr(module, class_name)

    # Auto-detect: look for classes ending in "Classifier" or "LROnly"
    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name != "BaseEstimator" and name != "ClassifierMixin":
            if "Classifier" in name or "LROnly" in name or "Candidate" in name:
                candidates.append((name, obj))

    if not candidates:
        raise ValueError(f"No classifier class found in {solution_path}")

    # Prefer Gen10, Gen11, etc. if present
    for name, cls in sorted(candidates, reverse=True):
        if name.startswith("Gen"):
            return cls

    # Otherwise return the last one (most recently defined)
    return candidates[-1][1]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_split(
    classifier_class: type,
    data_dir: str,
    split_num: int,
    n_cv_folds: int = 3,
    verbose: bool = False
) -> dict:
    """
    Evaluate a classifier on a single split using CV.

    Returns:
        Dict with f1_mean, f1_std, and per-fold scores
    """
    # Load data
    train_lc, test_lc, train_meta, test_meta = load_single_split(data_dir, split_num)

    if train_meta is None or len(train_meta) == 0:
        return {"f1_mean": 0.0, "f1_std": 0.0, "error": "No training data"}

    # Extract features
    X = extract_features_batch(
        train_lc,
        metadata=train_meta.set_index('object_id'),
        use_evolved=True,
        verbose=False
    )

    # Get labels
    y = train_meta.set_index('object_id')['target']

    # Align
    common_ids = X.index.intersection(y.index)
    X = X.loc[common_ids]
    y = y.loc[common_ids]

    # Check for positive samples
    n_pos = (y == 1).sum()
    if n_pos < n_cv_folds:
        # Not enough positives for CV, use full training
        try:
            clf = classifier_class()
            clf.fit(X, y)
            y_pred = clf.predict(X)
            f1 = f1_score(y, y_pred, zero_division=0)
            return {"f1_mean": f1, "f1_std": 0.0, "n_pos": n_pos, "note": "no_cv"}
        except Exception as e:
            return {"f1_mean": 0.0, "f1_std": 0.0, "error": str(e)}

    # Stratified CV
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        try:
            clf = classifier_class()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            fold_scores.append(f1)
        except Exception as e:
            if verbose:
                print(f"  Split {split_num} fold error: {e}")
            fold_scores.append(0.0)

    return {
        "f1_mean": float(np.mean(fold_scores)),
        "f1_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "n_pos": n_pos
    }


def run_full_evaluation(
    classifier_class: type,
    data_dir: str = "../data",
    verbose: bool = True
) -> dict:
    """
    Run full evaluation with CV and holdout validation.

    Returns:
        Dict with cv_f1, holdout_f1, gap, valid, and detailed results
    """
    results = {
        "cv_splits": {},
        "holdout_splits": {},
        "cv_f1_mean": 0.0,
        "cv_f1_std": 0.0,
        "holdout_f1_mean": 0.0,
        "holdout_f1_std": 0.0,
        "gap": 0.0,
        "valid": False,
        "acceptance": {}
    }

    # Evaluate on CV splits
    cv_scores = []
    if verbose:
        print(f"Evaluating on {len(CV_SPLITS)} CV splits...")

    for split_num in CV_SPLITS:
        try:
            split_result = evaluate_on_split(
                classifier_class, data_dir, split_num, verbose=verbose
            )
            results["cv_splits"][split_num] = split_result
            cv_scores.append(split_result["f1_mean"])
            if verbose:
                print(f"  Split {split_num:02d}: F1={split_result['f1_mean']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  Split {split_num:02d}: ERROR - {e}")
            results["cv_splits"][split_num] = {"error": str(e)}

    # Evaluate on holdout splits
    holdout_scores = []
    if verbose:
        print(f"\nEvaluating on {len(HOLDOUT_SPLITS)} holdout splits...")

    for split_num in HOLDOUT_SPLITS:
        try:
            split_result = evaluate_on_split(
                classifier_class, data_dir, split_num, verbose=verbose
            )
            results["holdout_splits"][split_num] = split_result
            holdout_scores.append(split_result["f1_mean"])
            if verbose:
                print(f"  Split {split_num:02d}: F1={split_result['f1_mean']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  Split {split_num:02d}: ERROR - {e}")
            results["holdout_splits"][split_num] = {"error": str(e)}

    # Calculate aggregates
    if cv_scores:
        results["cv_f1_mean"] = float(np.mean(cv_scores))
        results["cv_f1_std"] = float(np.std(cv_scores))

    if holdout_scores:
        results["holdout_f1_mean"] = float(np.mean(holdout_scores))
        results["holdout_f1_std"] = float(np.std(holdout_scores))

    # Calculate gap (CV - Holdout)
    # Negative gap means holdout is better (good!)
    results["gap"] = results["cv_f1_mean"] - results["holdout_f1_mean"]

    # Check acceptance criteria
    holdout_ok = results["holdout_f1_mean"] >= MIN_HOLDOUT_F1
    gap_ok = results["gap"] <= MAX_CV_HOLDOUT_GAP

    results["acceptance"] = {
        "holdout_f1_check": {
            "threshold": MIN_HOLDOUT_F1,
            "value": results["holdout_f1_mean"],
            "passed": holdout_ok
        },
        "gap_check": {
            "threshold": MAX_CV_HOLDOUT_GAP,
            "value": results["gap"],
            "passed": gap_ok
        }
    }

    results["valid"] = holdout_ok and gap_ok

    # Fitness is holdout F1 (what we optimize for)
    results["fitness"] = results["holdout_f1_mean"]

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SDK-compatible benchmark for MALLORN TDE classification"
    )
    parser.add_argument(
        "--evaluate", "-e",
        type=str,
        required=True,
        help="Path to solution file to evaluate"
    )
    parser.add_argument(
        "--class-name", "-c",
        type=str,
        default=None,
        help="Specific class name to evaluate (auto-detects if not specified)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="../data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet and not args.json

    # Load solution
    try:
        classifier_class = load_solution_class(args.evaluate, args.class_name)
        if verbose:
            print(f"Loaded: {classifier_class.__name__}")
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "valid": False, "fitness": 0.0}, cls=NumpyEncoder))
        else:
            print(f"Error loading solution: {e}")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_full_evaluation(
            classifier_class,
            data_dir=args.data_dir,
            verbose=verbose
        )
        results["solution_file"] = args.evaluate
        results["class_name"] = classifier_class.__name__
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "valid": False, "fitness": 0.0}, cls=NumpyEncoder))
        else:
            print(f"Evaluation error: {e}")
        sys.exit(1)

    # Output results
    if args.json:
        print(json.dumps(results, indent=2, cls=NumpyEncoder))
    else:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Solution: {results['class_name']}")
        print(f"CV F1:      {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std']:.4f})")
        print(f"Holdout F1: {results['holdout_f1_mean']:.4f} (+/- {results['holdout_f1_std']:.4f})")
        print(f"Gap:        {results['gap']:.4f}")
        print()
        print("Acceptance Checks:")
        for check_name, check in results["acceptance"].items():
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {check_name}: {check['value']:.4f} vs {check['threshold']:.4f} -> {status}")
        print()
        print(f"VALID: {results['valid']}")
        print(f"FITNESS: {results['fitness']:.4f}")

    # Exit with appropriate code
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
