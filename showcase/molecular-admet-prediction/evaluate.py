#!/usr/bin/env python3
"""
Evaluation script for hERG toxicity prediction models.
Uses the Therapeutics Data Commons (TDC) hERG benchmark.

Usage:
    python evaluate.py solution.py --json
    python evaluate.py solution.py --verbose
"""

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


def load_herg_data():
    """Load hERG dataset from TDC."""
    try:
        from tdc.single_pred import Tox
    except ImportError:
        print("Error: PyTDC not installed. Run: pip install PyTDC", file=sys.stderr)
        sys.exit(1)

    # Load hERG dataset
    data = Tox(name='hERG')
    split = data.get_split(method='scaffold', seed=42)

    return {
        'train': (split['train']['Drug'].tolist(), split['train']['Y'].tolist()),
        'valid': (split['valid']['Drug'].tolist(), split['valid']['Y'].tolist()),
        'test': (split['test']['Drug'].tolist(), split['test']['Y'].tolist())
    }


def load_solution(solution_path: str):
    """Dynamically load a solution module."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for HERGPredictor class
    if hasattr(module, 'HERGPredictor'):
        return module.HERGPredictor()
    else:
        raise ValueError(f"Solution must define a HERGPredictor class")


def evaluate_model(predictor, X_train, y_train, X_test, y_test, measure_time=True):
    """Evaluate a predictor on given data."""
    # Fit the model
    predictor.fit(X_train, y_train)

    # Measure inference time
    if measure_time:
        start = time.time()
        y_proba = predictor.predict_proba(X_test)
        inference_time = (time.time() - start) / len(X_test) * 1000  # ms per molecule
    else:
        y_proba = predictor.predict_proba(X_test)
        inference_time = 0

    # Ensure proper shape
    if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.ravel()

    y_proba = np.array(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    y_test = np.array(y_test)

    # Calculate metrics
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = 0.5  # Default if only one class present

    return {
        'roc_auc': float(roc_auc),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'inference_ms': float(inference_time)
    }


def run_cv_evaluation(predictor_class, X, y, n_splits=5):
    """Run cross-validation evaluation."""
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        predictor = predictor_class()
        X_train, X_val = X[train_idx].tolist(), X[val_idx].tolist()
        y_train, y_val = y[train_idx].tolist(), y[val_idx].tolist()

        fold_result = evaluate_model(predictor, X_train, y_train, X_val, y_val)
        cv_results.append(fold_result)

    # Aggregate CV results
    return {
        'mean_roc_auc': float(np.mean([r['roc_auc'] for r in cv_results])),
        'std_roc_auc': float(np.std([r['roc_auc'] for r in cv_results])),
        'mean_f1': float(np.mean([r['f1'] for r in cv_results])),
        'fold_results': cv_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate hERG prediction model')
    parser.add_argument('solution', type=str, help='Path to solution Python file')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--cv-only', action='store_true', help='Only run CV, skip test set')
    args = parser.parse_args()

    try:
        # Load data
        if args.verbose:
            print("Loading hERG dataset from TDC...")
        data = load_herg_data()

        X_train, y_train = data['train']
        X_valid, y_valid = data['valid']
        X_test, y_test = data['test']

        if args.verbose:
            print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
            print(f"Class balance (train): {sum(y_train)/len(y_train):.2%} positive")

        # Load solution
        if args.verbose:
            print(f"Loading solution from {args.solution}...")
        predictor = load_solution(args.solution)
        predictor_class = type(predictor)

        # Run CV on training data
        if args.verbose:
            print("Running 5-fold cross-validation...")
        cv_results = run_cv_evaluation(predictor_class, X_train, y_train, n_splits=5)

        # Evaluate on validation set (for model selection)
        if args.verbose:
            print("Evaluating on validation set...")
        predictor = predictor_class()
        valid_results = evaluate_model(predictor, X_train, y_train, X_valid, y_valid)

        # Evaluate on test set (final score)
        if not args.cv_only:
            if args.verbose:
                print("Evaluating on test set...")
            predictor = predictor_class()
            # Train on train+valid for final test
            X_trainval = X_train + X_valid
            y_trainval = y_train + y_valid
            test_results = evaluate_model(predictor, X_trainval, y_trainval, X_test, y_test)
        else:
            test_results = None

        # Get feature importance if available
        feature_importance = None
        if hasattr(predictor, 'get_feature_importance'):
            try:
                feature_importance = predictor.get_feature_importance()
            except Exception:
                pass

        # Compile results
        results = {
            'fitness': test_results['roc_auc'] if test_results else valid_results['roc_auc'],
            'roc_auc': test_results['roc_auc'] if test_results else valid_results['roc_auc'],
            'valid': True,
            'cv': cv_results,
            'validation': valid_results,
            'test': test_results,
            'feature_importance': feature_importance,
            'detailed': {
                'cv_mean_auc': cv_results['mean_roc_auc'],
                'cv_std_auc': cv_results['std_roc_auc'],
                'valid_auc': valid_results['roc_auc'],
                'test_auc': test_results['roc_auc'] if test_results else None,
                'inference_ms': test_results['inference_ms'] if test_results else valid_results['inference_ms']
            }
        }

        # Check validity constraints
        if results['detailed']['inference_ms'] > 100:
            results['valid'] = False
            results['error'] = f"Inference too slow: {results['detailed']['inference_ms']:.1f}ms > 100ms limit"

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"hERG Toxicity Prediction Results")
            print(f"{'='*60}")
            print(f"\nCross-Validation (5-fold on training data):")
            print(f"  ROC-AUC: {cv_results['mean_roc_auc']:.4f} +/- {cv_results['std_roc_auc']:.4f}")
            print(f"  F1:      {cv_results['mean_f1']:.4f}")
            print(f"\nValidation Set:")
            print(f"  ROC-AUC:   {valid_results['roc_auc']:.4f}")
            print(f"  Accuracy:  {valid_results['accuracy']:.4f}")
            print(f"  Precision: {valid_results['precision']:.4f}")
            print(f"  Recall:    {valid_results['recall']:.4f}")
            if test_results:
                print(f"\nTest Set (Final Score):")
                print(f"  ROC-AUC:   {test_results['roc_auc']:.4f}")
                print(f"  Accuracy:  {test_results['accuracy']:.4f}")
                print(f"  F1:        {test_results['f1']:.4f}")
                print(f"  Inference: {test_results['inference_ms']:.2f} ms/molecule")
            print(f"\n{'='*60}")
            print(f"FITNESS: {results['fitness']:.4f}")
            print(f"VALID: {results['valid']}")
            print(f"{'='*60}")

        return 0 if results['valid'] else 1

    except Exception as e:
        error_result = {
            'fitness': 0.0,
            'roc_auc': 0.0,
            'valid': False,
            'error': str(e)
        }
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
