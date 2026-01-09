#!/usr/bin/env python3
"""
Fitness evaluation script for evolve-sdk.

This script evaluates a candidate solution for the diabetes classification task.
It uses 8-fold CV + 2-fold holdout validation, and rejects solutions that overfit.

Usage:
    python evolve_fitness.py <solution_path> [--json]

The solution file should define a `create_pipeline()` function that returns
a sklearn Pipeline or classifier.
"""

import argparse
import json
import sys
import importlib.util
import numpy as np
import pandas as pd
import openml
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

warnings.filterwarnings('ignore')


def load_solution(solution_path: str):
    """Load a solution module and get its create_pipeline function."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'create_pipeline'):
        raise ValueError(f"Solution must define create_pipeline() function")

    return module.create_pipeline


def load_diabetes():
    """Load diabetes dataset from OpenML."""
    dataset = openml.datasets.get_dataset(37)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    return X, y.values


def evaluate_solution(create_pipeline, X, y, n_cv_folds=8, n_holdout_folds=2, n_seeds=3):
    """
    Evaluate a solution with CV + holdout validation.

    Returns:
        dict with cv_f1, holdout_f1, gap, and acceptance status
    """
    total_folds = n_cv_folds + n_holdout_folds

    all_cv_scores = []
    all_holdout_scores = []

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42 + seed)
        all_splits = list(skf.split(X, y))

        cv_splits = all_splits[:n_cv_folds]
        holdout_splits = all_splits[n_cv_folds:]

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # CV evaluation
        cv_scores = []
        for train_idx, test_idx in cv_splits:
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                pipeline = create_pipeline()
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                cv_scores.append(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                cv_scores.append(0.0)

        all_cv_scores.extend(cv_scores)

        # Holdout evaluation
        holdout_scores = []
        for train_idx, test_idx in holdout_splits:
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                pipeline = create_pipeline()
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                holdout_scores.append(0.0)

        all_holdout_scores.extend(holdout_scores)

    cv_mean = np.mean(all_cv_scores)
    cv_std = np.std(all_cv_scores)
    holdout_mean = np.mean(all_holdout_scores)
    holdout_std = np.std(all_holdout_scores)
    gap = cv_mean - holdout_mean

    # Acceptance criteria
    accepted = True
    rejection_reason = None

    if gap > 0.08:
        accepted = False
        rejection_reason = f"Overfitting detected: CV-holdout gap {gap:.3f} > 0.08"

    return {
        "cv_f1": float(cv_mean),
        "cv_std": float(cv_std),
        "holdout_f1": float(holdout_mean),
        "holdout_std": float(holdout_std),
        "gap": float(gap),
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "fitness": float(holdout_mean) if accepted else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a diabetes classification solution")
    parser.add_argument("solution", help="Path to solution Python file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        # Load solution
        create_pipeline = load_solution(args.solution)

        # Load data
        X, y = load_diabetes()

        # Evaluate
        result = evaluate_solution(create_pipeline, X, y)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"CV F1:      {result['cv_f1']:.4f} (+/- {result['cv_std']:.4f})")
            print(f"Holdout F1: {result['holdout_f1']:.4f} (+/- {result['holdout_std']:.4f})")
            print(f"Gap:        {result['gap']:.4f}")
            print(f"Accepted:   {result['accepted']}")
            if result['rejection_reason']:
                print(f"Reason:     {result['rejection_reason']}")
            print(f"Fitness:    {result['fitness']:.4f}")

        # Exit with error if rejected
        sys.exit(0 if result['accepted'] else 1)

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "fitness": 0.0}))
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
