#!/usr/bin/env python3
"""
Evaluation harness for OpenML datasets.

This is the fitness function for evolve-sdk. It evaluates a classifier
on a specific OpenML dataset using proper train/valid/test splits.

Usage:
    python evaluate_dataset.py <solution_file> --dataset-id 31 --json
"""

import argparse
import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# Suppress warnings for clean JSON output
warnings.filterwarnings('ignore')


def load_solution(solution_path: str) -> Any:
    """
    Load and execute a solution file to get the classifier.

    The solution file should define a `get_classifier()` function
    that returns an sklearn-compatible classifier.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'get_classifier'):
        return module.get_classifier()
    elif hasattr(module, 'Classifier'):
        return module.Classifier()
    else:
        raise ValueError(f"Solution must define get_classifier() or Classifier class")


def preprocess_data(X: pd.DataFrame) -> np.ndarray:
    """Standard preprocessing for all datasets."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Separate numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    processed = []

    # Numeric features
    if len(numeric_cols) > 0:
        X_num = X[numeric_cols].values
        imputer = SimpleImputer(strategy='median')
        X_num = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        processed.append(X_num)

    # Categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        vals = X[col].fillna('__MISSING__').astype(str)
        encoded = le.fit_transform(vals).reshape(-1, 1)
        processed.append(encoded)

    if processed:
        return np.hstack(processed)
    return np.array([]).reshape(len(X), 0)


def evaluate_on_dataset(
    classifier,
    dataset_id: int,
    n_cv_folds: int = 8,
    n_holdout_folds: int = 2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a classifier on an OpenML dataset with proper validation.

    Uses train/CV/holdout split strategy from MALLORN lessons:
    - CV folds (default 8): Used for model selection
    - Holdout folds (default 2): Used for overfitting detection

    Returns comprehensive metrics including overfitting indicators.
    """
    import openml

    # Load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    if y.dtype == 'object' or y.dtype.name == 'category':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name='target')

    # Preprocess features
    X_processed = preprocess_data(X)
    y_values = y.values

    # Dataset info
    n_classes = len(np.unique(y_values))
    class_counts = pd.Series(y_values).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    # Create CV and holdout splits
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=random_state)
    all_splits = list(skf.split(X_processed, y_values))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    # Evaluate on CV folds
    cv_scores = []
    cv_train_scores = []

    for train_idx, test_idx in cv_splits:
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        try:
            clf = classifier.__class__(**classifier.get_params())
            clf.fit(X_train, y_train)

            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)

            average = 'binary' if n_classes == 2 else 'macro'
            cv_scores.append(f1_score(y_test, y_pred_test, average=average, zero_division=0))
            cv_train_scores.append(f1_score(y_train, y_pred_train, average=average, zero_division=0))
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'fitness': 0.0
            }

    # Evaluate on holdout folds (overfitting detection)
    holdout_scores = []

    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        try:
            clf = classifier.__class__(**classifier.get_params())
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            average = 'binary' if n_classes == 2 else 'macro'
            holdout_scores.append(f1_score(y_test, y_pred, average=average, zero_division=0))
        except Exception:
            holdout_scores.append(0.0)

    # Calculate metrics
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_train_mean = np.mean(cv_train_scores)
    holdout_mean = np.mean(holdout_scores)

    # Overfitting indicators
    train_cv_gap = cv_train_mean - cv_mean
    cv_holdout_gap = cv_mean - holdout_mean

    # Determine if overfitting
    # MALLORN lesson: cv_holdout_gap > 0.10 is severe overfitting
    is_overfit = cv_holdout_gap > 0.10

    # Fitness is holdout score (generalizes better than CV)
    # But penalize if clear overfitting
    fitness = holdout_mean
    if is_overfit:
        # Penalize overfitting
        fitness = holdout_mean * 0.9

    return {
        'valid': True,
        'fitness': fitness,
        'cv_f1_mean': cv_mean,
        'cv_f1_std': cv_std,
        'holdout_f1_mean': holdout_mean,
        'train_f1_mean': cv_train_mean,
        'train_cv_gap': train_cv_gap,
        'cv_holdout_gap': cv_holdout_gap,
        'is_overfit': is_overfit,
        'dataset_id': dataset_id,
        'dataset_name': dataset.name,
        'n_samples': len(X),
        'n_features': X_processed.shape[1],
        'n_classes': n_classes,
        'imbalance_ratio': imbalance_ratio,
        'cv_scores': cv_scores,
        'holdout_scores': holdout_scores
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a classifier on an OpenML dataset'
    )
    parser.add_argument('solution', help='Path to solution file')
    parser.add_argument('--dataset-id', type=int, required=True,
                        help='OpenML dataset ID')
    parser.add_argument('--cv-folds', type=int, default=8,
                        help='Number of CV folds (default: 8)')
    parser.add_argument('--holdout-folds', type=int, default=2,
                        help='Number of holdout folds (default: 2)')
    parser.add_argument('--json', action='store_true',
                        help='Output JSON format')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    try:
        # Load solution
        classifier = load_solution(args.solution)

        # Evaluate
        result = evaluate_on_dataset(
            classifier,
            dataset_id=args.dataset_id,
            n_cv_folds=args.cv_folds,
            n_holdout_folds=args.holdout_folds,
            random_state=args.seed
        )

        if args.json:
            # Remove numpy arrays for JSON serialization
            result_json = {k: v for k, v in result.items()
                         if not isinstance(v, (np.ndarray, list)) or k in ['cv_scores', 'holdout_scores']}
            result_json['cv_scores'] = [float(x) for x in result.get('cv_scores', [])]
            result_json['holdout_scores'] = [float(x) for x in result.get('holdout_scores', [])]
            print(json.dumps(result_json, indent=2))
        else:
            if result['valid']:
                print(f"Dataset: {result['dataset_name']} (ID={result['dataset_id']})")
                print(f"Samples: {result['n_samples']}, Features: {result['n_features']}, "
                      f"Classes: {result['n_classes']}")
                print(f"\nCV F1:      {result['cv_f1_mean']:.4f} +/- {result['cv_f1_std']:.4f}")
                print(f"Holdout F1: {result['holdout_f1_mean']:.4f}")
                print(f"Train F1:   {result['train_f1_mean']:.4f}")
                print(f"\nTrain-CV Gap:    {result['train_cv_gap']:.4f}")
                print(f"CV-Holdout Gap:  {result['cv_holdout_gap']:.4f}")
                print(f"Overfit:         {'YES' if result['is_overfit'] else 'NO'}")
                print(f"\nFitness: {result['fitness']:.4f}")
            else:
                print(f"FAILED: {result['error']}")

        sys.exit(0 if result['valid'] else 1)

    except Exception as e:
        if args.json:
            print(json.dumps({'valid': False, 'error': str(e), 'fitness': 0.0}))
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
