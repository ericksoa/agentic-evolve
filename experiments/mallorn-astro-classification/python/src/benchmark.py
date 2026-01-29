#!/usr/bin/env python3
"""
Benchmark harness for MALLORN TDE classification.

This module provides the evaluation framework for comparing
baseline and evolved classifiers.

Usage:
    python benchmark.py                    # Run all baselines
    python benchmark.py --evolved          # Test evolved classifier
    python benchmark.py --quick            # Quick test with subset
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_batch, extract_baseline_features
from classifier import TDEClassifier, EvolvedTDEClassifier, optimize_threshold
from baselines import run_all_baselines, tde_heuristic_classifier
from data_loader import (
    load_single_split, load_combined_training_data,
    get_data_summary, prepare_labels as dl_prepare_labels
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(
    data_dir: str = '../data',
    split_num: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load MALLORN competition data from a single split.

    Args:
        data_dir: Path to data directory
        split_num: Which split to load (1-20)

    Returns:
        Tuple of (train_lc, test_lc, train_meta)
    """
    train_lc, test_lc, train_meta, _ = load_single_split(data_dir, split_num)
    return train_lc, test_lc, train_meta


def prepare_labels(metadata: pd.DataFrame, target_class: str = 'TDE') -> pd.Series:
    """
    Prepare binary labels for TDE classification.

    Args:
        metadata: Metadata DataFrame with 'target' column
        target_class: The positive class label (not used if target is already binary)

    Returns:
        Binary labels (1 for TDE, 0 otherwise)
    """
    # MALLORN data already has binary target column (1=TDE, 0=non-TDE)
    if metadata['target'].dtype in ['int64', 'int32', 'float64']:
        labels = metadata['target'].astype(int)
    else:
        # If target is string (e.g., 'TDE', 'AGN'), convert to binary
        labels = (metadata['target'] == target_class).astype(int)

    labels.index = metadata['object_id']
    return labels


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate classification predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Optional predicted probabilities

    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'n_true_positive': ((y_true == 1) & (y_pred == 1)).sum(),
        'n_false_positive': ((y_true == 0) & (y_pred == 1)).sum(),
        'n_false_negative': ((y_true == 1) & (y_pred == 0)).sum(),
        'n_true_negative': ((y_true == 0) & (y_pred == 0)).sum(),
    }

    # Add optimal threshold info if probabilities provided
    if y_proba is not None:
        opt_threshold, opt_f1 = optimize_threshold(y_true, y_proba)
        results['optimal_threshold'] = opt_threshold
        results['optimal_f1'] = opt_f1

    return results


def cross_validate(
    classifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Perform stratified cross-validation.

    Args:
        classifier: Classifier instance with fit/predict methods
        X: Feature DataFrame
        y: Target labels
        n_folds: Number of CV folds
        verbose: Print fold results

    Returns:
        Dictionary with mean and std of metrics
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit classifier
        classifier.fit(X_train, y_train)

        # Predict
        y_pred = classifier.predict(X_val)
        y_proba = classifier.predict_proba(X_val)[:, 1]

        # Evaluate
        fold_metrics = evaluate_predictions(y_val.values, y_pred, y_proba)
        fold_results.append(fold_metrics)

        if verbose:
            print(f"  Fold {fold_idx + 1}: F1={fold_metrics['f1']:.4f}, "
                  f"P={fold_metrics['precision']:.4f}, R={fold_metrics['recall']:.4f}")

    # Aggregate results
    results = {}
    for metric in ['f1', 'precision', 'recall', 'optimal_f1']:
        values = [r[metric] for r in fold_results if metric in r]
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)

    return results


# =============================================================================
# FITNESS FUNCTION (for /evolve)
# =============================================================================

def fitness(
    feature_extractor=None,
    classifier=None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    light_curves: Optional[pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> float:
    """
    Fitness function for evolutionary optimization.

    This function evaluates how well a feature extractor and/or classifier
    performs on TDE classification. Returns F1 score (higher is better).

    Args:
        feature_extractor: Feature extraction function (evolved)
        classifier: Classifier instance (evolved)
        X_train, y_train: Pre-extracted training features/labels
        X_val, y_val: Pre-extracted validation features/labels
        light_curves: Raw light curve data (if re-extracting features)
        metadata: Metadata with labels
        verbose: Print debug info

    Returns:
        F1 score (float between 0 and 1)
    """
    try:
        # Use pre-extracted features if provided
        if X_train is not None and y_train is not None:
            X_tr, y_tr = X_train, y_train
            X_v, y_v = X_val, y_val
        else:
            # Extract features from light curves
            raise NotImplementedError("Feature re-extraction not implemented")

        # Use default classifier if none provided
        if classifier is None:
            classifier = TDEClassifier()

        # Fit and predict
        classifier.fit(X_tr, y_tr)
        y_pred = classifier.predict(X_v)

        # Calculate F1 score
        score = f1_score(y_v, y_pred, zero_division=0)

        if verbose:
            print(f"Fitness: {score:.4f}")

        return score

    except Exception as e:
        if verbose:
            print(f"Fitness evaluation failed: {e}")
        return 0.0


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(
    data_dir: str = '../data',
    use_evolved: bool = False,
    quick: bool = False,
    cv_folds: int = 5
):
    """
    Run the full benchmark suite.

    Args:
        data_dir: Path to data directory
        use_evolved: Test evolved classifier
        quick: Use subset for quick testing
        cv_folds: Number of cross-validation folds
    """
    print("=" * 60)
    print("MALLORN TDE Classification Benchmark")
    print("=" * 60)

    # Check if data exists
    data_path = Path(data_dir)
    split_dir = data_path / 'split_01'
    if not split_dir.exists():
        print(f"\nData not found in {data_dir}")
        print("\nTo download the data:")
        print("1. Go to: https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge")
        print("2. Click 'Join Competition' and accept the rules")
        print("3. Then run:")
        print(f"   kaggle competitions download -c mallorn-astronomical-classification-challenge -p {data_dir}")
        print(f"   cd {data_dir} && unzip mallorn-astronomical-classification-challenge.zip")
        return

    # Load data
    print("\nLoading data (split 1)...")
    try:
        train_lc, test_lc, train_meta = load_data(data_dir, split_num=1)
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Make sure the data is properly extracted in the data/ directory")
        return

    if train_meta is None:
        print("\nWarning: No metadata found. Looking for embedded labels...")
        # Try to find labels in the light curve file
        print("Cannot proceed without metadata/labels")
        return

    print(f"  Training objects: {train_meta['object_id'].nunique()}")
    print(f"  Training observations: {len(train_lc)}")
    print(f"  Test objects: {test_lc['object_id'].nunique()}")

    # Prepare labels
    y = prepare_labels(train_meta)
    print(f"\nClass distribution:")
    print(f"  TDE: {(y == 1).sum()} ({100 * y.mean():.1f}%)")
    print(f"  Non-TDE: {(y == 0).sum()} ({100 * (1 - y.mean()):.1f}%)")

    # Quick mode: use subset
    if quick:
        print("\nQuick mode: using 20% of data")
        object_ids = train_meta['object_id'].unique()
        np.random.seed(42)
        subset_ids = np.random.choice(object_ids, size=len(object_ids) // 5, replace=False)
        train_lc = train_lc[train_lc['object_id'].isin(subset_ids)]
        train_meta = train_meta[train_meta['object_id'].isin(subset_ids)]
        y = y[y.index.isin(subset_ids)]

    # Extract features
    print("\nExtracting features...")
    X = extract_features_batch(
        train_lc,
        metadata=train_meta.set_index('object_id'),
        use_evolved=use_evolved,
        verbose=True
    )

    # Align X and y
    common_ids = X.index.intersection(y.index)
    X = X.loc[common_ids]
    y = y.loc[common_ids]

    print(f"  Features extracted: {X.shape[1]}")
    print(f"  Objects: {X.shape[0]}")

    # Run baselines
    print("\n" + "=" * 60)
    print("Running Baseline Algorithms")
    print("=" * 60)

    baseline_results = run_all_baselines(X, y, cv_folds=cv_folds, verbose=True)
    print("\nBaseline Summary:")
    print(baseline_results[['f1_mean', 'f1_std']].sort_values('f1_mean', ascending=False))

    # Test evolved classifier
    if use_evolved:
        print("\n" + "=" * 60)
        print("Testing Evolved Classifier")
        print("=" * 60)

        evolved = EvolvedTDEClassifier()
        results = cross_validate(evolved, X, y, n_folds=cv_folds, verbose=True)

        print(f"\nEvolved Results:")
        print(f"  F1: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
        print(f"  Optimal F1: {results['optimal_f1_mean']:.4f}")

    # Best result
    print("\n" + "=" * 60)
    print("Best Result")
    print("=" * 60)
    best_baseline = baseline_results['f1_mean'].idxmax()
    best_f1 = baseline_results.loc[best_baseline, 'f1_mean']
    print(f"  {best_baseline}: F1 = {best_f1:.4f}")

    return baseline_results


def main():
    parser = argparse.ArgumentParser(description='MALLORN TDE Classification Benchmark')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--evolved', action='store_true',
                        help='Test evolved classifier')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with subset')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        use_evolved=args.evolved,
        quick=args.quick,
        cv_folds=args.cv_folds
    )


if __name__ == '__main__':
    main()
