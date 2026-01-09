#!/usr/bin/env python3
"""
Benchmark AdaptiveEnsembleClassifier on multiple OpenML datasets.

Usage:
    python -m adaptive_ensemble.benchmark
    python -m adaptive_ensemble.benchmark --datasets 37 1461 40994
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .classifier import AdaptiveEnsembleClassifier
from .analysis import DatasetAnalyzer


# Default test datasets from OpenML
DEFAULT_DATASETS = {
    37: "diabetes",      # 768 samples, 8 features, binary
    1461: "bank-marketing",  # 4521 samples, 16 features, binary
    1464: "blood-transfusion",  # 748 samples, 4 features, binary
    1480: "ilpd",        # 583 samples, 10 features, binary
    1494: "qsar-biodeg", # 1055 samples, 41 features, binary
}


def load_dataset(dataset_id: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load dataset from OpenML."""
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    return X, y, dataset.name


def evaluate_classifier(clf, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_seeds: int = 3) -> Dict:
    """Evaluate classifier with stratified CV over multiple seeds."""
    f1_scores = []
    acc_scores = []

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + seed)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                clf_clone = clf.__class__(**clf.get_params())
                clf_clone.fit(X_train, y_train)
                y_pred = clf_clone.predict(X_test)

                f1_scores.append(f1_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'macro', zero_division=0))
                acc_scores.append(accuracy_score(y_test, y_pred))
            except Exception as e:
                print(f"  Warning: {e}")
                f1_scores.append(0)
                acc_scores.append(0)

    return {
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'acc_mean': np.mean(acc_scores),
        'acc_std': np.std(acc_scores),
    }


def run_benchmark(dataset_ids: List[int] = None, verbose: bool = True):
    """Run benchmark on specified datasets."""
    if dataset_ids is None:
        dataset_ids = list(DEFAULT_DATASETS.keys())

    results = []

    for dataset_id in dataset_ids:
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_id}: {DEFAULT_DATASETS.get(dataset_id, 'unknown')}")
        print('='*60)

        try:
            X, y, name = load_dataset(dataset_id)
            print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

            # Analyze dataset
            analyzer = DatasetAnalyzer()
            profile = analyzer.analyze(X, y)
            print(f"Imbalance ratio: {profile.imbalance_ratio:.2f}")
            print(f"Recommended threshold: {profile.recommended_threshold}")

            # Baseline: LogisticRegression
            print("\nEvaluating LogisticRegression baseline...")
            lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
            lr_results = evaluate_classifier(lr, X, y)
            print(f"  F1: {lr_results['f1_mean']:.4f} (+/- {lr_results['f1_std']:.4f})")

            # Baseline: RandomForest
            print("Evaluating RandomForest baseline...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
            rf_results = evaluate_classifier(rf, X, y)
            print(f"  F1: {rf_results['f1_mean']:.4f} (+/- {rf_results['f1_std']:.4f})")

            # AdaptiveEnsemble
            print("Evaluating AdaptiveEnsembleClassifier...")
            ae = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
            ae_results = evaluate_classifier(ae, X, y)
            print(f"  F1: {ae_results['f1_mean']:.4f} (+/- {ae_results['f1_std']:.4f})")

            # Improvement
            improvement = (ae_results['f1_mean'] - lr_results['f1_mean']) / lr_results['f1_mean'] * 100 if lr_results['f1_mean'] > 0 else 0
            print(f"\nImprovement over LogReg: {improvement:+.1f}%")

            results.append({
                'dataset_id': dataset_id,
                'name': name,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'imbalance': profile.imbalance_ratio,
                'lr_f1': lr_results['f1_mean'],
                'rf_f1': rf_results['f1_mean'],
                'ae_f1': ae_results['f1_mean'],
                'improvement': improvement,
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print(f"\nOverall:")
        print(f"  Datasets tested: {len(results)}")
        print(f"  Average improvement over LogReg: {df['improvement'].mean():+.1f}%")
        print(f"  Datasets where AE beats LogReg: {(df['ae_f1'] > df['lr_f1']).sum()}/{len(results)}")
        print(f"  Datasets where AE beats RF: {(df['ae_f1'] > df['rf_f1']).sum()}/{len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark AdaptiveEnsembleClassifier")
    parser.add_argument('--datasets', nargs='+', type=int, default=None,
                        help='OpenML dataset IDs to test')
    parser.add_argument('--quiet', action='store_true', help='Less output')

    args = parser.parse_args()

    run_benchmark(args.datasets, verbose=not args.quiet)


if __name__ == '__main__':
    main()
