#!/usr/bin/env python3
"""
Benchmark the improved v2 ThresholdOptimizedClassifier.
Compare against v1 results to measure improvement.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '.')
from adaptive_ensemble import ThresholdOptimizedClassifier

# Same datasets as before
DATASETS = {
    31: "credit-g",
    1046: "mozilla4",
    37: "diabetes",
    1464: "blood-transfusion",
    1480: "ilpd",
    15: "breast-w",
    1494: "qsar-biodeg",
    1489: "phoneme",
    1462: "banknote-auth",
    44: "spambase",
    1068: "pc1",
    1067: "kc2",
}


def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = X.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    return X, y


def compute_imbalance(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.min() if counts.min() > 0 else 1.0


def evaluate_classifier(clf_class, clf_kwargs, X, y, n_splits=5, n_seeds=2):
    """Evaluate classifier with stratified CV."""
    f1_scores = []

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + seed)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = clf_class(**clf_kwargs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(f1_scores), np.std(f1_scores)


def run_benchmark():
    results = []

    for dataset_id, name in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print('='*60)

        try:
            X, y = load_dataset(dataset_id)
            imbalance = compute_imbalance(y)
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Imbalance: {imbalance:.1f}x")

            # Baseline LogReg
            lr_scores = []
            for seed in range(2):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + seed)
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
                    lr.fit(X_train_s, y_train)
                    y_pred = lr.predict(X_test_s)
                    lr_scores.append(f1_score(y_test, y_pred, zero_division=0))
            lr_f1 = np.mean(lr_scores)

            # v1: Fixed threshold range (simulating old behavior)
            v1_f1, _ = evaluate_classifier(
                ThresholdOptimizedClassifier,
                {'threshold_range': (0.20, 0.55), 'skip_if_confident': False, 'random_state': 42},
                X, y
            )

            # v2: Auto threshold range with smart skipping
            v2_f1, _ = evaluate_classifier(
                ThresholdOptimizedClassifier,
                {'threshold_range': 'auto', 'skip_if_confident': True, 'random_state': 42},
                X, y
            )

            # v2 with calibration
            v2_cal_f1, _ = evaluate_classifier(
                ThresholdOptimizedClassifier,
                {'threshold_range': 'auto', 'calibrate': True, 'random_state': 42},
                X, y
            )

            # Get diagnostics from one fit
            clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
            clf.fit(X, y)
            strategy = clf.diagnostics_['strategy']
            overlap = clf.overlap_pct_
            opt_thresh = clf.optimal_threshold_

            print(f"  LogReg:     {lr_f1:.3f}")
            print(f"  v1 (fixed): {v1_f1:.3f} ({(v1_f1-lr_f1)/lr_f1*100:+.1f}%)")
            print(f"  v2 (auto):  {v2_f1:.3f} ({(v2_f1-lr_f1)/lr_f1*100:+.1f}%)")
            print(f"  v2+calib:   {v2_cal_f1:.3f} ({(v2_cal_f1-lr_f1)/lr_f1*100:+.1f}%)")
            print(f"  Strategy: {strategy}, Overlap: {overlap:.1f}%, Threshold: {opt_thresh:.2f}")

            results.append({
                'dataset': name,
                'imbalance': round(imbalance, 1),
                'overlap': round(overlap, 1),
                'strategy': strategy,
                'LogReg': round(lr_f1, 3),
                'v1_fixed': round(v1_f1, 3),
                'v2_auto': round(v2_f1, 3),
                'v2_calib': round(v2_cal_f1, 3),
                'v1_gain': round((v1_f1-lr_f1)/lr_f1*100, 1),
                'v2_gain': round((v2_f1-lr_f1)/lr_f1*100, 1),
                'v2_vs_v1': round((v2_f1-v1_f1)/v1_f1*100, 1),
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS: v1 vs v2")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[['dataset', 'overlap', 'strategy', 'LogReg', 'v1_fixed', 'v2_auto', 'v1_gain', 'v2_gain']].to_string(index=False))

    print(f"\n--- Summary ---")
    print(f"Avg v1 gain: {df['v1_gain'].mean():+.2f}%")
    print(f"Avg v2 gain: {df['v2_gain'].mean():+.2f}%")
    print(f"v2 improvement over v1: {df['v2_vs_v1'].mean():+.2f}%")

    # By strategy
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy]
        print(f"\nStrategy '{strategy}' ({len(subset)} datasets):")
        print(f"  Avg v1 gain: {subset['v1_gain'].mean():+.2f}%")
        print(f"  Avg v2 gain: {subset['v2_gain'].mean():+.2f}%")

    return df


if __name__ == '__main__':
    run_benchmark()
