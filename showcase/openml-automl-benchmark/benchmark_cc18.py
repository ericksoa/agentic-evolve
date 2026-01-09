#!/usr/bin/env python3
"""
Benchmark on OpenML CC-18 Suite (72 datasets).

This validates the detection logic generalizes to a larger set of datasets.
Uses --limit to test on a subset first.
"""

import numpy as np
import pandas as pd
import warnings
import argparse
from datetime import datetime
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

# OpenML CC-18 suite ID
CC18_SUITE_ID = 99


def load_dataset(dataset_id):
    """Load and preprocess a dataset."""
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format='dataframe'
        )

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

        # Check if binary
        if len(np.unique(y)) != 2:
            return None, None, "Not binary"

        # Select numeric features only
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            return None, None, "No numeric features"

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

        return X, y, None
    except Exception as e:
        return None, None, str(e)


def compute_imbalance(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.min() if counts.min() > 0 else 1.0


def evaluate_classifier(clf_class, clf_kwargs, X, y, n_splits=5, n_seeds=2):
    """Evaluate a classifier with cross-validation."""
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


def run_benchmark(limit=None, save_results=True):
    """Run benchmark on CC-18 suite."""

    # Get CC-18 dataset IDs
    print("Fetching CC-18 suite...")
    suite = openml.study.get_suite(CC18_SUITE_ID)
    dataset_ids = suite.data

    if limit:
        dataset_ids = dataset_ids[:limit]
        print(f"Running on first {limit} datasets")
    else:
        print(f"Running on all {len(dataset_ids)} datasets")

    results = []
    skipped = []

    for i, dataset_id in enumerate(dataset_ids):
        print(f"\n[{i+1}/{len(dataset_ids)}] Dataset ID: {dataset_id}")

        X, y, error = load_dataset(dataset_id)

        if error:
            print(f"  Skipped: {error}")
            skipped.append({'dataset_id': dataset_id, 'reason': error})
            continue

        try:
            name = openml.datasets.get_dataset(dataset_id).name
            imbalance = compute_imbalance(y)

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

            # ThresholdOptimized
            v4_f1, _ = evaluate_classifier(
                ThresholdOptimizedClassifier,
                {'threshold_range': 'auto', 'skip_if_confident': True, 'random_state': 42},
                X, y
            )

            # Get diagnostics
            clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
            clf.fit(X, y)
            d = clf.diagnostics_

            gain = (v4_f1 - lr_f1) / lr_f1 * 100 if lr_f1 > 0 else 0

            print(f"  {name[:30]}: LR={lr_f1:.3f} v4={v4_f1:.3f} ({gain:+.1f}%) [{d['strategy']}]")

            results.append({
                'dataset_id': dataset_id,
                'dataset': name,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'imbalance': round(imbalance, 1),
                'overlap': round(d['overlap_pct'], 1),
                'strategy': d['strategy'],
                'LogReg': round(lr_f1, 3),
                'v4': round(v4_f1, 3),
                'gain_%': round(gain, 1),
            })

        except Exception as e:
            print(f"  Error: {e}")
            skipped.append({'dataset_id': dataset_id, 'reason': str(e)})

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS: CC-18 Suite")
    print("="*80)

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\nProcessed {len(df)} datasets, skipped {len(skipped)}")

        # Overall stats
        avg_gain = df['gain_%'].mean()
        improved = (df['gain_%'] > 0.5).sum()
        harmed = (df['gain_%'] < -0.5).sum()
        neutral = len(df) - improved - harmed

        print(f"\n--- Summary ---")
        print(f"Avg gain: {avg_gain:+.2f}%")
        print(f"Outcomes: {improved} improved, {neutral} neutral, {harmed} harmed")

        # By strategy
        print("\n--- By Strategy ---")
        for strategy in sorted(df['strategy'].unique()):
            subset = df[df['strategy'] == strategy]
            s_gain = subset['gain_%'].mean()
            s_improved = (subset['gain_%'] > 0.5).sum()
            s_harmed = (subset['gain_%'] < -0.5).sum()
            print(f"  {strategy}: {len(subset)} datasets, avg {s_gain:+.2f}%, {s_improved} improved, {s_harmed} harmed")

        # Top gainers
        print("\n--- Top 10 Gainers ---")
        top = df.nlargest(10, 'gain_%')[['dataset', 'strategy', 'LogReg', 'v4', 'gain_%']]
        print(top.to_string(index=False))

        # Worst performers
        print("\n--- Worst Performers ---")
        worst = df.nsmallest(5, 'gain_%')[['dataset', 'strategy', 'LogReg', 'v4', 'gain_%']]
        print(worst.to_string(index=False))

        # Save results
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"cc18_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark on CC-18 suite')
    parser.add_argument('--limit', type=int, default=None, help='Limit to first N datasets')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to CSV')
    args = parser.parse_args()

    run_benchmark(limit=args.limit, save_results=not args.no_save)
