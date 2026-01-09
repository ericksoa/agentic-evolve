#!/usr/bin/env python3
"""
Benchmark v3: ThresholdOptimizedClassifier with sensitivity check.
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

            # v3: Auto with sensitivity check
            v3_f1, _ = evaluate_classifier(
                ThresholdOptimizedClassifier,
                {'threshold_range': 'auto', 'skip_if_confident': True, 'random_state': 42},
                X, y
            )

            # Get diagnostics
            clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
            clf.fit(X, y)
            d = clf.diagnostics_

            gain = (v3_f1 - lr_f1) / lr_f1 * 100

            print(f"  LogReg: {lr_f1:.3f}")
            print(f"  v3:     {v3_f1:.3f} ({gain:+.1f}%)")
            print(f"  Strategy: {d['strategy']}")
            print(f"  Overlap: {d['overlap_pct']:.1f}%, F1 range: {d['f1_range']:.3f}")
            print(f"  Potential gain: {d['potential_gain']*100:+.1f}%, Threshold dist: {d['threshold_distance_from_05']:.2f}")
            print(f"  Optimal threshold: {clf.optimal_threshold_:.2f}")

            results.append({
                'dataset': name,
                'imbalance': round(imbalance, 1),
                'overlap': round(d['overlap_pct'], 1),
                'f1_range': round(d['f1_range'], 3),
                'potential_gain': round(d['potential_gain']*100, 1),
                'thresh_dist': round(d['threshold_distance_from_05'], 2),
                'strategy': d['strategy'],
                'LogReg': round(lr_f1, 3),
                'v3': round(v3_f1, 3),
                'actual_gain': round(gain, 1),
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS: v3 with Sensitivity Check")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[['dataset', 'strategy', 'overlap', 'f1_range', 'potential_gain', 'LogReg', 'v3', 'actual_gain']].to_string(index=False))

    print(f"\n--- Summary ---")
    print(f"Avg v3 gain: {df['actual_gain'].mean():+.2f}%")

    # By strategy
    for strategy in sorted(df['strategy'].unique()):
        subset = df[df['strategy'] == strategy]
        avg_gain = subset['actual_gain'].mean()
        print(f"  {strategy}: {len(subset)} datasets, avg gain {avg_gain:+.2f}%")

    # Count improvements vs harm
    improved = (df['actual_gain'] > 0.5).sum()
    harmed = (df['actual_gain'] < -0.5).sum()
    neutral = len(df) - improved - harmed
    print(f"\nOutcomes: {improved} improved, {neutral} neutral, {harmed} harmed")

    return df


if __name__ == '__main__':
    run_benchmark()
