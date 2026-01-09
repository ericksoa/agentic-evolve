#!/usr/bin/env python3
"""
Benchmark v4: ThresholdOptimizedClassifier with bootstrap confidence intervals.

Adds:
- 95% bootstrap confidence intervals for F1 scores
- Statistical significance testing vs baseline
- Supports --quick flag for faster runs
"""

import numpy as np
import pandas as pd
import warnings
import argparse
warnings.filterwarnings('ignore')

import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy import stats

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


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval for the mean of scores.

    Returns (mean, lower_bound, upper_bound).
    """
    scores = np.array(scores)
    n = len(scores)

    # Bootstrap resampling
    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return np.mean(scores), lower, upper


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=1000):
    """
    Paired bootstrap test to determine if difference is significant.

    Returns p-value (probability that B is not better than A).
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Observed difference
    obs_diff = np.mean(scores_b) - np.mean(scores_a)

    # Bootstrap null distribution
    rng = np.random.default_rng(42)
    n = len(scores_a)
    pooled = np.concatenate([scores_a, scores_b])

    null_diffs = []
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(pooled)
        null_a = shuffled[:n]
        null_b = shuffled[n:]
        null_diffs.append(np.mean(null_b) - np.mean(null_a))

    # P-value: proportion of null diffs >= observed diff
    p_value = np.mean(np.array(null_diffs) >= obs_diff)

    return p_value


def evaluate_with_ci(clf_class, clf_kwargs, X, y, n_splits=5, n_seeds=3, n_bootstrap=1000):
    """
    Evaluate classifier and return F1 scores for each fold/seed.

    Returns list of scores (can be used for bootstrap CI and paired tests).
    """
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
    return f1_scores


def run_benchmark(quick=False, n_bootstrap=1000):
    """Run benchmark with bootstrap CIs."""
    n_seeds = 2 if quick else 3
    results = []

    for dataset_id, name in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print('='*60)

        try:
            X, y = load_dataset(dataset_id)
            imbalance = compute_imbalance(y)

            # Baseline LogReg - collect all fold scores
            lr_scores = []
            for seed in range(n_seeds):
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

            # v4: Auto with sensitivity check
            v4_scores = evaluate_with_ci(
                ThresholdOptimizedClassifier,
                {'threshold_range': 'auto', 'skip_if_confident': True, 'random_state': 42},
                X, y, n_seeds=n_seeds, n_bootstrap=n_bootstrap
            )

            # Bootstrap CIs
            lr_mean, lr_lo, lr_hi = bootstrap_ci(lr_scores, n_bootstrap)
            v4_mean, v4_lo, v4_hi = bootstrap_ci(v4_scores, n_bootstrap)

            # Paired test for significance
            p_value = paired_bootstrap_test(lr_scores, v4_scores, n_bootstrap)
            is_significant = p_value < 0.05

            # Get diagnostics from a single fit
            clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
            clf.fit(X, y)
            d = clf.diagnostics_

            gain = (v4_mean - lr_mean) / lr_mean * 100
            gain_lo = (v4_lo - lr_mean) / lr_mean * 100
            gain_hi = (v4_hi - lr_mean) / lr_mean * 100

            sig_marker = "*" if is_significant else ""
            print(f"  LogReg: {lr_mean:.3f} [{lr_lo:.3f}, {lr_hi:.3f}]")
            print(f"  v4:     {v4_mean:.3f} [{v4_lo:.3f}, {v4_hi:.3f}] ({gain:+.1f}%){sig_marker}")
            print(f"  Strategy: {d['strategy']}, p-value: {p_value:.3f}")

            results.append({
                'dataset': name,
                'imbalance': round(imbalance, 1),
                'overlap': round(d['overlap_pct'], 1),
                'strategy': d['strategy'],
                'LogReg': round(lr_mean, 3),
                'LogReg_CI': f"[{lr_lo:.3f}, {lr_hi:.3f}]",
                'v4': round(v4_mean, 3),
                'v4_CI': f"[{v4_lo:.3f}, {v4_hi:.3f}]",
                'gain_%': round(gain, 1),
                'p_value': round(p_value, 3),
                'significant': is_significant,
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS: v4 with Bootstrap Confidence Intervals")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[['dataset', 'strategy', 'LogReg', 'v4', 'gain_%', 'p_value', 'significant']].to_string(index=False))

    print(f"\n--- Summary ---")
    print(f"Avg v4 gain: {df['gain_%'].mean():+.2f}%")

    # Count significant improvements
    sig_improved = ((df['gain_%'] > 0) & df['significant']).sum()
    sig_harmed = ((df['gain_%'] < 0) & df['significant']).sum()
    print(f"Significant improvements: {sig_improved}")
    print(f"Significant harm: {sig_harmed}")

    # By strategy
    print("\n--- By Strategy ---")
    for strategy in sorted(df['strategy'].unique()):
        subset = df[df['strategy'] == strategy]
        avg_gain = subset['gain_%'].mean()
        sig_count = subset['significant'].sum()
        print(f"  {strategy}: {len(subset)} datasets, avg gain {avg_gain:+.2f}%, {sig_count} significant")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark v4 with bootstrap CIs')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer seeds/bootstraps)')
    args = parser.parse_args()

    n_bootstrap = 100 if args.quick else 1000
    run_benchmark(quick=args.quick, n_bootstrap=n_bootstrap)
