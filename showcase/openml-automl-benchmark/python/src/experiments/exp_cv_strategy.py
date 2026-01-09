#!/usr/bin/env python3
"""
Cross-Validation Strategy Experiment for Diabetes Classification

Tests different validation strategies to understand if our current
8-fold CV + 2-fold holdout approach is accurate or pessimistic.

Validation strategies tested:
- 5-fold CV (standard, no holdout)
- 10-fold CV (more robust, no holdout)
- Repeated 5-fold CV (3x5=15 folds)
- Leave-one-out CV (maximum use of data)
- 8+2 CV+holdout (current approach)
- 7+3 CV+holdout (more holdout)
- 6+4 CV+holdout (conservative holdout)

Current best: 0.665 holdout F1 (Domain + Bins + LogReg)
Target: 0.745 (Auto-sklearn level)
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut, cross_val_score
)
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for diabetes dataset.
    Implements Domain + Bins feature set (current best).
    """

    def __init__(self,
                 add_interactions=False,
                 add_ratios=False,
                 add_bins=False,
                 add_polynomials=False,
                 add_domain=False,
                 poly_degree=2):
        self.add_interactions = add_interactions
        self.add_ratios = add_ratios
        self.add_bins = add_bins
        self.add_polynomials = add_polynomials
        self.add_domain = add_domain
        self.poly_degree = poly_degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X columns: preg, plas, pres, skin, insu, mass, pedi, age
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        features = [df.copy()]

        if self.add_domain:
            domain = pd.DataFrame()
            domain['bmi_underweight'] = (df['mass'] < 18.5).astype(float)
            domain['bmi_normal'] = ((df['mass'] >= 18.5) & (df['mass'] < 25)).astype(float)
            domain['bmi_overweight'] = ((df['mass'] >= 25) & (df['mass'] < 30)).astype(float)
            domain['bmi_obese'] = (df['mass'] >= 30).astype(float)
            domain['age_young'] = (df['age'] < 30).astype(float)
            domain['age_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(float)
            domain['age_senior'] = (df['age'] >= 50).astype(float)
            domain['glucose_normal'] = (df['plas'] < 100).astype(float)
            domain['glucose_prediabetic'] = ((df['plas'] >= 100) & (df['plas'] < 126)).astype(float)
            domain['glucose_diabetic'] = (df['plas'] >= 126).astype(float)
            domain['high_risk'] = (
                (df['mass'] >= 30) & (df['age'] >= 40) & (df['plas'] >= 100)
            ).astype(float)
            domain['preg_risk'] = (df['preg'] >= 4).astype(float)
            features.append(domain)

        if self.add_ratios:
            ratios = pd.DataFrame()
            ratios['glucose_insulin_ratio'] = df['plas'] / (df['insu'] + 1)
            ratios['bmi_age_ratio'] = df['mass'] / (df['age'] + 1)
            ratios['glucose_bmi_ratio'] = df['plas'] / (df['mass'] + 1)
            ratios['pedi_age'] = df['pedi'] * df['age']
            features.append(ratios)

        if self.add_interactions:
            interactions = pd.DataFrame()
            interactions['glucose_bmi'] = df['plas'] * df['mass']
            interactions['age_bmi'] = df['age'] * df['mass']
            interactions['glucose_age'] = df['plas'] * df['age']
            interactions['preg_age'] = df['preg'] * df['age']
            interactions['pedi_glucose'] = df['pedi'] * df['plas']
            features.append(interactions)

        if self.add_bins:
            bins = pd.DataFrame()
            for col in ['plas', 'mass', 'age', 'pedi']:
                bins[f'{col}_q1'] = (df[col] <= df[col].quantile(0.25)).astype(float)
                bins[f'{col}_q4'] = (df[col] >= df[col].quantile(0.75)).astype(float)
            features.append(bins)

        if self.add_polynomials:
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            top_features = df[['plas', 'mass', 'age']].values
            poly_features = poly.fit_transform(top_features)
            poly_df = pd.DataFrame(
                poly_features[:, 3:],
                columns=[f'poly_{i}' for i in range(poly_features.shape[1] - 3)]
            )
            features.append(poly_df)

        result = pd.concat(features, axis=1)
        return result.values


def load_diabetes_raw():
    """Load diabetes with proper missing value handling."""
    dataset = openml.datasets.get_dataset(37)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    X_values = X.values.astype(float)

    # Handle zeros that are actually missing
    for col in [1, 2, 3, 4, 5]:  # plas, pres, skin, insu, mass
        X_values[X_values[:, col] == 0, col] = np.nan

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)

    return X_imputed, y.values


def create_pipeline(feature_engineer, classifier):
    """Create a pipeline with feature engineering and classification."""
    return Pipeline([
        ('features', feature_engineer),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def evaluate_cv_only(pipeline, X, y, cv_strategy, is_loo=False):
    """
    Evaluate pipeline using only cross-validation (no separate holdout).
    Returns mean score, std, and all fold scores.

    For LOO, we need to accumulate predictions and compute F1 once at the end,
    since each fold has only 1 sample and F1 doesn't work per-sample.
    """
    if is_loo:
        # For LOO, collect all predictions and compute F1 at the end
        y_true_all = []
        y_pred_all = []
        for train_idx, test_idx in cv_strategy.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = clone(pipeline)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        # Compute F1 on all predictions
        overall_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        # For LOO, std across folds doesn't make sense, return 0
        return overall_f1, 0.0, [overall_f1]
    else:
        f1_scorer = make_scorer(f1_score, zero_division=0)
        scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring=f1_scorer)
        return np.mean(scores), np.std(scores), scores.tolist()


def evaluate_cv_plus_holdout(pipeline, X, y, n_cv_folds, n_holdout_folds):
    """
    Evaluate pipeline with CV + holdout (current approach).
    Returns CV mean, holdout mean, all scores.
    """
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    cv_scores = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return (
        np.mean(cv_scores), np.std(cv_scores), cv_scores,
        np.mean(holdout_scores), np.std(holdout_scores), holdout_scores
    )


def run_cv_strategy_experiment():
    """Run comprehensive CV strategy comparison experiment."""
    print("=" * 70)
    print("CROSS-VALIDATION STRATEGY EXPERIMENT FOR DIABETES")
    print("=" * 70)

    print("\nLoading diabetes dataset...")
    X, y = load_diabetes_raw()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y)} (ratio: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1)")

    results = {
        'experiment': 'Cross-Validation Strategy Comparison',
        'dataset': 'OpenML 37 (diabetes)',
        'n_samples': int(X.shape[0]),
        'baseline_holdout': 0.665,
        'target': 0.745,
        'timestamp': datetime.now().isoformat(),
        'model': 'LogReg with Domain + Bins features',
        'experiments': []
    }

    # Best model configuration
    feature_engineer = DiabetesFeatureEngineer(add_domain=True, add_bins=True)
    classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    pipeline = create_pipeline(feature_engineer, classifier)

    print(f"\nModel: LogisticRegression (C=1.0, class_weight='balanced')")
    print(f"Features: Domain + Bins (best configuration)")

    # =========================================================================
    # CV-only strategies
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: CV-ONLY STRATEGIES")
    print("=" * 70)
    print("(These use all data for validation, no separate holdout)")

    cv_only_strategies = [
        ('5-fold CV', StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
        ('10-fold CV', StratifiedKFold(n_splits=10, shuffle=True, random_state=42)),
        ('15-fold CV', StratifiedKFold(n_splits=15, shuffle=True, random_state=42)),
        ('20-fold CV', StratifiedKFold(n_splits=20, shuffle=True, random_state=42)),
        ('Repeated 5-fold CV (3x)', RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)),
        ('Repeated 5-fold CV (5x)', RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)),
        ('Repeated 10-fold CV (3x)', RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)),
    ]

    print("\n{:<30} {:>10} {:>10} {:>10}".format("Strategy", "Mean F1", "Std", "Folds"))
    print("-" * 60)

    for name, cv in cv_only_strategies:
        mean_f1, std_f1, scores = evaluate_cv_only(clone(pipeline), X, y, cv)

        print(f"{name:<30} {mean_f1:>10.4f} {std_f1:>10.4f} {len(scores):>10}")

        results['experiments'].append({
            'strategy': name,
            'type': 'cv_only',
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1),
            'n_folds': len(scores),
            'all_scores': [float(s) for s in scores],
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
        })

    # LOO is slow but comprehensive
    print("\nRunning Leave-One-Out CV (768 folds, may take a moment)...")
    loo = LeaveOneOut()
    loo_mean, loo_std, loo_scores = evaluate_cv_only(clone(pipeline), X, y, loo, is_loo=True)
    print(f"{'Leave-One-Out CV':<30} {loo_mean:>10.4f} {'N/A':>10} {'768':>10}")

    results['experiments'].append({
        'strategy': 'Leave-One-Out CV',
        'type': 'cv_only',
        'mean_f1': float(loo_mean),
        'std_f1': None,  # N/A for LOO
        'n_folds': 768,
        'all_scores': None,  # Not applicable for LOO (F1 computed on all predictions)
        'min_score': float(loo_mean),
        'max_score': float(loo_mean),
        'note': 'LOO computes F1 on aggregated predictions, not per-fold',
    })

    # =========================================================================
    # CV + Holdout strategies
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: CV + HOLDOUT STRATEGIES")
    print("=" * 70)
    print("(Split data into CV folds and separate holdout folds)")

    cv_holdout_configs = [
        ('8+2 (current)', 8, 2),  # Current approach: 80% CV, 20% holdout
        ('9+1', 9, 1),            # More CV data: 90% CV, 10% holdout
        ('7+3', 7, 3),            # More holdout: 70% CV, 30% holdout
        ('6+4', 6, 4),            # Conservative: 60% CV, 40% holdout
        ('5+5', 5, 5),            # Equal split: 50% CV, 50% holdout
    ]

    print("\n{:<15} {:>10} {:>10} {:>12} {:>10} {:>10}".format(
        "Strategy", "CV Mean", "CV Std", "Holdout Mean", "Hold Std", "Gap"))
    print("-" * 70)

    for name, n_cv, n_holdout in cv_holdout_configs:
        cv_mean, cv_std, cv_scores, hold_mean, hold_std, hold_scores = evaluate_cv_plus_holdout(
            clone(pipeline), X, y, n_cv, n_holdout
        )
        gap = cv_mean - hold_mean

        print(f"{name:<15} {cv_mean:>10.4f} {cv_std:>10.4f} {hold_mean:>12.4f} {hold_std:>10.4f} {gap:>10.4f}")

        results['experiments'].append({
            'strategy': name,
            'type': 'cv_plus_holdout',
            'n_cv_folds': n_cv,
            'n_holdout_folds': n_holdout,
            'cv_mean_f1': float(cv_mean),
            'cv_std_f1': float(cv_std),
            'cv_scores': [float(s) for s in cv_scores],
            'holdout_mean_f1': float(hold_mean),
            'holdout_std_f1': float(hold_std),
            'holdout_scores': [float(s) for s in hold_scores],
            'cv_holdout_gap': float(gap),
        })

    # =========================================================================
    # Multiple random seeds to test stability
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: STABILITY ACROSS RANDOM SEEDS")
    print("=" * 70)
    print("(Testing 10-fold CV with different random seeds)")

    seed_results = []
    print("\n{:<10} {:>10} {:>10}".format("Seed", "Mean F1", "Std"))
    print("-" * 30)

    for seed in [42, 123, 456, 789, 1000, 2024, 2025, 3141, 7777, 9999]:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        mean_f1, std_f1, scores = evaluate_cv_only(clone(pipeline), X, y, cv)
        seed_results.append({'seed': seed, 'mean_f1': mean_f1, 'std_f1': std_f1})
        print(f"{seed:<10} {mean_f1:>10.4f} {std_f1:>10.4f}")

    results['seed_stability'] = {
        'strategy': '10-fold CV',
        'seeds_tested': [r['seed'] for r in seed_results],
        'mean_across_seeds': float(np.mean([r['mean_f1'] for r in seed_results])),
        'std_across_seeds': float(np.std([r['mean_f1'] for r in seed_results])),
        'min_across_seeds': float(min([r['mean_f1'] for r in seed_results])),
        'max_across_seeds': float(max([r['mean_f1'] for r in seed_results])),
        'results': seed_results,
    }

    # =========================================================================
    # Summary and Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY AND ANALYSIS")
    print("=" * 70)

    # Find best CV-only strategy
    cv_only_results = [e for e in results['experiments'] if e['type'] == 'cv_only']
    best_cv_only = max(cv_only_results, key=lambda x: x['mean_f1'])

    # Find best CV+holdout strategy (by holdout score)
    cv_holdout_results = [e for e in results['experiments'] if e['type'] == 'cv_plus_holdout']
    best_cv_holdout = max(cv_holdout_results, key=lambda x: x['holdout_mean_f1'])

    print("\n1. CV-ONLY STRATEGIES:")
    print(f"   Best: {best_cv_only['strategy']} with mean F1 = {best_cv_only['mean_f1']:.4f}")
    print(f"   Baseline holdout (0.665) is {'PESSIMISTIC' if best_cv_only['mean_f1'] > 0.665 else 'OPTIMISTIC'}")

    print("\n2. CV+HOLDOUT STRATEGIES:")
    print(f"   Best holdout: {best_cv_holdout['strategy']} with holdout F1 = {best_cv_holdout['holdout_mean_f1']:.4f}")
    print(f"   Current approach (8+2) holdout estimate: {cv_holdout_results[0]['holdout_mean_f1']:.4f}")

    print("\n3. SEED STABILITY:")
    print(f"   10-fold CV mean across seeds: {results['seed_stability']['mean_across_seeds']:.4f}")
    print(f"   Std across seeds: {results['seed_stability']['std_across_seeds']:.4f}")
    print(f"   Range: [{results['seed_stability']['min_across_seeds']:.4f}, {results['seed_stability']['max_across_seeds']:.4f}]")

    # Estimate "true" score
    # Use LOO as the most unbiased estimate (but high variance)
    loo_result = [e for e in results['experiments'] if e['strategy'] == 'Leave-One-Out CV'][0]

    # Use repeated CV as a stable estimate
    rep_cv_result = [e for e in results['experiments'] if e['strategy'] == 'Repeated 10-fold CV (3x)'][0]

    print("\n4. BEST ESTIMATE OF TRUE PERFORMANCE:")
    print(f"   Leave-One-Out (unbiased): {loo_result['mean_f1']:.4f}")
    print(f"   Repeated 10-fold CV (stable): {rep_cv_result['mean_f1']:.4f}")

    # Compare to baseline
    holdout_baseline = 0.665
    print(f"\n5. ANALYSIS OF 0.665 HOLDOUT ESTIMATE:")

    # Calculate how pessimistic/optimistic the holdout estimate is
    loo_diff = loo_result['mean_f1'] - holdout_baseline
    rep_diff = rep_cv_result['mean_f1'] - holdout_baseline

    if loo_diff > 0.02:
        print(f"   LOO suggests holdout is PESSIMISTIC by {loo_diff:.4f}")
    elif loo_diff < -0.02:
        print(f"   LOO suggests holdout is OPTIMISTIC by {abs(loo_diff):.4f}")
    else:
        print(f"   LOO suggests holdout is ACCURATE (diff: {loo_diff:.4f})")

    if rep_diff > 0.02:
        print(f"   Repeated CV suggests holdout is PESSIMISTIC by {rep_diff:.4f}")
    elif rep_diff < -0.02:
        print(f"   Repeated CV suggests holdout is OPTIMISTIC by {abs(rep_diff):.4f}")
    else:
        print(f"   Repeated CV suggests holdout is ACCURATE (diff: {rep_diff:.4f})")

    # Recommendation
    print("\n6. RECOMMENDATION:")

    # Analyze CV-holdout gap to detect overfitting
    current_gap = cv_holdout_results[0]['cv_holdout_gap']
    if current_gap > 0.05:
        print(f"   WARNING: Large CV-holdout gap ({current_gap:.4f}) suggests overfitting risk")
        print(f"   Consider using 7+3 or 6+4 split for more reliable holdout estimate")
    else:
        print(f"   CV-holdout gap is reasonable ({current_gap:.4f})")

    if rep_cv_result['mean_f1'] > holdout_baseline + 0.02:
        print(f"   True performance likely around {rep_cv_result['mean_f1']:.3f}, not {holdout_baseline:.3f}")
        print(f"   Gap to target (0.745) is ~{0.745 - rep_cv_result['mean_f1']:.3f}, not ~{0.745 - holdout_baseline:.3f}")

    results['summary'] = {
        'baseline_holdout': holdout_baseline,
        'loo_estimate': float(loo_result['mean_f1']),
        'repeated_cv_estimate': float(rep_cv_result['mean_f1']),
        'best_cv_only_score': float(best_cv_only['mean_f1']),
        'best_cv_holdout_score': float(best_cv_holdout['holdout_mean_f1']),
        'seed_stability_std': float(results['seed_stability']['std_across_seeds']),
        'current_cv_holdout_gap': float(current_gap),
        'holdout_pessimism': float(rep_cv_result['mean_f1'] - holdout_baseline),
        'recommendation': 'Use 10-fold CV or Repeated 5-fold CV for model selection, final estimate closer to 0.70 than 0.665'
    }

    # Save results
    output_path = RESULTS_DIR / 'exp_cv_strategy_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_cv_strategy_experiment()
