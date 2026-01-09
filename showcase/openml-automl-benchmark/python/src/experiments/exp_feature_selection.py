#!/usr/bin/env python3
"""
Feature Selection Experiment for Diabetes Classification

Goal: Test whether feature selection can improve upon the 0.665 holdout F1 baseline
achieved with Domain + Bins + LogReg.

Methods tested:
- SelectKBest (ANOVA F-score)
- RFE (Recursive Feature Elimination)
- Mutual Information
- L1-based (Lasso) feature selection

Feature counts tested: 5, 8, 10, 12, 15

Models: LogReg, RandomForest

Validation: 8-fold CV + 2-fold holdout (same as evolve_diabetes_features.py)
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain + Bins feature engineering for diabetes dataset.
    This is the current best configuration (0.665 holdout F1).
    """

    def __init__(self):
        self.feature_names_ = None
        self.quantiles_ = {}

    def fit(self, X, y=None):
        # Store quantiles for bin features
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])
        for col in ['plas', 'mass', 'age', 'pedi']:
            self.quantiles_[f'{col}_q1'] = df[col].quantile(0.25)
            self.quantiles_[f'{col}_q4'] = df[col].quantile(0.75)
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        features = [df.copy()]

        # Domain features
        domain = pd.DataFrame()

        # BMI categories (WHO classification)
        domain['bmi_underweight'] = (df['mass'] < 18.5).astype(float)
        domain['bmi_normal'] = ((df['mass'] >= 18.5) & (df['mass'] < 25)).astype(float)
        domain['bmi_overweight'] = ((df['mass'] >= 25) & (df['mass'] < 30)).astype(float)
        domain['bmi_obese'] = (df['mass'] >= 30).astype(float)

        # Age groups
        domain['age_young'] = (df['age'] < 30).astype(float)
        domain['age_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(float)
        domain['age_senior'] = (df['age'] >= 50).astype(float)

        # Glucose categories
        domain['glucose_normal'] = (df['plas'] < 100).astype(float)
        domain['glucose_prediabetic'] = ((df['plas'] >= 100) & (df['plas'] < 126)).astype(float)
        domain['glucose_diabetic'] = (df['plas'] >= 126).astype(float)

        # Risk flags
        domain['high_risk'] = (
            (df['mass'] >= 30) & (df['age'] >= 40) & (df['plas'] >= 100)
        ).astype(float)
        domain['preg_risk'] = (df['preg'] >= 4).astype(float)

        features.append(domain)

        # Quantile bin features
        bins = pd.DataFrame()
        for col in ['plas', 'mass', 'age', 'pedi']:
            bins[f'{col}_q1'] = (df[col] <= self.quantiles_[f'{col}_q1']).astype(float)
            bins[f'{col}_q4'] = (df[col] >= self.quantiles_[f'{col}_q4']).astype(float)

        features.append(bins)

        result = pd.concat(features, axis=1)
        self.feature_names_ = list(result.columns)
        return result.values

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            # Return default names
            base = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
            domain = ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese',
                     'age_young', 'age_middle', 'age_senior',
                     'glucose_normal', 'glucose_prediabetic', 'glucose_diabetic',
                     'high_risk', 'preg_risk']
            bins = [f'{col}_{q}' for col in ['plas', 'mass', 'age', 'pedi'] for q in ['q1', 'q4']]
            return base + domain + bins
        return self.feature_names_


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

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)

    return X_imputed, y.values


def evaluate_with_feature_selection(X, y, selector_name, selector, n_features, classifier_name, classifier,
                                    n_cv_folds=8, n_holdout_folds=2):
    """
    Evaluate a classifier with feature selection.

    Pipeline: FeatureEngineer -> Scaler -> FeatureSelector -> Classifier
    """
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    cv_scores = []
    holdout_scores = []
    selected_features_all = []

    # Feature engineer (fit once to get feature names)
    fe = DiabetesFeatureEngineer()
    fe.fit(X)
    feature_names = fe.get_feature_names_out()

    for split_type, splits in [('cv', cv_splits), ('holdout', holdout_splits)]:
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Feature engineering
            fe = DiabetesFeatureEngineer()
            X_train_fe = fe.fit_transform(X_train)
            X_test_fe = fe.transform(X_test)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fe)
            X_test_scaled = scaler.transform(X_test_fe)

            # Feature selection
            sel = clone(selector)
            X_train_sel = sel.fit_transform(X_train_scaled, y_train)
            X_test_sel = sel.transform(X_test_scaled)

            # Track selected features
            if hasattr(sel, 'get_support'):
                mask = sel.get_support()
                selected_features = [feature_names[i] for i, m in enumerate(mask) if m]
                selected_features_all.append(selected_features)

            # Classification
            clf = clone(classifier)
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)

            score = f1_score(y_test, y_pred, zero_division=0)
            if split_type == 'cv':
                cv_scores.append(score)
            else:
                holdout_scores.append(score)

    return {
        'selector': selector_name,
        'n_features': n_features,
        'classifier': classifier_name,
        'cv_f1_mean': float(np.mean(cv_scores)),
        'cv_f1_std': float(np.std(cv_scores)),
        'holdout_f1_mean': float(np.mean(holdout_scores)),
        'holdout_f1_std': float(np.std(holdout_scores)),
        'gap': float(np.mean(cv_scores) - np.mean(holdout_scores)),
        'selected_features': selected_features_all[0] if selected_features_all else []
    }


def run_feature_selection_experiment():
    """Run the complete feature selection experiment."""
    print("=" * 70)
    print("FEATURE SELECTION EXPERIMENT FOR DIABETES")
    print("=" * 70)
    print(f"Baseline: Domain + Bins + LogReg = 0.665 holdout F1")
    print(f"Target: 0.745 (Auto-sklearn level)")
    print()

    # Load data
    print("Loading diabetes dataset...")
    X, y = load_diabetes_raw()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")

    # Get engineered feature count
    fe = DiabetesFeatureEngineer()
    X_fe = fe.fit_transform(X)
    n_total_features = X_fe.shape[1]
    feature_names = fe.get_feature_names_out()
    print(f"After Domain+Bins engineering: {n_total_features} features")
    print(f"Features: {feature_names}")
    print()

    # Define classifiers
    classifiers = {
        'LogReg': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
    }

    # Define feature counts to test
    n_features_list = [5, 8, 10, 12, 15]

    # Results storage
    all_results = []

    # Run experiments
    print("=" * 70)
    print("RUNNING FEATURE SELECTION EXPERIMENTS")
    print("=" * 70)

    for n_features in n_features_list:
        if n_features > n_total_features:
            print(f"Skipping n_features={n_features} (> total {n_total_features})")
            continue

        print(f"\n--- Testing with {n_features} features ---")

        # Define selectors for this feature count
        selectors = {
            'SelectKBest_ANOVA': SelectKBest(score_func=f_classif, k=n_features),
            'SelectKBest_MI': SelectKBest(score_func=mutual_info_classif, k=n_features),
            'RFE_LogReg': RFE(
                estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                n_features_to_select=n_features
            ),
            'L1_Selection': SelectFromModel(
                LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=2000, random_state=42),
                max_features=n_features,
                threshold=-np.inf
            )
        }

        for selector_name, selector in selectors.items():
            for clf_name, clf in classifiers.items():
                print(f"  {selector_name} + {clf_name}...", end=" ", flush=True)

                result = evaluate_with_feature_selection(
                    X, y, selector_name, selector, n_features, clf_name, clf
                )
                all_results.append(result)

                # Print inline result
                status = "BEATS" if result['holdout_f1_mean'] > 0.665 else "below"
                print(f"Holdout F1: {result['holdout_f1_mean']:.4f} ({status} baseline)")

    # Run baseline for comparison (no feature selection)
    print("\n--- Baseline (no feature selection) ---")
    for clf_name, clf in classifiers.items():
        print(f"  No Selection + {clf_name}...", end=" ", flush=True)

        # Manual evaluation without selector
        total_folds = 10
        skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
        all_splits = list(skf.split(X, y))
        cv_splits = all_splits[:8]
        holdout_splits = all_splits[8:]

        cv_scores = []
        holdout_scores = []

        for split_type, splits in [('cv', cv_splits), ('holdout', holdout_splits)]:
            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                fe = DiabetesFeatureEngineer()
                X_train_fe = fe.fit_transform(X_train)
                X_test_fe = fe.transform(X_test)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fe)
                X_test_scaled = scaler.transform(X_test_fe)

                c = clone(clf)
                c.fit(X_train_scaled, y_train)
                y_pred = c.predict(X_test_scaled)

                score = f1_score(y_test, y_pred, zero_division=0)
                if split_type == 'cv':
                    cv_scores.append(score)
                else:
                    holdout_scores.append(score)

        baseline_result = {
            'selector': 'None',
            'n_features': n_total_features,
            'classifier': clf_name,
            'cv_f1_mean': float(np.mean(cv_scores)),
            'cv_f1_std': float(np.std(cv_scores)),
            'holdout_f1_mean': float(np.mean(holdout_scores)),
            'holdout_f1_std': float(np.std(holdout_scores)),
            'gap': float(np.mean(cv_scores) - np.mean(holdout_scores)),
            'selected_features': feature_names
        }
        all_results.append(baseline_result)
        print(f"Holdout F1: {baseline_result['holdout_f1_mean']:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by holdout F1
    sorted_results = sorted(all_results, key=lambda x: x['holdout_f1_mean'], reverse=True)

    print("\nTop 10 Configurations by Holdout F1:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Selector':<20} {'K':<4} {'Model':<8} {'CV F1':<8} {'Holdout':<8} {'Gap':<8}")
    print("-" * 70)

    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:<5} {r['selector']:<20} {r['n_features']:<4} {r['classifier']:<8} "
              f"{r['cv_f1_mean']:.4f}   {r['holdout_f1_mean']:.4f}   {r['gap']:.4f}")

    # Best result
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"Selector: {best['selector']}")
    print(f"Number of features: {best['n_features']}")
    print(f"Classifier: {best['classifier']}")
    print(f"CV F1: {best['cv_f1_mean']:.4f} (+/- {best['cv_f1_std']:.4f})")
    print(f"Holdout F1: {best['holdout_f1_mean']:.4f} (+/- {best['holdout_f1_std']:.4f})")
    print(f"CV-Holdout Gap: {best['gap']:.4f}")

    if best['selected_features']:
        print(f"\nSelected Features ({len(best['selected_features'])}):")
        for f in best['selected_features']:
            print(f"  - {f}")

    # Comparison to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print(f"Current baseline (Domain+Bins+LogReg): 0.665 holdout F1")
    print(f"Best from this experiment: {best['holdout_f1_mean']:.4f}")

    improvement = best['holdout_f1_mean'] - 0.665
    if improvement > 0:
        print(f"Improvement: +{improvement:.4f} ({improvement/0.665*100:.1f}%)")
        print("STATUS: BEATS BASELINE")
    else:
        print(f"Difference: {improvement:.4f}")
        print("STATUS: DID NOT BEAT BASELINE")

    print(f"\nTarget (Auto-sklearn): 0.745")
    print(f"Gap to target: {0.745 - best['holdout_f1_mean']:.4f}")

    # Feature importance analysis
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Count how often each feature is selected
    feature_selection_counts = {}
    for r in all_results:
        if r['selected_features'] and r['selector'] != 'None':
            for f in r['selected_features']:
                feature_selection_counts[f] = feature_selection_counts.get(f, 0) + 1

    if feature_selection_counts:
        sorted_features = sorted(feature_selection_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nMost frequently selected features:")
        for f, count in sorted_features[:15]:
            print(f"  {f}: selected {count} times")

    # Save results
    output = {
        'experiment': 'feature_selection',
        'dataset': 'diabetes (OpenML 37)',
        'baseline_holdout_f1': 0.665,
        'target_holdout_f1': 0.745,
        'timestamp': datetime.now().isoformat(),
        'n_total_features': n_total_features,
        'n_features_tested': n_features_list,
        'selectors_tested': ['SelectKBest_ANOVA', 'SelectKBest_MI', 'RFE_LogReg', 'L1_Selection', 'None'],
        'classifiers_tested': ['LogReg', 'RF'],
        'best_config': best,
        'all_results': sorted_results,
        'feature_selection_frequency': dict(sorted_features) if feature_selection_counts else {},
        'beat_baseline': best['holdout_f1_mean'] > 0.665
    }

    output_path = RESULTS_DIR / 'exp_feature_selection_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    results = run_feature_selection_experiment()
