#!/usr/bin/env python3
"""
SMOTE Experiment for Diabetes Classification

Tests various oversampling techniques to handle class imbalance:
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- BorderlineSMOTE (Focus on decision boundary samples)

Diabetes has 1.87x class imbalance (500 negative, 268 positive).
Current best: 0.665 holdout F1 (Domain + Bins + LogReg)
Target: 0.745 (Auto-sklearn level)
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import SMOTE variants
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for diabetes dataset.
    Copied from evolve_diabetes_features.py for consistency.
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


def evaluate_with_smote(feature_engineer, classifier, sampler, X, y,
                        n_cv_folds=8, n_holdout_folds=2):
    """
    Evaluate pipeline with SMOTE/ADASYN oversampling.

    IMPORTANT: Oversampling is applied ONLY to training data within each fold,
    never to the test set. This prevents data leakage.
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

        # Apply feature engineering
        fe = clone(feature_engineer)
        X_train_fe = fe.fit_transform(X_train, y_train)
        X_test_fe = fe.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fe)
        X_test_scaled = scaler.transform(X_test_fe)

        # Apply oversampling to training data ONLY
        if sampler is not None:
            samp = clone(sampler)
            X_train_resampled, y_train_resampled = samp.fit_resample(X_train_scaled, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train

        # Train classifier
        clf = clone(classifier)
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test_scaled)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply feature engineering
        fe = clone(feature_engineer)
        X_train_fe = fe.fit_transform(X_train, y_train)
        X_test_fe = fe.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fe)
        X_test_scaled = scaler.transform(X_test_fe)

        # Apply oversampling to training data ONLY
        if sampler is not None:
            samp = clone(sampler)
            X_train_resampled, y_train_resampled = samp.fit_resample(X_train_scaled, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train

        # Train classifier
        clf = clone(classifier)
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test_scaled)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(cv_scores), np.mean(holdout_scores), cv_scores, holdout_scores


def run_smote_experiment():
    """Run comprehensive SMOTE experiment."""
    print("=" * 70)
    print("SMOTE EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)

    print("\nLoading diabetes dataset...")
    X, y = load_diabetes_raw()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y)} (ratio: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1)")

    results = {
        'experiment': 'SMOTE for Diabetes Classification',
        'dataset': 'OpenML 37 (diabetes)',
        'baseline': 0.665,
        'target': 0.745,
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # Define SMOTE variants with different configurations
    samplers = {
        'No Oversampling': None,
        'SMOTE (k=5)': SMOTE(k_neighbors=5, random_state=42),
        'SMOTE (k=3)': SMOTE(k_neighbors=3, random_state=42),
        'SMOTE (k=7)': SMOTE(k_neighbors=7, random_state=42),
        'BorderlineSMOTE-1': BorderlineSMOTE(kind='borderline-1', k_neighbors=5, random_state=42),
        'BorderlineSMOTE-2': BorderlineSMOTE(kind='borderline-2', k_neighbors=5, random_state=42),
        'ADASYN': ADASYN(n_neighbors=5, random_state=42),
    }

    # Define feature engineering configurations
    fe_configs = {
        'Base Features': DiabetesFeatureEngineer(),
        'Domain + Bins (best)': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
    }

    # Define classifiers
    classifiers = {
        'LogReg': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'LR+RF Ensemble': VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1))
            ],
            voting='soft'
        ),
    }

    best_holdout = 0
    best_config = None

    # Run experiments
    exp_num = 0
    total_experiments = len(fe_configs) * len(samplers) * len(classifiers)

    for fe_name, fe in fe_configs.items():
        print(f"\n{'='*70}")
        print(f"Feature Engineering: {fe_name}")
        print('='*70)

        for sampler_name, sampler in samplers.items():
            print(f"\n  Sampler: {sampler_name}")
            print("  " + "-"*50)

            for clf_name, clf in classifiers.items():
                exp_num += 1
                print(f"    [{exp_num}/{total_experiments}] {clf_name}...", end=" ", flush=True)

                try:
                    cv_f1, holdout_f1, cv_scores, holdout_scores = evaluate_with_smote(
                        clone(fe), clf, sampler, X, y
                    )

                    print(f"CV: {cv_f1:.4f}, Holdout: {holdout_f1:.4f}")

                    exp_result = {
                        'feature_engineering': fe_name,
                        'sampler': sampler_name,
                        'classifier': clf_name,
                        'cv_f1': float(cv_f1),
                        'holdout_f1': float(holdout_f1),
                        'cv_scores': [float(s) for s in cv_scores],
                        'holdout_scores': [float(s) for s in holdout_scores],
                        'gap': float(cv_f1 - holdout_f1),
                    }
                    results['experiments'].append(exp_result)

                    if holdout_f1 > best_holdout:
                        best_holdout = holdout_f1
                        best_config = exp_result

                except Exception as e:
                    print(f"ERROR: {e}")
                    results['experiments'].append({
                        'feature_engineering': fe_name,
                        'sampler': sampler_name,
                        'classifier': clf_name,
                        'error': str(e),
                    })

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    results['best_config'] = best_config
    results['best_holdout_f1'] = float(best_holdout)
    results['beat_baseline'] = bool(best_holdout > 0.665)
    results['improvement_over_baseline'] = float(best_holdout - 0.665)

    print(f"\nBaseline (Domain + Bins + LogReg): 0.665")
    print(f"Best achieved: {best_holdout:.4f}")
    print(f"Beat baseline: {'YES' if best_holdout > 0.665 else 'NO'}")
    print(f"Improvement: {(best_holdout - 0.665):+.4f}")

    print(f"\nBest configuration:")
    print(f"  Feature Engineering: {best_config['feature_engineering']}")
    print(f"  Sampler: {best_config['sampler']}")
    print(f"  Classifier: {best_config['classifier']}")
    print(f"  Holdout F1: {best_config['holdout_f1']:.4f}")
    print(f"  CV-Holdout Gap: {best_config['gap']:.4f}")

    # Find best by sampler type
    print("\nBest holdout F1 by sampler:")
    sampler_best = {}
    for exp in results['experiments']:
        if 'error' not in exp:
            samp = exp['sampler']
            if samp not in sampler_best or exp['holdout_f1'] > sampler_best[samp]:
                sampler_best[samp] = exp['holdout_f1']

    for samp, score in sorted(sampler_best.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if score == best_holdout else ""
        print(f"  {samp}: {score:.4f}{marker}")

    # Save results
    output_path = RESULTS_DIR / 'exp_smote_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_smote_experiment()
