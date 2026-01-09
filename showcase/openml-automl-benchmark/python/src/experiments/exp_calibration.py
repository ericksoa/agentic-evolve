#!/usr/bin/env python3
"""
Probability Calibration Experiment for Diabetes Classification

Tests CalibratedClassifierCV with different methods to improve F1 scores:
- sigmoid (Platt scaling) - works well for models that output uncalibrated probabilities
- isotonic regression - more flexible, non-parametric

Combined with threshold optimization (instead of default 0.5).

Current best: 0.665 holdout F1 (Domain + Bins + LogReg)
Target: 0.745 (Auto-sklearn level)

Hypothesis: Better probability estimates + optimal threshold can improve F1.
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for diabetes dataset.
    Using Domain + Bins (the best configuration from previous experiments).
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


def find_optimal_threshold(y_true, y_prob, thresholds):
    """Find the threshold that maximizes F1 score."""
    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def evaluate_with_calibration(feature_engineer, base_classifier, calibration_method,
                               thresholds, X, y, n_cv_folds=8, n_holdout_folds=2):
    """
    Evaluate pipeline with probability calibration and threshold optimization.

    Parameters:
    - calibration_method: None (no calibration), 'sigmoid' (Platt scaling), or 'isotonic'
    - thresholds: list of thresholds to test

    Returns scores for each threshold and the optimal threshold.
    """
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    # Results for each threshold
    cv_scores_by_threshold = {t: [] for t in thresholds}
    holdout_scores_by_threshold = {t: [] for t in thresholds}

    # Track optimal threshold per fold
    cv_optimal_thresholds = []
    holdout_optimal_thresholds = []

    # CV evaluation
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

        # Create and train classifier with optional calibration
        clf = clone(base_classifier)

        if calibration_method is not None:
            # CalibratedClassifierCV uses internal CV for calibration
            calibrated_clf = CalibratedClassifierCV(
                clf, method=calibration_method, cv=3
            )
            calibrated_clf.fit(X_train_scaled, y_train)
            y_prob = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Evaluate at each threshold
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cv_scores_by_threshold[thresh].append(f1)

        # Find optimal threshold for this fold
        opt_thresh, _ = find_optimal_threshold(y_test, y_prob, thresholds)
        cv_optimal_thresholds.append(opt_thresh)

    # Holdout evaluation
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

        # Create and train classifier with optional calibration
        clf = clone(base_classifier)

        if calibration_method is not None:
            calibrated_clf = CalibratedClassifierCV(
                clf, method=calibration_method, cv=3
            )
            calibrated_clf.fit(X_train_scaled, y_train)
            y_prob = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Evaluate at each threshold
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            holdout_scores_by_threshold[thresh].append(f1)

        # Find optimal threshold for this fold
        opt_thresh, _ = find_optimal_threshold(y_test, y_prob, thresholds)
        holdout_optimal_thresholds.append(opt_thresh)

    # Compute mean scores
    cv_means = {t: np.mean(scores) for t, scores in cv_scores_by_threshold.items()}
    holdout_means = {t: np.mean(scores) for t, scores in holdout_scores_by_threshold.items()}

    # Find best threshold based on CV performance
    best_cv_threshold = max(cv_means, key=cv_means.get)
    best_cv_f1 = cv_means[best_cv_threshold]

    # Corresponding holdout performance at the CV-selected threshold
    holdout_f1_at_cv_best = holdout_means[best_cv_threshold]

    # Also track what holdout would be at default 0.5
    holdout_f1_default = holdout_means.get(0.5, holdout_means[0.5])

    return {
        'cv_scores_by_threshold': {str(k): float(v) for k, v in cv_means.items()},
        'holdout_scores_by_threshold': {str(k): float(v) for k, v in holdout_means.items()},
        'best_cv_threshold': float(best_cv_threshold),
        'best_cv_f1': float(best_cv_f1),
        'holdout_f1_at_cv_best': float(holdout_f1_at_cv_best),
        'holdout_f1_default_threshold': float(holdout_f1_default),
        'cv_optimal_threshold_mean': float(np.mean(cv_optimal_thresholds)),
        'holdout_optimal_threshold_mean': float(np.mean(holdout_optimal_thresholds)),
    }


def run_calibration_experiment():
    """Run comprehensive calibration experiment."""
    print("=" * 70)
    print("PROBABILITY CALIBRATION EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)

    print("\nLoading diabetes dataset...")
    X, y = load_diabetes_raw()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y)} (ratio: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1)")

    results = {
        'experiment': 'Probability Calibration for Diabetes Classification',
        'dataset': 'OpenML 37 (diabetes)',
        'baseline': 0.665,
        'target': 0.745,
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # Feature engineering: Domain + Bins (the best from previous experiments)
    feature_engineer = DiabetesFeatureEngineer(add_domain=True, add_bins=True)

    # Thresholds to test
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

    # Calibration methods
    calibration_methods = {
        'None (uncalibrated)': None,
        'Sigmoid (Platt scaling)': 'sigmoid',
        'Isotonic regression': 'isotonic',
    }

    # Base classifiers
    base_classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'LogReg': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
    }

    if XGB_AVAILABLE:
        base_classifiers['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            eval_metric='logloss', verbosity=0
        )

    best_holdout = 0
    best_config = None

    # Run experiments
    exp_num = 0
    total_experiments = len(calibration_methods) * len(base_classifiers)

    for calib_name, calib_method in calibration_methods.items():
        print(f"\n{'='*70}")
        print(f"Calibration Method: {calib_name}")
        print('='*70)

        for clf_name, clf in base_classifiers.items():
            exp_num += 1
            print(f"\n  [{exp_num}/{total_experiments}] {clf_name}...")

            try:
                eval_results = evaluate_with_calibration(
                    clone(feature_engineer), clf, calib_method, thresholds, X, y
                )

                print(f"    Thresholds tested: {thresholds}")
                print(f"    Best CV threshold: {eval_results['best_cv_threshold']}")
                print(f"    CV F1 at best: {eval_results['best_cv_f1']:.4f}")
                print(f"    Holdout F1 at best CV threshold: {eval_results['holdout_f1_at_cv_best']:.4f}")
                print(f"    Holdout F1 at default 0.5: {eval_results['holdout_f1_default_threshold']:.4f}")

                exp_result = {
                    'calibration_method': calib_name,
                    'classifier': clf_name,
                    'thresholds_tested': thresholds,
                    **eval_results
                }
                results['experiments'].append(exp_result)

                # Track best configuration (using holdout at CV-selected threshold)
                if eval_results['holdout_f1_at_cv_best'] > best_holdout:
                    best_holdout = eval_results['holdout_f1_at_cv_best']
                    best_config = exp_result

            except Exception as e:
                print(f"    ERROR: {e}")
                results['experiments'].append({
                    'calibration_method': calib_name,
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

    print(f"\nBaseline (Domain + Bins + LogReg, threshold=0.5): 0.665")
    print(f"Best achieved: {best_holdout:.4f}")
    print(f"Beat baseline: {'YES' if best_holdout > 0.665 else 'NO'}")
    print(f"Improvement: {(best_holdout - 0.665):+.4f}")

    if best_config:
        print(f"\nBest configuration:")
        print(f"  Calibration: {best_config['calibration_method']}")
        print(f"  Classifier: {best_config['classifier']}")
        print(f"  Threshold: {best_config['best_cv_threshold']}")
        print(f"  Holdout F1: {best_config['holdout_f1_at_cv_best']:.4f}")

    # Detailed breakdown by calibration method
    print("\nBest holdout F1 by calibration method:")
    calib_best = {}
    for exp in results['experiments']:
        if 'error' not in exp:
            method = exp['calibration_method']
            holdout = exp['holdout_f1_at_cv_best']
            if method not in calib_best or holdout > calib_best[method]:
                calib_best[method] = holdout

    for method, score in sorted(calib_best.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if score == best_holdout else ""
        print(f"  {method}: {score:.4f}{marker}")

    # Best by classifier
    print("\nBest holdout F1 by classifier:")
    clf_best = {}
    for exp in results['experiments']:
        if 'error' not in exp:
            clf = exp['classifier']
            holdout = exp['holdout_f1_at_cv_best']
            if clf not in clf_best or holdout > clf_best[clf]:
                clf_best[clf] = holdout

    for clf, score in sorted(clf_best.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if score == best_holdout else ""
        print(f"  {clf}: {score:.4f}{marker}")

    # Threshold analysis
    print("\nThreshold impact analysis (averaged across all experiments):")
    threshold_avg = {t: [] for t in thresholds}
    for exp in results['experiments']:
        if 'error' not in exp:
            for t in thresholds:
                if str(t) in exp['holdout_scores_by_threshold']:
                    threshold_avg[t].append(exp['holdout_scores_by_threshold'][str(t)])

    for thresh, scores in sorted(threshold_avg.items()):
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"  Threshold {thresh}: {avg:.4f} (+/- {std:.4f})")

    # Save results
    output_path = RESULTS_DIR / 'exp_calibration_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_calibration_experiment()
