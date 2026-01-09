#!/usr/bin/env python3
"""
Stacking Experiment for Diabetes Classification

Tests sklearn's StackingClassifier with various configurations:
- Base models: LogReg, RandomForest, XGBoost, SVM
- Meta-learners: LogReg, Ridge
- With and without Domain+Bins feature engineering

Target: Beat 0.665 holdout F1 (current best with Domain + Bins + LogReg)
Goal: Match Auto-sklearn level (~0.745)

Reference: AutoML tools like Auto-sklearn use stacking extensively.
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
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

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not available, some configurations will be skipped")

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
            # Domain-specific features based on diabetes knowledge
            domain = pd.DataFrame()

            # BMI categories (WHO classification)
            domain['bmi_underweight'] = (df['mass'] < 18.5).astype(float)
            domain['bmi_normal'] = ((df['mass'] >= 18.5) & (df['mass'] < 25)).astype(float)
            domain['bmi_overweight'] = ((df['mass'] >= 25) & (df['mass'] < 30)).astype(float)
            domain['bmi_obese'] = (df['mass'] >= 30).astype(float)

            # Age groups (risk increases with age)
            domain['age_young'] = (df['age'] < 30).astype(float)
            domain['age_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(float)
            domain['age_senior'] = (df['age'] >= 50).astype(float)

            # Glucose categories (pre-diabetes thresholds)
            domain['glucose_normal'] = (df['plas'] < 100).astype(float)
            domain['glucose_prediabetic'] = ((df['plas'] >= 100) & (df['plas'] < 126)).astype(float)
            domain['glucose_diabetic'] = (df['plas'] >= 126).astype(float)

            # High-risk flag (multiple risk factors)
            domain['high_risk'] = (
                (df['mass'] >= 30) & (df['age'] >= 40) & (df['plas'] >= 100)
            ).astype(float)

            # Pregnancy risk (gestational diabetes risk)
            domain['preg_risk'] = (df['preg'] >= 4).astype(float)

            features.append(domain)

        if self.add_ratios:
            # Ratio features
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
    print("Loading diabetes dataset from OpenML (dataset 37)...")
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

    print(f"Loaded: {X_imputed.shape[0]} samples, {X_imputed.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y.values)}")

    return X_imputed, y.values


def create_base_estimators(use_xgb=True, use_svm=True):
    """Create base estimators for stacking."""
    estimators = [
        ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
    ]

    if use_xgb and XGB_AVAILABLE:
        estimators.append(
            ('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=4, random_state=42,
                eval_metric='logloss', verbosity=0, n_jobs=-1
            ))
        )

    if use_svm:
        # SVC with probability=True for soft stacking
        estimators.append(
            ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
        )

    return estimators


def create_stacking_classifier(base_estimators, meta_learner='logreg', use_proba=True):
    """Create a StackingClassifier with given base estimators and meta-learner."""
    if meta_learner == 'logreg':
        final_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif meta_learner == 'logreg_balanced':
        final_estimator = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    elif meta_learner == 'ridge':
        final_estimator = RidgeClassifier(alpha=1.0, random_state=42)
    elif meta_learner == 'rf':
        final_estimator = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    else:
        raise ValueError(f"Unknown meta_learner: {meta_learner}")

    # stack_method: 'predict_proba' for soft stacking, 'predict' for hard
    stack_method = 'predict_proba' if use_proba else 'predict'

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,  # Internal CV for generating meta-features
        stack_method=stack_method,
        passthrough=False,  # Don't pass original features to meta-learner
        n_jobs=-1
    )


def create_pipeline(feature_engineer, classifier):
    """Create a pipeline with feature engineering and classification."""
    return Pipeline([
        ('features', feature_engineer),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])


def evaluate_pipeline(pipeline, X, y, n_cv_folds=8, n_holdout_folds=2):
    """Evaluate pipeline with CV + holdout."""
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

    return {
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'holdout_mean': np.mean(holdout_scores),
        'holdout_std': np.std(holdout_scores),
        'cv_scores': cv_scores,
        'holdout_scores': holdout_scores,
        'gap': np.mean(cv_scores) - np.mean(holdout_scores)
    }


def run_stacking_experiment():
    """Run the full stacking experiment."""
    print("=" * 70)
    print("STACKING EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)
    print(f"\nBaseline to beat: 0.665 holdout F1 (Domain + Bins + LogReg)")
    print(f"Target: 0.745 (Auto-sklearn level)")
    print()

    X, y = load_diabetes_raw()

    results = []

    # Define configurations to test
    configs = []

    # =========================================================================
    # CONFIG 1: No feature engineering, basic stacking
    # =========================================================================
    configs.append({
        'name': 'Stack-Basic (LR+RF+XGB+SVM) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg',
        'use_proba': True,
        'features': 'base_only'
    })

    # =========================================================================
    # CONFIG 2: No FE, stacking with Ridge meta-learner
    # =========================================================================
    configs.append({
        'name': 'Stack-Basic (LR+RF+XGB+SVM) -> Ridge',
        'feature_engineer': DiabetesFeatureEngineer(),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'ridge',
        'use_proba': False,  # Ridge needs predict, not predict_proba
        'features': 'base_only'
    })

    # =========================================================================
    # CONFIG 3: Domain+Bins features + stacking
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainBins (LR+RF+XGB+SVM) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg',
        'use_proba': True,
        'features': 'domain+bins'
    })

    # =========================================================================
    # CONFIG 4: Domain+Bins + balanced LogReg meta-learner
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainBins (LR+RF+XGB+SVM) -> LogReg-Balanced',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg_balanced',
        'use_proba': True,
        'features': 'domain+bins'
    })

    # =========================================================================
    # CONFIG 5: Domain+Bins + Ridge meta-learner
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainBins (LR+RF+XGB+SVM) -> Ridge',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'ridge',
        'use_proba': False,
        'features': 'domain+bins'
    })

    # =========================================================================
    # CONFIG 6: Domain+Bins + simpler base (LR+RF only)
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainBins (LR+RF) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=False, use_svm=False),
        'meta_learner': 'logreg',
        'use_proba': True,
        'features': 'domain+bins'
    })

    # =========================================================================
    # CONFIG 7: All features + stacking
    # =========================================================================
    configs.append({
        'name': 'Stack-AllFE (LR+RF+XGB+SVM) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(
            add_domain=True, add_bins=True, add_ratios=True, add_interactions=True
        ),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg',
        'use_proba': True,
        'features': 'all'
    })

    # =========================================================================
    # CONFIG 8: Domain only (no bins) + stacking
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainOnly (LR+RF+XGB+SVM) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg',
        'use_proba': True,
        'features': 'domain'
    })

    # =========================================================================
    # CONFIG 9: Hard stacking (using predict instead of predict_proba)
    # =========================================================================
    configs.append({
        'name': 'Stack-Hard-DomainBins (LR+RF+XGB+SVM) -> LogReg',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'logreg',
        'use_proba': False,  # Hard voting
        'features': 'domain+bins'
    })

    # =========================================================================
    # CONFIG 10: Stacking with RF as meta-learner
    # =========================================================================
    configs.append({
        'name': 'Stack-DomainBins (LR+RF+XGB+SVM) -> RF-Meta',
        'feature_engineer': DiabetesFeatureEngineer(add_domain=True, add_bins=True),
        'base_estimators': create_base_estimators(use_xgb=True, use_svm=True),
        'meta_learner': 'rf',
        'use_proba': True,
        'features': 'domain+bins'
    })

    # =========================================================================
    # RUN ALL CONFIGURATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING CONFIGURATIONS")
    print("=" * 70)

    best_holdout = 0
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}")
        print("-" * 60)

        try:
            # Create stacking classifier
            stacker = create_stacking_classifier(
                config['base_estimators'],
                config['meta_learner'],
                config['use_proba']
            )

            # Create pipeline
            pipeline = create_pipeline(config['feature_engineer'], stacker)

            # Evaluate
            metrics = evaluate_pipeline(pipeline, X, y, n_cv_folds=8, n_holdout_folds=2)

            result = {
                'config_name': config['name'],
                'features': config['features'],
                'meta_learner': config['meta_learner'],
                'use_proba': bool(config['use_proba']),
                'cv_mean': float(metrics['cv_mean']),
                'cv_std': float(metrics['cv_std']),
                'holdout_mean': float(metrics['holdout_mean']),
                'holdout_std': float(metrics['holdout_std']),
                'gap': float(metrics['gap']),
                'cv_scores': [float(s) for s in metrics['cv_scores']],
                'holdout_scores': [float(s) for s in metrics['holdout_scores']],
                'beats_baseline': bool(metrics['holdout_mean'] > 0.665),
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            # Print results
            status = "BEATS BASELINE!" if result['beats_baseline'] else "below baseline"
            print(f"  CV F1:      {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            print(f"  Holdout F1: {metrics['holdout_mean']:.4f} (+/- {metrics['holdout_std']:.4f})")
            print(f"  Gap:        {metrics['gap']:.4f}")
            print(f"  Status:     {status}")

            if metrics['holdout_mean'] > best_holdout:
                best_holdout = metrics['holdout_mean']
                best_config = result
                print(f"  >>> NEW BEST! <<<")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'config_name': config['name'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nBaseline: 0.665 holdout F1 (Domain + Bins + LogReg)")
    print(f"Target:   0.745 holdout F1 (Auto-sklearn level)")

    if best_config:
        print(f"\nBest Configuration:")
        print(f"  Name:       {best_config['config_name']}")
        print(f"  Features:   {best_config['features']}")
        print(f"  Holdout F1: {best_config['holdout_mean']:.4f}")
        print(f"  CV F1:      {best_config['cv_mean']:.4f}")

        if best_config['holdout_mean'] > 0.665:
            improvement = ((best_config['holdout_mean'] - 0.665) / 0.665) * 100
            print(f"\n  BEATS BASELINE by {improvement:.1f}%")
        else:
            deficit = ((0.665 - best_config['holdout_mean']) / 0.665) * 100
            print(f"\n  Below baseline by {deficit:.1f}%")

        if best_config['holdout_mean'] >= 0.745:
            print(f"  MATCHES/BEATS AUTO-SKLEARN TARGET!")
        else:
            gap_to_target = 0.745 - best_config['holdout_mean']
            print(f"  Gap to Auto-sklearn target: {gap_to_target:.4f}")

    # Rank all results
    print("\n" + "-" * 70)
    print("RANKINGS (by holdout F1)")
    print("-" * 70)

    valid_results = [r for r in results if 'holdout_mean' in r]
    ranked = sorted(valid_results, key=lambda x: x['holdout_mean'], reverse=True)

    for i, r in enumerate(ranked):
        status = "[BEATS]" if r.get('beats_baseline', False) else "       "
        print(f"{i+1}. {status} {r['holdout_mean']:.4f} - {r['config_name']}")

    # Save results
    output = {
        'experiment': 'stacking',
        'dataset': 'diabetes (OpenML 37)',
        'baseline': 0.665,
        'target': 0.745,
        'best_holdout_f1': float(best_holdout),
        'best_config': best_config,
        'beats_baseline': bool(best_holdout > 0.665),
        'all_results': results,
        'timestamp': datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / 'exp_stacking_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    results = run_stacking_experiment()
