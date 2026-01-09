#!/usr/bin/env python3
"""
Aggressive Regularization Experiment for Diabetes Classification

This experiment tests very strong regularization to improve generalization
on the small diabetes dataset (768 samples).

Strategies tested:
- LogReg with strong regularization: C values 0.001 to 1.0
- LogReg penalty types: l1, l2, elasticnet
- RF with strong constraints: max_depth, min_samples_leaf, max_features
- Uses Domain+Bins feature engineering (current best)

Evaluation: 8-fold CV + 2-fold holdout validation
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
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration for LogReg regularization sweep
LOGREG_C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
LOGREG_PENALTIES = ['l1', 'l2', 'elasticnet']

# Configuration for RF constraints sweep
RF_MAX_DEPTHS = [2, 3, 4, 5]
RF_MIN_SAMPLES_LEAF = [5, 10, 20, 50]
RF_MAX_FEATURES = ['sqrt', 'log2', 0.3, 0.5]


class DiabetesDomainBinsTransformer(BaseEstimator, TransformerMixin):
    """
    Combined Domain + Bins feature engineering for diabetes dataset.
    This is the best configuration from previous experiments (0.665 baseline).
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Calculate quantiles for binning during fit
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])
        self.quantiles_ = {}
        for col in ['plas', 'mass', 'age', 'pedi']:
            self.quantiles_[col] = {
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        features = [df.copy()]

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

        # Quantile-based bins for continuous variables
        bins = pd.DataFrame()
        for col in ['plas', 'mass', 'age', 'pedi']:
            bins[f'{col}_q1'] = (df[col] <= self.quantiles_[col]['q25']).astype(float)
            bins[f'{col}_q4'] = (df[col] >= self.quantiles_[col]['q75']).astype(float)

        features.append(bins)

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
    # Columns: 0=preg, 1=plas, 2=pres, 3=skin, 4=insu, 5=mass, 6=pedi, 7=age
    for col in [1, 2, 3, 4, 5]:  # plas, pres, skin, insu, mass
        X_values[X_values[:, col] == 0, col] = np.nan

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)

    print(f"Loaded: {X_imputed.shape[0]} samples, {X_imputed.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y.values)}")

    return X_imputed, y.values


def create_pipeline(feature_transformer, classifier):
    """Create a pipeline with feature transformation and classification."""
    return Pipeline([
        ('features', feature_transformer),
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
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'cv_scores': [float(s) for s in cv_scores],
        'holdout_mean': float(np.mean(holdout_scores)),
        'holdout_std': float(np.std(holdout_scores)),
        'holdout_scores': [float(s) for s in holdout_scores],
        'gap': float(np.mean(cv_scores) - np.mean(holdout_scores))
    }


def run_logreg_regularization_sweep(X, y, transformer):
    """Run LogReg regularization sweep."""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION REGULARIZATION SWEEP")
    print("=" * 70)

    results = []
    best_result = None
    best_holdout = 0.0

    # Calculate total experiments
    total_experiments = len(LOGREG_C_VALUES) * len(LOGREG_PENALTIES)
    current_exp = 0

    print(f"\nTesting {total_experiments} LogReg configurations...")

    for C, penalty in product(LOGREG_C_VALUES, LOGREG_PENALTIES):
        current_exp += 1

        # Skip invalid combinations
        if penalty == 'elasticnet':
            solver = 'saga'
            l1_ratio = 0.5
        elif penalty == 'l1':
            solver = 'saga'
            l1_ratio = None
        else:  # l2
            solver = 'lbfgs'
            l1_ratio = None

        config_name = f"C={C}, penalty={penalty}"
        print(f"\n[{current_exp}/{total_experiments}] {config_name}")

        try:
            # Create classifier
            clf_params = {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'class_weight': 'balanced',
                'max_iter': 2000,
                'random_state': 42
            }
            if l1_ratio is not None:
                clf_params['l1_ratio'] = l1_ratio

            clf = LogisticRegression(**clf_params)
            pipeline = create_pipeline(clone(transformer), clf)

            # Evaluate
            metrics = evaluate_pipeline(pipeline, X, y)

            result = {
                'model': 'LogReg',
                'config': {
                    'C': C,
                    'penalty': penalty,
                    'solver': solver,
                    'l1_ratio': l1_ratio
                },
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            print(f"  CV F1:      {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            print(f"  Holdout F1: {metrics['holdout_mean']:.4f} (+/- {metrics['holdout_std']:.4f})")
            print(f"  Gap:        {metrics['gap']:.4f}")

            if metrics['holdout_mean'] > best_holdout:
                best_holdout = metrics['holdout_mean']
                best_result = result
                print(f"  ** NEW BEST **")

        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                'model': 'LogReg',
                'config': {'C': C, 'penalty': penalty},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

    return results, best_result


def run_rf_constraints_sweep(X, y, transformer):
    """Run RandomForest constraints sweep."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST CONSTRAINTS SWEEP")
    print("=" * 70)

    results = []
    best_result = None
    best_holdout = 0.0

    # Calculate total experiments
    total_experiments = len(RF_MAX_DEPTHS) * len(RF_MIN_SAMPLES_LEAF) * len(RF_MAX_FEATURES)
    current_exp = 0

    print(f"\nTesting {total_experiments} RF configurations...")

    for max_depth, min_samples_leaf, max_features in product(
        RF_MAX_DEPTHS, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES
    ):
        current_exp += 1

        config_name = f"depth={max_depth}, min_leaf={min_samples_leaf}, max_feat={max_features}"
        print(f"\n[{current_exp}/{total_experiments}] {config_name}")

        try:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            pipeline = create_pipeline(clone(transformer), clf)

            # Evaluate
            metrics = evaluate_pipeline(pipeline, X, y)

            result = {
                'model': 'RF',
                'config': {
                    'n_estimators': 100,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features
                },
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            print(f"  CV F1:      {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            print(f"  Holdout F1: {metrics['holdout_mean']:.4f} (+/- {metrics['holdout_std']:.4f})")
            print(f"  Gap:        {metrics['gap']:.4f}")

            if metrics['holdout_mean'] > best_holdout:
                best_holdout = metrics['holdout_mean']
                best_result = result
                print(f"  ** NEW BEST **")

        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                'model': 'RF',
                'config': {
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features
                },
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

    return results, best_result


def run_regularization_experiment():
    """Run the full regularization experiment."""
    print("=" * 70)
    print("AGGRESSIVE REGULARIZATION EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)
    print(f"Baseline to beat: 0.665 holdout F1 (Domain + Bins + LogReg)")
    print(f"Target: 0.745 (Auto-sklearn level)")
    print()

    # Load data
    X, y = load_diabetes_raw()

    # Create feature transformer (Domain + Bins - current best)
    transformer = DiabetesDomainBinsTransformer()

    # Run sweeps
    logreg_results, best_logreg = run_logreg_regularization_sweep(X, y, transformer)
    rf_results, best_rf = run_rf_constraints_sweep(X, y, transformer)

    # Combine results
    all_results = logreg_results + rf_results

    # Find overall best
    valid_results = [r for r in all_results if 'metrics' in r]
    sorted_results = sorted(valid_results, key=lambda x: x['metrics']['holdout_mean'], reverse=True)

    overall_best = sorted_results[0] if sorted_results else None
    best_holdout = overall_best['metrics']['holdout_mean'] if overall_best else 0.0

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if overall_best:
        print(f"\nOverall Best Configuration:")
        print(f"  Model:        {overall_best['model']}")
        for key, value in overall_best['config'].items():
            print(f"  {key}: {value}")
        print(f"\nBest Metrics:")
        print(f"  CV F1:          {overall_best['metrics']['cv_mean']:.4f}")
        print(f"  Holdout F1:     {overall_best['metrics']['holdout_mean']:.4f}")
        print(f"  Gap:            {overall_best['metrics']['gap']:.4f}")

    # Compare to baseline
    baseline = 0.665
    target = 0.745
    print(f"\nComparison:")
    print(f"  vs Baseline (0.665): {'+' if best_holdout > baseline else ''}{(best_holdout - baseline)*100:.2f}%")
    print(f"  vs Target (0.745):   {'+' if best_holdout > target else ''}{(best_holdout - target)*100:.2f}%")
    print(f"  Beat baseline: {'YES' if best_holdout > baseline else 'NO'}")

    # Best LogReg
    if best_logreg:
        print(f"\nBest LogReg: C={best_logreg['config']['C']}, penalty={best_logreg['config']['penalty']}")
        print(f"  Holdout F1: {best_logreg['metrics']['holdout_mean']:.4f}")

    # Best RF
    if best_rf:
        print(f"\nBest RF: depth={best_rf['config']['max_depth']}, min_leaf={best_rf['config']['min_samples_leaf']}, max_feat={best_rf['config']['max_features']}")
        print(f"  Holdout F1: {best_rf['metrics']['holdout_mean']:.4f}")

    # Create summary report
    summary = {
        'experiment': 'aggressive_regularization',
        'dataset': 'diabetes (OpenML 37)',
        'feature_engineering': 'Domain + Bins',
        'date': datetime.now().isoformat(),
        'baseline': 0.665,
        'target': 0.745,
        'overall_best': {
            'model': overall_best['model'] if overall_best else None,
            'config': overall_best['config'] if overall_best else None,
            'holdout_f1': best_holdout,
            'cv_f1': overall_best['metrics']['cv_mean'] if overall_best else None,
            'gap': overall_best['metrics']['gap'] if overall_best else None
        },
        'best_logreg': {
            'config': best_logreg['config'] if best_logreg else None,
            'holdout_f1': best_logreg['metrics']['holdout_mean'] if best_logreg else None
        },
        'best_rf': {
            'config': best_rf['config'] if best_rf else None,
            'holdout_f1': best_rf['metrics']['holdout_mean'] if best_rf else None
        },
        'beat_baseline': best_holdout > 0.665,
        'logreg_results': logreg_results,
        'rf_results': rf_results
    }

    # Save results
    results_path = RESULTS_DIR / 'exp_regularization_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print top 10 leaderboard
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY HOLDOUT F1")
    print("=" * 70)
    for i, r in enumerate(sorted_results[:10], 1):
        m = r['metrics']
        if r['model'] == 'LogReg':
            cfg_str = f"C={r['config']['C']}, penalty={r['config']['penalty']}"
        else:
            cfg_str = f"depth={r['config']['max_depth']}, min_leaf={r['config']['min_samples_leaf']}, max_feat={r['config']['max_features']}"
        print(f"{i:2}. {r['model']:6} | {cfg_str:40} | holdout={m['holdout_mean']:.4f}, cv={m['cv_mean']:.4f}")

    return summary


if __name__ == '__main__':
    summary = run_regularization_experiment()
