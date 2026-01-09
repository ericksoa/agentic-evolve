#!/usr/bin/env python3
"""
Binning Strategy Experiment for Diabetes Classification

This experiment tests different discretization (binning) strategies to find
the optimal configuration for diabetes classification.

Strategies tested:
- Number of bins: 3, 4, 5, 6, 8, 10
- Binning strategy: quantile (equal-frequency), uniform (equal-width), kmeans
- With and without domain features
- Classifiers: LogisticRegression, RandomForest

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
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
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

# Configuration
BIN_COUNTS = [3, 4, 5, 6, 8, 10]
BIN_STRATEGIES = ['quantile', 'uniform', 'kmeans']
USE_DOMAIN_OPTIONS = [False, True]
CLASSIFIERS = {
    'LogReg': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
}

# Columns to bin (continuous features)
COLS_TO_BIN = ['plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']


class DiabetesBinningTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies binning to diabetes features.

    Parameters
    ----------
    n_bins : int
        Number of bins to use
    strategy : str
        Binning strategy: 'quantile', 'uniform', or 'kmeans'
    add_domain : bool
        Whether to add domain-specific categorical features
    cols_to_bin : list
        List of column names to bin
    """

    def __init__(self, n_bins=5, strategy='quantile', add_domain=False, cols_to_bin=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.add_domain = add_domain
        self.cols_to_bin = cols_to_bin or COLS_TO_BIN

    def fit(self, X, y=None):
        # X columns: preg, plas, pres, skin, insu, mass, pedi, age
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        # Fit binners for each column
        self.binners_ = {}
        for col in self.cols_to_bin:
            binner = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense',
                                       strategy=self.strategy, random_state=42)
            binner.fit(df[[col]])
            self.binners_[col] = binner

        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])

        features = [df.copy()]

        # Add binned features
        for col in self.cols_to_bin:
            binned = self.binners_[col].transform(df[[col]])
            bin_df = pd.DataFrame(
                binned,
                columns=[f'{col}_bin{i}' for i in range(binned.shape[1])]
            )
            features.append(bin_df)

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

        result = pd.concat(features, axis=1)
        return result.values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']

        for col in self.cols_to_bin:
            names += [f'{col}_bin{i}' for i in range(self.n_bins)]

        if self.add_domain:
            names += ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese',
                     'age_young', 'age_middle', 'age_senior',
                     'glucose_normal', 'glucose_prediabetic', 'glucose_diabetic',
                     'high_risk', 'preg_risk']

        return names


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


def run_binning_experiment():
    """Run the comprehensive binning experiment."""
    print("=" * 70)
    print("BINNING STRATEGY EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)
    print(f"Baseline to beat: 0.665 holdout F1")
    print(f"Target: 0.745 (Auto-sklearn level)")
    print()

    # Load data
    X, y = load_diabetes_raw()

    # Results storage
    all_results = []
    best_result = None
    best_holdout = 0.0

    # Calculate total experiments
    total_experiments = len(BIN_COUNTS) * len(BIN_STRATEGIES) * len(USE_DOMAIN_OPTIONS) * len(CLASSIFIERS)
    current_exp = 0

    print(f"\nRunning {total_experiments} experiments...")
    print("-" * 70)

    # Run all combinations
    for n_bins, strategy, use_domain, (clf_name, clf_base) in product(
        BIN_COUNTS, BIN_STRATEGIES, USE_DOMAIN_OPTIONS, CLASSIFIERS.items()
    ):
        current_exp += 1

        # Create config description
        config_name = f"bins={n_bins}, strategy={strategy}, domain={use_domain}, clf={clf_name}"
        print(f"\n[{current_exp}/{total_experiments}] {config_name}")

        try:
            # Create transformer and pipeline
            transformer = DiabetesBinningTransformer(
                n_bins=n_bins,
                strategy=strategy,
                add_domain=use_domain
            )

            pipeline = create_pipeline(transformer, clone(clf_base))

            # Evaluate
            metrics = evaluate_pipeline(pipeline, X, y)

            # Store result
            result = {
                'config': {
                    'n_bins': n_bins,
                    'strategy': strategy,
                    'use_domain': use_domain,
                    'classifier': clf_name
                },
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)

            # Print summary
            print(f"  CV F1:      {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            print(f"  Holdout F1: {metrics['holdout_mean']:.4f} (+/- {metrics['holdout_std']:.4f})")
            print(f"  Gap:        {metrics['gap']:.4f}")

            # Track best
            if metrics['holdout_mean'] > best_holdout:
                best_holdout = metrics['holdout_mean']
                best_result = result
                print(f"  ** NEW BEST **")

        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                'config': {
                    'n_bins': n_bins,
                    'strategy': strategy,
                    'use_domain': use_domain,
                    'classifier': clf_name
                },
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if best_result:
        print(f"\nBest Configuration:")
        print(f"  Number of bins: {best_result['config']['n_bins']}")
        print(f"  Strategy:       {best_result['config']['strategy']}")
        print(f"  Domain features:{best_result['config']['use_domain']}")
        print(f"  Classifier:     {best_result['config']['classifier']}")
        print(f"\nBest Metrics:")
        print(f"  CV F1:          {best_result['metrics']['cv_mean']:.4f}")
        print(f"  Holdout F1:     {best_result['metrics']['holdout_mean']:.4f}")
        print(f"  Gap:            {best_result['metrics']['gap']:.4f}")

        # Compare to baselines
        baseline = 0.665
        target = 0.745
        print(f"\nComparison:")
        print(f"  vs Baseline (0.665): {'+' if best_holdout > baseline else '-'}{abs(best_holdout - baseline)*100:.2f}%")
        print(f"  vs Target (0.745):   {'+' if best_holdout > target else '-'}{abs(best_holdout - target)*100:.2f}%")
        print(f"  Beat baseline: {'YES' if best_holdout > baseline else 'NO'}")

    # Create summary report
    summary = {
        'experiment': 'binning_strategy',
        'dataset': 'diabetes (OpenML 37)',
        'date': datetime.now().isoformat(),
        'baseline': 0.665,
        'target': 0.745,
        'best_config': best_result['config'] if best_result else None,
        'best_holdout_f1': best_holdout,
        'best_cv_f1': best_result['metrics']['cv_mean'] if best_result else None,
        'beat_baseline': best_holdout > 0.665,
        'all_results': all_results
    }

    # Save results
    results_path = RESULTS_DIR / 'exp_binning_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Also create a sorted leaderboard
    valid_results = [r for r in all_results if 'metrics' in r]
    sorted_results = sorted(valid_results, key=lambda x: x['metrics']['holdout_mean'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY HOLDOUT F1")
    print("=" * 70)
    for i, r in enumerate(sorted_results[:10], 1):
        cfg = r['config']
        m = r['metrics']
        print(f"{i:2}. bins={cfg['n_bins']:2}, {cfg['strategy']:8}, domain={str(cfg['use_domain']):5}, "
              f"{cfg['classifier']:6} | holdout={m['holdout_mean']:.4f}, cv={m['cv_mean']:.4f}")

    return summary, best_result


if __name__ == '__main__':
    summary, best = run_binning_experiment()
