#!/usr/bin/env python3
"""
MLP (Neural Network) Experiment for Diabetes Classification

This experiment tests sklearn's MLPClassifier with various architectures
and hyperparameters to explore whether neural networks can improve on
the current best diabetes classification performance.

Configurations tested:
- Hidden layers: (16,), (32,), (64,), (32,16), (64,32), (128,64)
- Activation: relu, tanh
- Regularization (alpha): 0.0001, 0.001, 0.01, 0.1
- With and without Domain+Bins feature engineering

Key features:
- Early stopping to prevent overfitting
- 8-fold CV + 2-fold holdout validation (matching other experiments)

Current best: 0.665 holdout F1 (Domain + Bins + LogReg)
Target: 0.745 (Auto-sklearn level)
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.neural_network import MLPClassifier
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

# Configuration
HIDDEN_LAYER_SIZES = [
    (16,),
    (32,),
    (64,),
    (32, 16),
    (64, 32),
    (128, 64),
]
ACTIVATIONS = ['relu', 'tanh']
ALPHAS = [0.0001, 0.001, 0.01, 0.1]  # L2 regularization strength
FEATURE_CONFIGS = ['none', 'domain', 'bins', 'domain+bins']


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for diabetes dataset.

    Supports:
    - Domain features: BMI categories, age groups, glucose thresholds, risk flags
    - Bin features: Quantile-based discretization of continuous variables
    """

    def __init__(self, add_domain=False, add_bins=False):
        self.add_domain = add_domain
        self.add_bins = add_bins

    def fit(self, X, y=None):
        if self.add_bins:
            # Compute quantiles for binning during fit
            df = pd.DataFrame(X, columns=['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'])
            self.quantiles_ = {}
            for col in ['plas', 'mass', 'age', 'pedi']:
                self.quantiles_[col] = {
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
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

        if self.add_bins:
            # Quantile-based bins for continuous variables
            bins = pd.DataFrame()

            for col in ['plas', 'mass', 'age', 'pedi']:
                # Create binary indicators for low and high quartiles
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


def get_feature_engineer(config_name):
    """Create feature engineer based on config name."""
    if config_name == 'none':
        return DiabetesFeatureEngineer(add_domain=False, add_bins=False)
    elif config_name == 'domain':
        return DiabetesFeatureEngineer(add_domain=True, add_bins=False)
    elif config_name == 'bins':
        return DiabetesFeatureEngineer(add_domain=False, add_bins=True)
    elif config_name == 'domain+bins':
        return DiabetesFeatureEngineer(add_domain=True, add_bins=True)
    else:
        raise ValueError(f"Unknown feature config: {config_name}")


def run_mlp_experiment():
    """Run the comprehensive MLP experiment."""
    print("=" * 70)
    print("MLP (NEURAL NETWORK) EXPERIMENT FOR DIABETES CLASSIFICATION")
    print("=" * 70)
    print(f"Current best: 0.665 holdout F1 (Domain + Bins + LogReg)")
    print(f"Target: 0.745 (Auto-sklearn level)")
    print()

    # Load data
    X, y = load_diabetes_raw()

    # Results storage
    all_results = []
    best_result = None
    best_holdout = 0.0

    # Calculate total experiments
    total_experiments = (
        len(HIDDEN_LAYER_SIZES) * len(ACTIVATIONS) * len(ALPHAS) * len(FEATURE_CONFIGS)
    )
    current_exp = 0

    print(f"\nRunning {total_experiments} experiments...")
    print(f"Hidden layers: {HIDDEN_LAYER_SIZES}")
    print(f"Activations: {ACTIVATIONS}")
    print(f"Alphas (regularization): {ALPHAS}")
    print(f"Feature configs: {FEATURE_CONFIGS}")
    print("-" * 70)

    # Run all combinations
    for hidden_layers, activation, alpha, feature_config in product(
        HIDDEN_LAYER_SIZES, ACTIVATIONS, ALPHAS, FEATURE_CONFIGS
    ):
        current_exp += 1

        # Create config description
        layers_str = 'x'.join(map(str, hidden_layers))
        config_name = f"layers={layers_str}, act={activation}, alpha={alpha}, features={feature_config}"
        print(f"\n[{current_exp}/{total_experiments}] {config_name}")

        try:
            # Create MLP classifier with early stopping
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                alpha=alpha,
                solver='adam',
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
                verbose=False
            )

            # Create feature transformer
            feature_transformer = get_feature_engineer(feature_config)

            # Create and evaluate pipeline
            pipeline = create_pipeline(feature_transformer, mlp)
            metrics = evaluate_pipeline(pipeline, X, y)

            # Store result
            result = {
                'config': {
                    'hidden_layer_sizes': hidden_layers,
                    'activation': activation,
                    'alpha': alpha,
                    'feature_config': feature_config
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
                    'hidden_layer_sizes': hidden_layers,
                    'activation': activation,
                    'alpha': alpha,
                    'feature_config': feature_config
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
        layers_str = 'x'.join(map(str, best_result['config']['hidden_layer_sizes']))
        print(f"  Hidden layers:   {layers_str}")
        print(f"  Activation:      {best_result['config']['activation']}")
        print(f"  Alpha (L2 reg):  {best_result['config']['alpha']}")
        print(f"  Features:        {best_result['config']['feature_config']}")
        print(f"\nBest Metrics:")
        print(f"  CV F1:           {best_result['metrics']['cv_mean']:.4f}")
        print(f"  Holdout F1:      {best_result['metrics']['holdout_mean']:.4f}")
        print(f"  Gap:             {best_result['metrics']['gap']:.4f}")

        # Compare to baselines
        baseline = 0.665
        target = 0.745
        print(f"\nComparison:")
        if best_holdout > baseline:
            print(f"  vs Baseline (0.665): +{(best_holdout - baseline)*100:.2f}% IMPROVEMENT")
        else:
            print(f"  vs Baseline (0.665): -{(baseline - best_holdout)*100:.2f}% below")

        if best_holdout > target:
            print(f"  vs Target (0.745):   +{(best_holdout - target)*100:.2f}% EXCEEDED")
        else:
            print(f"  vs Target (0.745):   -{(target - best_holdout)*100:.2f}% below")

        print(f"  Beat baseline: {'YES' if best_holdout > baseline else 'NO'}")

    # Create summary report
    summary = {
        'experiment': 'mlp_neural_network',
        'dataset': 'diabetes (OpenML 37)',
        'date': datetime.now().isoformat(),
        'baseline': 0.665,
        'target': 0.745,
        'best_config': best_result['config'] if best_result else None,
        'best_holdout_f1': best_holdout,
        'best_cv_f1': best_result['metrics']['cv_mean'] if best_result else None,
        'best_gap': best_result['metrics']['gap'] if best_result else None,
        'beat_baseline': best_holdout > 0.665,
        'total_experiments': total_experiments,
        'all_results': all_results
    }

    # Save results
    results_path = RESULTS_DIR / 'exp_mlp_results.json'
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
        layers_str = 'x'.join(map(str, cfg['hidden_layer_sizes']))
        print(f"{i:2}. layers={layers_str:10}, {cfg['activation']:4}, alpha={cfg['alpha']:.4f}, "
              f"feat={cfg['feature_config']:11} | holdout={m['holdout_mean']:.4f}, cv={m['cv_mean']:.4f}")

    # Analyze results by feature configuration
    print("\n" + "=" * 70)
    print("ANALYSIS BY FEATURE CONFIGURATION")
    print("=" * 70)
    for feat_config in FEATURE_CONFIGS:
        feat_results = [r for r in valid_results if r['config']['feature_config'] == feat_config]
        if feat_results:
            holdouts = [r['metrics']['holdout_mean'] for r in feat_results]
            print(f"{feat_config:12}: mean={np.mean(holdouts):.4f}, max={np.max(holdouts):.4f}, min={np.min(holdouts):.4f}")

    # Analyze results by architecture
    print("\n" + "=" * 70)
    print("ANALYSIS BY ARCHITECTURE")
    print("=" * 70)
    for hidden in HIDDEN_LAYER_SIZES:
        arch_results = [r for r in valid_results if r['config']['hidden_layer_sizes'] == hidden]
        if arch_results:
            holdouts = [r['metrics']['holdout_mean'] for r in arch_results]
            layers_str = 'x'.join(map(str, hidden))
            print(f"{layers_str:10}: mean={np.mean(holdouts):.4f}, max={np.max(holdouts):.4f}, min={np.min(holdouts):.4f}")

    return summary, best_result


if __name__ == '__main__':
    summary, best = run_mlp_experiment()
