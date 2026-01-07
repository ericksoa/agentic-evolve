"""
Baseline algorithms for comparison.

These baselines establish performance benchmarks that the evolved
algorithms should beat.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple


def create_baseline_pipelines() -> Dict[str, Pipeline]:
    """
    Create baseline ML pipelines for comparison.

    Returns:
        Dictionary of pipeline name -> sklearn Pipeline
    """
    baselines = {}

    # Logistic Regression (simple baseline)
    baselines['logistic_regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

    # Random Forest
    baselines['random_forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting
    baselines['gradient_boosting'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    # XGBoost
    baselines['xgboost'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=10,  # Approximate class imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])

    # LightGBM
    baselines['lightgbm'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ))
    ])

    return baselines


def evaluate_baseline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Evaluate a baseline pipeline using cross-validation.

    Args:
        pipeline: sklearn Pipeline to evaluate
        X: Feature DataFrame
        y: Target labels
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with evaluation metrics
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Define scorers
    scorers = {
        'f1': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0)
    }

    results = {}
    for metric_name, scorer in scorers.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer)
        results[f'{metric_name}_mean'] = scores.mean()
        results[f'{metric_name}_std'] = scores.std()

    return results


def run_all_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all baseline algorithms and compare performance.

    Args:
        X: Feature DataFrame
        y: Target labels
        cv_folds: Number of cross-validation folds
        verbose: Print progress

    Returns:
        DataFrame with baseline results
    """
    baselines = create_baseline_pipelines()
    results = []

    for name, pipeline in baselines.items():
        if verbose:
            print(f"Evaluating {name}...")

        try:
            metrics = evaluate_baseline(pipeline, X, y, cv_folds)
            metrics['algorithm'] = name
            results.append(metrics)

            if verbose:
                print(f"  F1: {metrics['f1_mean']:.4f} (+/- {metrics['f1_std']:.4f})")

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    return pd.DataFrame(results).set_index('algorithm')


# =============================================================================
# SIMPLE HEURISTIC BASELINES
# =============================================================================

def tde_heuristic_classifier(features: pd.DataFrame) -> np.ndarray:
    """
    Simple heuristic-based TDE classifier.

    TDEs typically have:
    - Strong blue colors (high UV flux)
    - Characteristic decay timescale (~weeks to months)
    - Smooth light curves (low variance in residuals)
    - Power-law decay close to -5/3

    Args:
        features: Feature DataFrame

    Returns:
        Binary predictions
    """
    predictions = np.zeros(len(features))

    # Score based on TDE characteristics
    scores = np.zeros(len(features))

    # Blue color (g-r should be negative for blue objects)
    if 'g_r_color' in features.columns:
        gr_color = features['g_r_color'].fillna(0)
        scores += (gr_color < -0.5).astype(float) * 0.3

    # Decay alpha close to -5/3 = -1.67
    for band in ['g', 'r', 'i']:
        col = f'{band}_decay_alpha'
        if col in features.columns:
            alpha = features[col].fillna(0)
            alpha_diff = np.abs(alpha - (-5/3))
            scores += (alpha_diff < 0.3).astype(float) * 0.2

    # High amplitude (TDEs are bright)
    if 'global_amplitude' in features.columns:
        amp = features['global_amplitude'].fillna(0)
        amp_threshold = amp.quantile(0.7)
        scores += (amp > amp_threshold).astype(float) * 0.2

    # Predict TDE if score > 0.5
    predictions = (scores > 0.5).astype(int)

    return predictions


def random_baseline(y: pd.Series, seed: int = 42) -> np.ndarray:
    """
    Random baseline that predicts based on class frequencies.

    Args:
        y: True labels (used to estimate class frequencies)
        seed: Random seed

    Returns:
        Random predictions matching class distribution
    """
    np.random.seed(seed)
    p_positive = y.mean()
    return (np.random.random(len(y)) < p_positive).astype(int)


def all_positive_baseline(n_samples: int) -> np.ndarray:
    """Predict all samples as TDE."""
    return np.ones(n_samples, dtype=int)


def all_negative_baseline(n_samples: int) -> np.ndarray:
    """Predict all samples as non-TDE."""
    return np.zeros(n_samples, dtype=int)
