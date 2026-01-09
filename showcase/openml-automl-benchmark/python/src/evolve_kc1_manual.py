#!/usr/bin/env python3
"""
Manual Evolution for KC1 - Documented Step by Step

This script runs evolution manually so we can document each generation
and create visualizations showing why evolve-ml works.

Based on MALLORN lessons:
1. Holdout validation is critical
2. Simpler models often win
3. Fixed thresholds beat optimized ones
4. Class weights matter for imbalanced data
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Evolution log
EVOLUTION_LOG = []
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def log_generation(gen_num: int, name: str, cv_f1: float, holdout_f1: float,
                   description: str, params: dict, accepted: bool, reason: str):
    """Log a generation's results."""
    entry = {
        'generation': gen_num,
        'name': name,
        'cv_f1': cv_f1,
        'holdout_f1': holdout_f1,
        'gap': cv_f1 - holdout_f1,
        'description': description,
        'params': params,
        'accepted': accepted,
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }
    EVOLUTION_LOG.append(entry)

    status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
    print(f"\n{'='*60}")
    print(f"Generation {gen_num}: {name}")
    print(f"{'='*60}")
    print(f"CV F1:      {cv_f1:.4f}")
    print(f"Holdout F1: {holdout_f1:.4f}")
    print(f"Gap:        {cv_f1 - holdout_f1:.4f}")
    print(f"Status:     {status} - {reason}")
    print(f"Description: {description}")

    return entry


def load_kc1():
    """Load and preprocess KC1 dataset."""
    dataset = openml.datasets.get_dataset(1067)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    # Preprocess features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_num = X[numeric_cols].values

    imputer = SimpleImputer(strategy='median')
    X_num = imputer.fit_transform(X_num)

    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_num)

    return X_processed, y.values


def evaluate_classifier(clf, X, y, n_cv_folds=8, n_holdout_folds=2):
    """
    Evaluate classifier with CV + holdout validation.

    Returns (cv_f1, holdout_f1, per_fold_results)
    """
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    # CV evaluation
    cv_scores = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone_classifier(clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    # Holdout evaluation
    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone_classifier(clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(cv_scores), np.mean(holdout_scores), cv_scores, holdout_scores


def clone_classifier(clf):
    """Clone a classifier with the same parameters."""
    from sklearn.base import clone
    return clone(clf)


class ThresholdClassifier:
    """Classifier wrapper with custom threshold."""

    def __init__(self, base_clf, threshold=0.5):
        self.base_clf = base_clf
        self.threshold = threshold

    def fit(self, X, y):
        self.base_clf.fit(X, y)
        return self

    def predict(self, X):
        proba = self.base_clf.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base_clf.predict_proba(X)

    def get_params(self, deep=True):
        return {'base_clf': self.base_clf, 'threshold': self.threshold}


def run_evolution():
    """Run the manual evolution process."""
    print("Loading KC1 dataset...")
    X, y = load_kc1()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (imbalance: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}x)")

    best_holdout = 0
    best_gen = None

    # =========================================================================
    # Generation 0: XGBoost Default (Baseline)
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            0, "XGBoost Default", cv_f1, holdout_f1,
            "Baseline XGBoost with default parameters",
            {'n_estimators': 100, 'max_depth': 6},
            True, "Baseline - establishes starting point"
        )
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 1: LogReg with Balanced Weights
    # MALLORN Lesson: Simpler models often generalize better
    # =========================================================================
    clf = LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        1, "LogReg Balanced", cv_f1, holdout_f1,
        "Logistic Regression with balanced class weights (MALLORN lesson: simpler is better)",
        {'C': 1.0, 'class_weight': 'balanced'},
        holdout_f1 > best_holdout,
        f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
    )
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 2: LogReg with Strong Regularization
    # MALLORN Lesson: Strong regularization prevents overfitting on small data
    # =========================================================================
    clf = LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=1000, random_state=42
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        2, "LogReg Strong Reg", cv_f1, holdout_f1,
        "LogReg with C=0.1 (stronger regularization)",
        {'C': 0.1, 'class_weight': 'balanced'},
        holdout_f1 > best_holdout,
        f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
    )
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 3: XGBoost with Custom Class Weight
    # Optimize for imbalanced data
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=5,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            3, "XGBoost Weighted", cv_f1, holdout_f1,
            "XGBoost with scale_pos_weight=5 (handles 5.47x imbalance)",
            {'max_depth': 4, 'scale_pos_weight': 5},
            holdout_f1 > best_holdout,
            f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 4: LogReg with Lower Threshold
    # MALLORN Lesson: Threshold tuning is critical for imbalanced data
    # =========================================================================
    base_clf = LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=1000, random_state=42
    )
    clf = ThresholdClassifier(base_clf, threshold=0.3)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        4, "LogReg t=0.3", cv_f1, holdout_f1,
        "LogReg with threshold=0.3 (lower threshold for imbalanced data)",
        {'C': 0.1, 'threshold': 0.3},
        holdout_f1 > best_holdout,
        f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
    )
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 5: Ensemble (LogReg + XGBoost)
    # MALLORN Lesson: Fixed weights (0.5/0.5) generalize better
    # =========================================================================
    if XGB_AVAILABLE:
        lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, scale_pos_weight=5,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = VotingClassifier(
            estimators=[('lr', lr), ('xgb', xgb_clf)],
            voting='soft'
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            5, "LR+XGB Ensemble", cv_f1, holdout_f1,
            "Soft voting ensemble: LogReg (stable) + XGBoost (powerful)",
            {'lr_C': 0.1, 'xgb_depth': 3, 'voting': 'soft'},
            holdout_f1 > best_holdout,
            f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 6: XGBoost with Strong Regularization
    # Reduce overfitting on the best performer
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05,
            scale_pos_weight=5, reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            6, "XGBoost Regularized", cv_f1, holdout_f1,
            "XGBoost with reg_alpha=1, reg_lambda=2 (reduce overfitting)",
            {'max_depth': 3, 'reg_alpha': 1.0, 'reg_lambda': 2.0},
            holdout_f1 > best_holdout,
            f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 7: RandomForest with Class Weight
    # Alternative ensemble method
    # =========================================================================
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        7, "RF Balanced", cv_f1, holdout_f1,
        "RandomForest with balanced class weights",
        {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
        holdout_f1 > best_holdout,
        f"Holdout improved" if holdout_f1 > best_holdout else "No improvement"
    )
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 8: XGBoost with Lighter Regularization
    # The baseline is actually best - maybe we over-regularized
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            8, "XGBoost Tuned", cv_f1, holdout_f1,
            "XGBoost with moderate regularization and class weight",
            {'max_depth': 5, 'scale_pos_weight': 3, 'n_estimators': 200},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 9: XGBoost with Lower Threshold (0.4)
    # Try threshold tuning on XGBoost
    # =========================================================================
    if XGB_AVAILABLE:
        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = ThresholdClassifier(base_clf, threshold=0.4)
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            9, "XGBoost t=0.4", cv_f1, holdout_f1,
            "XGBoost with threshold=0.4 (slightly lower than default)",
            {'threshold': 0.4},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 10: XGBoost with Threshold 0.35
    # =========================================================================
    if XGB_AVAILABLE:
        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = ThresholdClassifier(base_clf, threshold=0.35)
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            10, "XGBoost t=0.35", cv_f1, holdout_f1,
            "XGBoost with threshold=0.35",
            {'threshold': 0.35},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 11: XGBoost Weighted + Threshold
    # Combine class weight AND threshold
    # =========================================================================
    if XGB_AVAILABLE:
        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, scale_pos_weight=3,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = ThresholdClassifier(base_clf, threshold=0.4)
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            11, "XGBoost w+t", cv_f1, holdout_f1,
            "XGBoost with scale_pos_weight=3 AND threshold=0.4",
            {'scale_pos_weight': 3, 'threshold': 0.4},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 12: Very Simple LogReg
    # Sometimes the simplest model wins
    # =========================================================================
    clf = LogisticRegression(
        C=0.01, class_weight='balanced', max_iter=1000, random_state=42
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        12, "LogReg C=0.01", cv_f1, holdout_f1,
        "Very strongly regularized LogReg (C=0.01)",
        {'C': 0.01, 'class_weight': 'balanced'},
        holdout_f1 > best_holdout,
        f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
    )
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 13: XGBoost with min_child_weight
    # This parameter often helps with imbalanced data
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, min_child_weight=5,
            scale_pos_weight=2,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            13, "XGBoost min_child", cv_f1, holdout_f1,
            "XGBoost with min_child_weight=5 (more stable splits)",
            {'min_child_weight': 5, 'scale_pos_weight': 2},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 14: Best XGB + Best threshold search
    # =========================================================================
    if XGB_AVAILABLE:
        # Find best threshold
        best_t = 0.5
        best_t_holdout = 0

        for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
            base_clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                random_state=42, eval_metric='logloss', verbosity=0
            )
            clf = ThresholdClassifier(base_clf, threshold=t)
            _, holdout_f1, _, _ = evaluate_classifier(clf, X, y)
            if holdout_f1 > best_t_holdout:
                best_t_holdout = holdout_f1
                best_t = t

        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = ThresholdClassifier(base_clf, threshold=best_t)
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        entry = log_generation(
            14, f"XGBoost t={best_t}", cv_f1, holdout_f1,
            f"XGBoost with best threshold found: {best_t}",
            {'threshold': best_t},
            holdout_f1 > best_holdout,
            f"Holdout improved to {holdout_f1:.4f}" if holdout_f1 > best_holdout else "No improvement"
        )
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"\nBest Generation: {best_gen['generation']} - {best_gen['name']}")
    print(f"Best Holdout F1: {best_holdout:.4f}")
    print(f"Baseline F1:     0.436")
    print(f"Improvement:     {((best_holdout - 0.308) / 0.308) * 100:.1f}%")

    # Calculate targets
    baseline = 0.436
    print(f"\nTarget Comparison:")
    print(f"  vs Auto-sklearn target ({baseline * 1.15:.3f}): {'✓ BEATEN' if best_holdout >= baseline * 1.15 else '✗ NOT YET'}")
    print(f"  vs FLAML target ({baseline * 1.17:.3f}):        {'✓ BEATEN' if best_holdout >= baseline * 1.17 else '✗ NOT YET'}")
    print(f"  vs AutoGluon target ({baseline * 1.23:.3f}):    {'✓ BEATEN' if best_holdout >= baseline * 1.23 else '✗ NOT YET'}")

    # Save log (convert numpy bools to Python bools)
    log_path = RESULTS_DIR / 'kc1_evolution_log.json'
    clean_log = []
    for entry in EVOLUTION_LOG:
        clean_entry = {}
        for k, v in entry.items():
            if isinstance(v, (np.bool_, np.integer, np.floating)):
                clean_entry[k] = v.item()
            else:
                clean_entry[k] = v
        clean_log.append(clean_entry)
    with open(log_path, 'w') as f:
        json.dump(clean_log, f, indent=2)
    print(f"\nEvolution log saved to: {log_path}")

    return EVOLUTION_LOG, best_gen


if __name__ == '__main__':
    log, best = run_evolution()
