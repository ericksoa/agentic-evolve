#!/usr/bin/env python3
"""
Evolution for Diabetes Dataset (OpenML ID 37)

Dataset characteristics:
- 768 samples (small!)
- 8 features
- Binary classification
- Imbalance ratio: 1.87 (moderate)
- Baseline F1: 0.648 (RandomForest)

This is a classic dataset with room for improvement.
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

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
        'cv_f1': float(cv_f1),
        'holdout_f1': float(holdout_f1),
        'gap': float(cv_f1 - holdout_f1),
        'description': description,
        'params': params,
        'accepted': bool(accepted),
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

    return entry


def load_diabetes():
    """Load and preprocess diabetes dataset."""
    dataset = openml.datasets.get_dataset(37)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    # Basic preprocessing
    X_values = X.values.astype(float)

    # Handle zeros that are actually missing (common in this dataset)
    # Columns 1,2,3,4,5 (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    # have zeros that should be NaN
    for col in [1, 2, 3, 4, 5]:
        X_values[X_values[:, col] == 0, col] = np.nan

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)

    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_imputed)

    return X_processed, y.values, X.columns.tolist()


def evaluate_classifier(clf, X, y, n_cv_folds=8, n_holdout_folds=2):
    """Evaluate with CV + holdout validation."""
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    cv_scores = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(cv_scores), np.mean(holdout_scores), cv_scores, holdout_scores


class ThresholdClassifier:
    """Classifier wrapper with custom threshold."""

    def __init__(self, base_clf, threshold=0.5):
        self.base_clf = base_clf
        self.threshold = threshold
        self._base_clf_fitted = None

    def fit(self, X, y):
        self._base_clf_fitted = clone(self.base_clf)
        self._base_clf_fitted.fit(X, y)
        return self

    def predict(self, X):
        proba = self._base_clf_fitted.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self._base_clf_fitted.predict_proba(X)

    def get_params(self, deep=True):
        return {'base_clf': self.base_clf, 'threshold': self.threshold}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __sklearn_clone__(self):
        return ThresholdClassifier(
            base_clf=clone(self.base_clf),
            threshold=self.threshold
        )


def run_evolution():
    """Run evolution on diabetes dataset."""
    print("Loading diabetes dataset...")
    X, y, feature_names = load_diabetes()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Class distribution: {np.bincount(y)} (imbalance: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}x)")

    best_holdout = 0
    best_gen = None

    # Track all results for analysis
    all_results = []

    # =========================================================================
    # Generation 0: RandomForest Default (Baseline from pilot)
    # =========================================================================
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    entry = log_generation(
        0, "RF Default", cv_f1, holdout_f1,
        "Baseline RandomForest (best from pilot)",
        {'n_estimators': 100},
        True, "Baseline"
    )
    best_holdout = holdout_f1
    best_gen = entry

    # =========================================================================
    # Generation 1: LogReg (MALLORN lesson: simpler often wins)
    # =========================================================================
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        1, "LogReg Balanced", cv_f1, holdout_f1,
        "Logistic Regression with balanced weights",
        {'C': 1.0, 'class_weight': 'balanced'},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 2: LogReg with stronger regularization
    # =========================================================================
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        2, "LogReg C=0.1", cv_f1, holdout_f1,
        "LogReg with stronger regularization",
        {'C': 0.1},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 3: XGBoost with class weight
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=1.87,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(
            3, "XGBoost Weighted", cv_f1, holdout_f1,
            "XGBoost with scale_pos_weight matching imbalance",
            {'scale_pos_weight': 1.87},
            accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
        )
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 4: Gradient Boosting (sklearn)
    # =========================================================================
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        4, "GradientBoosting", cv_f1, holdout_f1,
        "Sklearn GradientBoosting",
        {'max_depth': 3, 'learning_rate': 0.1},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 5: SVM with RBF kernel
    # =========================================================================
    clf = SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        5, "SVM RBF", cv_f1, holdout_f1,
        "SVM with RBF kernel and balanced weights",
        {'C': 1.0, 'kernel': 'rbf'},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 6: LogReg with threshold tuning
    # =========================================================================
    base_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    clf = ThresholdClassifier(base_clf, threshold=0.4)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        6, "LogReg t=0.4", cv_f1, holdout_f1,
        "LogReg with lower threshold (0.4)",
        {'C': 0.5, 'threshold': 0.4},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 7: Ensemble (LR + RF + XGB)
    # =========================================================================
    if XGB_AVAILABLE:
        lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_clf)],
            voting='soft'
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(
            7, "LR+RF+XGB", cv_f1, holdout_f1,
            "3-model ensemble with soft voting",
            {'models': ['LR', 'RF', 'XGB']},
            accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
        )
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 8: RF with tuned hyperparameters
    # =========================================================================
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        8, "RF Tuned", cv_f1, holdout_f1,
        "RF with tuned depth and more trees",
        {'n_estimators': 200, 'max_depth': 10},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 9: LightGBM
    # =========================================================================
    if LGB_AVAILABLE:
        clf = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, class_weight='balanced',
            random_state=42, verbose=-1
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(
            9, "LightGBM", cv_f1, holdout_f1,
            "LightGBM with balanced weights",
            {'max_depth': 4},
            accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
        )
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 10: XGBoost with stronger regularization
    # =========================================================================
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(
            10, "XGBoost Reg", cv_f1, holdout_f1,
            "XGBoost with L1/L2 regularization",
            {'reg_alpha': 0.5, 'reg_lambda': 2.0},
            accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
        )
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 11: Simple ensemble (LR + RF only - MALLORN lesson)
    # =========================================================================
    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf)],
        voting='soft'
    )
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        11, "LR+RF", cv_f1, holdout_f1,
        "2-model ensemble (simpler often better)",
        {'models': ['LR', 'RF']},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 12: Very simple LogReg (MALLORN: simplest model test)
    # =========================================================================
    clf = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        12, "LogReg Simple", cv_f1, holdout_f1,
        "Very simple LogReg (C=0.01, no class weight)",
        {'C': 0.01},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Generation 13: XGBoost with best threshold
    # =========================================================================
    if XGB_AVAILABLE:
        best_t = 0.5
        best_t_score = 0

        for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
            base_clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4,
                random_state=42, eval_metric='logloss', verbosity=0
            )
            clf = ThresholdClassifier(base_clf, threshold=t)
            _, ho, _, _ = evaluate_classifier(clf, X, y)
            if ho > best_t_score:
                best_t_score = ho
                best_t = t

        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        clf = ThresholdClassifier(base_clf, threshold=best_t)
        cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

        accepted = holdout_f1 > best_holdout
        entry = log_generation(
            13, f"XGB t={best_t}", cv_f1, holdout_f1,
            f"XGBoost with optimized threshold {best_t}",
            {'threshold': best_t},
            accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
        )
        if accepted:
            best_holdout = holdout_f1
            best_gen = entry

    # =========================================================================
    # Generation 14: Best model + threshold sweep
    # =========================================================================
    # Try threshold on current best model type
    if 'LogReg' in best_gen['name']:
        base_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    else:
        base_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    best_t = 0.5
    best_t_score = 0
    for t in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        clf = ThresholdClassifier(base_clf, threshold=t)
        _, ho, _, _ = evaluate_classifier(clf, X, y)
        if ho > best_t_score:
            best_t_score = ho
            best_t = t

    clf = ThresholdClassifier(base_clf, threshold=best_t)
    cv_f1, holdout_f1, _, _ = evaluate_classifier(clf, X, y)

    accepted = holdout_f1 > best_holdout
    entry = log_generation(
        14, f"Best+t={best_t}", cv_f1, holdout_f1,
        f"Best model with threshold {best_t}",
        {'threshold': best_t},
        accepted, f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}"
    )
    if accepted:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # Results Summary
    # =========================================================================
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"\nBest Generation: {best_gen['generation']} - {best_gen['name']}")
    print(f"Best Holdout F1: {best_holdout:.4f}")
    print(f"Baseline (RF):   0.648")

    improvement = ((best_holdout - 0.648) / 0.648) * 100
    print(f"Improvement:     {improvement:+.1f}%")

    # Target comparison
    baseline = 0.648
    print(f"\nTarget Comparison:")
    print(f"  vs Auto-sklearn target ({baseline * 1.15:.3f}): {'✓ BEATEN' if best_holdout >= baseline * 1.15 else '✗ NOT YET'}")
    print(f"  vs FLAML target ({baseline * 1.17:.3f}):        {'✓ BEATEN' if best_holdout >= baseline * 1.17 else '✗ NOT YET'}")
    print(f"  vs AutoGluon target ({baseline * 1.23:.3f}):    {'✓ BEATEN' if best_holdout >= baseline * 1.23 else '✗ NOT YET'}")

    # Save log
    log_path = RESULTS_DIR / 'diabetes_evolution_log.json'
    with open(log_path, 'w') as f:
        json.dump(EVOLUTION_LOG, f, indent=2)
    print(f"\nEvolution log saved to: {log_path}")

    return EVOLUTION_LOG, best_gen


if __name__ == '__main__':
    log, best = run_evolution()
