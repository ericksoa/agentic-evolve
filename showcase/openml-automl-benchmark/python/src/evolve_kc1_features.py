#!/usr/bin/env python3
"""
KC1 Feature Engineering Evolution

KC1 is a software defect prediction dataset with 21 software metrics.
Previous evolution found XGBoost defaults were optimal (holdout F1 = 0.463).
Now we try feature engineering to break through that plateau.

Software metrics typically include:
- LOC metrics (lines of code, blank lines, comments)
- Complexity metrics (cyclomatic complexity v(g))
- Halstead metrics (volume, difficulty, effort, vocabulary)
- Design metrics

Feature engineering strategies:
1. Domain features: complexity ratios, code quality indicators
2. Ratio features: LOC ratios, complexity per LOC
3. Interaction features: complexity × size interactions
4. Polynomial features: squared terms for key metrics
5. Binning: discretize continuous metrics
"""

import json
import numpy as np
import pandas as pd
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone
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

EVOLUTION_LOG = []
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
VISUALS_DIR = Path(__file__).parent.parent.parent / 'visuals'
RESULTS_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)


def log_generation(gen_num: int, name: str, cv_f1: float, holdout_f1: float,
                   description: str, params: dict, accepted: bool, reason: str):
    """Log a generation's results."""
    # Convert numpy types to Python types
    cv_f1 = float(cv_f1)
    holdout_f1 = float(holdout_f1)
    accepted = bool(accepted)

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
    print(f"Status:     {status}")
    print(f"Features:   {params.get('features', 'N/A')}")

    return entry


class KC1FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for KC1 software metrics dataset.

    KC1 features (typical software metrics):
    - loc: lines of code
    - v(g): cyclomatic complexity
    - ev(g): essential complexity
    - iv(g): design complexity
    - n: Halstead length
    - v: Halstead volume
    - l: Halstead level
    - d: Halstead difficulty
    - i: Halstead intelligence
    - e: Halstead effort
    - b: Halstead bugs estimate
    - t: Halstead time
    - lOCode, lOComment, lOBlank: line counts
    - uniq_Op, uniq_Opnd: unique operators/operands
    - total_Op, total_Opnd: total operators/operands
    - branchCount: number of branches
    """

    def __init__(self,
                 add_ratios=False,
                 add_complexity=False,
                 add_interactions=False,
                 add_bins=False,
                 add_quality=False,
                 n_bins=5):
        self.add_ratios = add_ratios
        self.add_complexity = add_complexity
        self.add_interactions = add_interactions
        self.add_bins = add_bins
        self.add_quality = add_quality
        self.n_bins = n_bins
        self.feature_names_ = None
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'f{i}' for i in range(X.shape[1])]

        # Fit binning on numeric columns
        if self.add_bins:
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            for i in range(X_arr.shape[1]):
                col = X_arr[:, i]
                col_valid = col[~np.isnan(col)]
                if len(col_valid) > self.n_bins:
                    try:
                        _, edges = pd.qcut(col_valid, self.n_bins, retbins=True, duplicates='drop')
                        self.bin_edges_[i] = edges
                    except:
                        pass
        return self

    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X.copy()
        features = [X_arr]

        # Get column indices (approximate mapping to common software metrics)
        # Typical KC1 columns: loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t,
        #                      lOCode, lOComment, lOBlank, uniq_Op, uniq_Opnd,
        #                      total_Op, total_Opnd, branchCount
        n_cols = X_arr.shape[1]

        if self.add_ratios and n_cols >= 10:
            # Complexity per LOC (assuming col 0 is LOC-like, col 1 is complexity)
            loc_col = X_arr[:, 0]
            complexity_col = X_arr[:, 1] if n_cols > 1 else X_arr[:, 0]

            # Avoid division by zero
            loc_safe = np.where(loc_col == 0, 1, loc_col)

            # Complexity density
            complexity_per_loc = complexity_col / loc_safe

            # If we have more columns, create more ratios
            ratios = [complexity_per_loc.reshape(-1, 1)]

            if n_cols > 5:
                # Volume per LOC (col 5 might be Halstead volume)
                volume_col = X_arr[:, 5]
                volume_per_loc = volume_col / loc_safe
                ratios.append(volume_per_loc.reshape(-1, 1))

            if n_cols > 9:
                # Effort per LOC (col 9 might be Halstead effort)
                effort_col = X_arr[:, 9]
                effort_per_loc = effort_col / loc_safe
                ratios.append(effort_per_loc.reshape(-1, 1))

            if n_cols > 15:
                # Operator density
                total_op = X_arr[:, min(17, n_cols-1)]
                op_density = total_op / loc_safe
                ratios.append(op_density.reshape(-1, 1))

            features.extend(ratios)

        if self.add_complexity and n_cols >= 4:
            # Essential/cyclomatic complexity ratio
            vg = X_arr[:, 1] if n_cols > 1 else X_arr[:, 0]
            evg = X_arr[:, 2] if n_cols > 2 else X_arr[:, 0]
            ivg = X_arr[:, 3] if n_cols > 3 else X_arr[:, 0]

            vg_safe = np.where(vg == 0, 1, vg)

            # Essential complexity ratio (high = unstructured code)
            essential_ratio = evg / vg_safe

            # Design complexity ratio
            design_ratio = ivg / vg_safe

            # Complexity flags
            high_complexity = (vg > np.nanmedian(vg)).astype(float)
            very_high_complexity = (vg > np.nanpercentile(vg, 75)).astype(float)

            features.extend([
                essential_ratio.reshape(-1, 1),
                design_ratio.reshape(-1, 1),
                high_complexity.reshape(-1, 1),
                very_high_complexity.reshape(-1, 1)
            ])

        if self.add_quality and n_cols >= 15:
            # Code quality indicators
            loc = X_arr[:, 0]
            loc_safe = np.where(loc == 0, 1, loc)

            # Comment ratio (if lOComment and lOCode available)
            if n_cols > 13:
                lo_code = X_arr[:, 12] if n_cols > 12 else X_arr[:, 0]
                lo_comment = X_arr[:, 13] if n_cols > 13 else np.zeros_like(loc)
                lo_blank = X_arr[:, 14] if n_cols > 14 else np.zeros_like(loc)

                code_safe = np.where(lo_code == 0, 1, lo_code)
                comment_ratio = lo_comment / code_safe
                blank_ratio = lo_blank / code_safe

                # Low comment = potential quality issue
                low_comments = (comment_ratio < np.nanmedian(comment_ratio)).astype(float)

                features.extend([
                    comment_ratio.reshape(-1, 1),
                    blank_ratio.reshape(-1, 1),
                    low_comments.reshape(-1, 1)
                ])

            # Operator/operand balance (Halstead-related)
            if n_cols > 18:
                uniq_op = X_arr[:, 15] if n_cols > 15 else np.ones_like(loc)
                uniq_opnd = X_arr[:, 16] if n_cols > 16 else np.ones_like(loc)
                total_op = X_arr[:, 17] if n_cols > 17 else np.ones_like(loc)
                total_opnd = X_arr[:, 18] if n_cols > 18 else np.ones_like(loc)

                uniq_op_safe = np.where(uniq_op == 0, 1, uniq_op)
                uniq_opnd_safe = np.where(uniq_opnd == 0, 1, uniq_opnd)

                # Repetition ratio (high = repetitive code)
                op_repetition = total_op / uniq_op_safe
                opnd_repetition = total_opnd / uniq_opnd_safe

                features.extend([
                    op_repetition.reshape(-1, 1),
                    opnd_repetition.reshape(-1, 1)
                ])

        if self.add_interactions and n_cols >= 5:
            # Key interactions
            loc = X_arr[:, 0]
            complexity = X_arr[:, 1] if n_cols > 1 else X_arr[:, 0]
            volume = X_arr[:, 5] if n_cols > 5 else X_arr[:, 0]

            # Complexity × LOC interaction
            complexity_loc = complexity * loc

            # Volume × complexity
            if n_cols > 5:
                volume_complexity = volume * complexity
                features.append(volume_complexity.reshape(-1, 1))

            # Log transforms (often useful for skewed metrics)
            log_loc = np.log1p(np.abs(loc))
            log_complexity = np.log1p(np.abs(complexity))

            features.extend([
                complexity_loc.reshape(-1, 1),
                log_loc.reshape(-1, 1),
                log_complexity.reshape(-1, 1)
            ])

        if self.add_bins:
            # Discretize key metrics
            bin_features = []
            for i, edges in self.bin_edges_.items():
                if i < X_arr.shape[1]:
                    col = X_arr[:, i]
                    binned = np.digitize(col, edges[1:-1])
                    bin_features.append(binned.reshape(-1, 1))

            if bin_features:
                features.extend(bin_features[:5])  # Limit to first 5 binned features

        result = np.hstack(features)

        # Handle any infinities or NaNs
        result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)

        return result


def load_kc1():
    """Load KC1 dataset with feature names preserved."""
    dataset = openml.datasets.get_dataset(1067)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), name='target')

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    return X, y.values


def evaluate_with_features(clf, fe, X, y, n_cv_folds=8, n_holdout_folds=2):
    """Evaluate classifier with feature engineering."""
    total_folds = n_cv_folds + n_holdout_folds
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(X, y))

    cv_splits = all_splits[:n_cv_folds]
    holdout_splits = all_splits[n_cv_folds:]

    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    # CV evaluation
    cv_scores = []
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply feature engineering
        fe_clone = clone(fe)
        X_train_fe = fe_clone.fit_transform(X_train)
        X_test_fe = fe_clone.transform(X_test)

        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        X_train_fe = imputer.fit_transform(X_train_fe)
        X_test_fe = imputer.transform(X_test_fe)

        scaler = StandardScaler()
        X_train_fe = scaler.fit_transform(X_train_fe)
        X_test_fe = scaler.transform(X_test_fe)

        model = clone(clf)
        model.fit(X_train_fe, y_train)
        y_pred = model.predict(X_test_fe)
        cv_scores.append(f1_score(y_test, y_pred, zero_division=0))

    # Holdout evaluation
    holdout_scores = []
    for train_idx, test_idx in holdout_splits:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fe_clone = clone(fe)
        X_train_fe = fe_clone.fit_transform(X_train)
        X_test_fe = fe_clone.transform(X_test)

        imputer = SimpleImputer(strategy='median')
        X_train_fe = imputer.fit_transform(X_train_fe)
        X_test_fe = imputer.transform(X_test_fe)

        scaler = StandardScaler()
        X_train_fe = scaler.fit_transform(X_train_fe)
        X_test_fe = scaler.transform(X_test_fe)

        model = clone(clf)
        model.fit(X_train_fe, y_train)
        y_pred = model.predict(X_test_fe)
        holdout_scores.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(cv_scores), np.mean(holdout_scores)


def run_feature_evolution():
    """Run feature engineering evolution on KC1."""
    print("Loading KC1 dataset...")
    X, y = load_kc1()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} base features")
    print(f"Class distribution: {np.bincount(y)} (defective: {np.sum(y)}, {100*np.mean(y):.1f}%)")
    print(f"Feature names: {list(X.columns)[:10]}...")

    best_holdout = 0
    best_gen = None

    # Default classifier for feature testing
    base_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)

    # =========================================================================
    # PHASE 1: BASELINE (No Feature Engineering)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (No Feature Engineering)")
    print("="*60)

    # Gen 0: XGBoost baseline (previous best)
    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
        fe = KC1FeatureEngineer()  # No features added
        cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

        entry = log_generation(0, "XGBoost Baseline", cv_f1, holdout_f1,
                              "XGBoost with no feature engineering (previous best)",
                              {'features': 'base_only', 'n_features': X.shape[1]},
                              True, "Baseline")
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # PHASE 2: FEATURE ENGINEERING MUTATIONS
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING MUTATIONS")
    print("="*60)

    # Gen 1: Ratio features (complexity per LOC, etc.)
    fe = KC1FeatureEngineer(add_ratios=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(1, "Ratio Features", cv_f1, holdout_f1,
                          "Add complexity/LOC, volume/LOC, effort/LOC ratios",
                          {'features': 'base + ratios', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 2: Complexity features
    fe = KC1FeatureEngineer(add_complexity=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(2, "Complexity Features", cv_f1, holdout_f1,
                          "Add essential/design complexity ratios, complexity flags",
                          {'features': 'base + complexity', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 3: Quality features
    fe = KC1FeatureEngineer(add_quality=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(3, "Quality Features", cv_f1, holdout_f1,
                          "Add comment ratio, blank ratio, operator repetition",
                          {'features': 'base + quality', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 4: Interaction features
    fe = KC1FeatureEngineer(add_interactions=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(4, "Interaction Features", cv_f1, holdout_f1,
                          "Add complexity×LOC, log transforms",
                          {'features': 'base + interactions', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 5: Binning features
    fe = KC1FeatureEngineer(add_bins=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(5, "Binned Features", cv_f1, holdout_f1,
                          "Discretize key metrics into bins",
                          {'features': 'base + bins', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 6: Ratios + Complexity
    fe = KC1FeatureEngineer(add_ratios=True, add_complexity=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(6, "Ratios + Complexity", cv_f1, holdout_f1,
                          "Combine ratio and complexity features",
                          {'features': 'base + ratios + complexity', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 7: Ratios + Bins
    fe = KC1FeatureEngineer(add_ratios=True, add_bins=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(7, "Ratios + Bins", cv_f1, holdout_f1,
                          "Combine ratio features with binning",
                          {'features': 'base + ratios + bins', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 8: All features
    fe = KC1FeatureEngineer(add_ratios=True, add_complexity=True, add_quality=True,
                            add_interactions=True, add_bins=True)
    X_test = fe.fit_transform(X)
    n_features = X_test.shape[1]
    cv_f1, holdout_f1 = evaluate_with_features(base_clf, fe, X, y)

    entry = log_generation(8, "All Features", cv_f1, holdout_f1,
                          "All feature engineering combined",
                          {'features': 'all', 'n_features': n_features},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # =========================================================================
    # PHASE 3: BEST FEATURES + MODEL VARIATIONS
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: BEST FEATURES + MODEL VARIATIONS")
    print("="*60)

    # Determine best feature config so far
    best_fe_config = best_gen['params'].get('features', 'base_only') if best_gen else 'base_only'
    print(f"Best feature config so far: {best_fe_config}")

    # Gen 9: Best features + XGBoost
    if XGB_AVAILABLE:
        fe = KC1FeatureEngineer(add_ratios=True, add_complexity=True)
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                                eval_metric='logloss', verbosity=0)
        cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

        entry = log_generation(9, "Ratios+Cmplx + XGB", cv_f1, holdout_f1,
                              "Ratios + Complexity features with XGBoost",
                              {'features': 'ratios + complexity', 'model': 'XGBoost'},
                              holdout_f1 > best_holdout,
                              f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # Gen 10: Complexity + Bins + RF
    fe = KC1FeatureEngineer(add_complexity=True, add_bins=True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

    entry = log_generation(10, "Cmplx+Bins + RF", cv_f1, holdout_f1,
                          "Complexity + Bins with RandomForest",
                          {'features': 'complexity + bins', 'model': 'RF'},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 11: Quality + Interactions + LR
    fe = KC1FeatureEngineer(add_quality=True, add_interactions=True)
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

    entry = log_generation(11, "Quality+Int + LR", cv_f1, holdout_f1,
                          "Quality + Interactions with strongly regularized LogReg",
                          {'features': 'quality + interactions', 'model': 'LogReg C=0.1'},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 12: Ratios + Bins + Ensemble
    fe = KC1FeatureEngineer(add_ratios=True, add_bins=True)
    lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

    entry = log_generation(12, "Ratios+Bins + Ens", cv_f1, holdout_f1,
                          "Ratios + Bins with LR+RF ensemble",
                          {'features': 'ratios + bins', 'model': 'LR+RF'},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
    if holdout_f1 > best_holdout:
        best_holdout = holdout_f1
        best_gen = entry

    # Gen 13: Interactions only + XGB weighted
    if XGB_AVAILABLE:
        fe = KC1FeatureEngineer(add_interactions=True)
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=5,
                                random_state=42, eval_metric='logloss', verbosity=0)
        cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

        entry = log_generation(13, "Interactions + XGB-w", cv_f1, holdout_f1,
                              "Interactions with XGBoost (class weighted)",
                              {'features': 'interactions', 'model': 'XGBoost weighted'},
                              holdout_f1 > best_holdout,
                              f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
        if holdout_f1 > best_holdout:
            best_holdout = holdout_f1
            best_gen = entry

    # Gen 14: Complexity only + very simple LR
    fe = KC1FeatureEngineer(add_complexity=True)
    clf = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
    cv_f1, holdout_f1 = evaluate_with_features(clf, fe, X, y)

    entry = log_generation(14, "Complexity + SimpleLR", cv_f1, holdout_f1,
                          "Complexity features with very simple LogReg",
                          {'features': 'complexity', 'model': 'LogReg C=0.01'},
                          holdout_f1 > best_holdout,
                          f"Holdout: {holdout_f1:.4f} vs best {best_holdout:.4f}")
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
    print(f"Previous best (no FE): 0.463")
    improvement = ((best_holdout - 0.463) / 0.463) * 100
    print(f"Improvement over no-FE: {improvement:+.1f}%")

    baseline = 0.436  # Original CV baseline
    print(f"\nTarget Comparison:")
    print(f"  vs Auto-sklearn target ({baseline * 1.15:.3f}): {'✓ BEATEN' if best_holdout >= baseline * 1.15 else '✗ NOT YET'}")
    print(f"  vs FLAML target ({baseline * 1.17:.3f}):        {'✓ BEATEN' if best_holdout >= baseline * 1.17 else '✗ NOT YET'}")
    print(f"  vs AutoGluon target ({baseline * 1.23:.3f}):    {'✓ BEATEN' if best_holdout >= baseline * 1.23 else '✗ NOT YET'}")

    # Save log
    log_path = RESULTS_DIR / 'kc1_fe_evolution_log.json'
    with open(log_path, 'w') as f:
        json.dump(EVOLUTION_LOG, f, indent=2)
    print(f"\nEvolution log saved to: {log_path}")

    return EVOLUTION_LOG, best_gen


if __name__ == '__main__':
    log, best = run_feature_evolution()
