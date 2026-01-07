"""
Classification logic for TDE identification.

This module contains the classifier that can be evolved to improve
TDE detection performance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Optional, Tuple, Dict, Any
import xgboost as xgb

# TabPFN for Gen 8
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


class TDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier for identifying TDEs from astronomical light curves.

    This classifier wraps various ML models and can be configured
    to use different underlying algorithms and parameters.
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        model_params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        use_class_weights: bool = True
    ):
        """
        Initialize the TDE classifier.

        Args:
            model_type: One of 'xgboost', 'random_forest', 'gradient_boosting'
            model_params: Parameters to pass to the underlying model
            threshold: Classification threshold for TDE prediction
            use_class_weights: Whether to use class weights for imbalanced data
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.threshold = threshold
        self.use_class_weights = use_class_weights

        self.model_ = None
        self.scaler_ = None
        self.imputer_ = None
        self.feature_names_ = None

    def _create_model(self, class_weight: Optional[Dict] = None):
        """Create the underlying ML model."""
        params = self.model_params.copy()

        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            if self.use_class_weights and class_weight:
                # XGBoost uses scale_pos_weight
                default_params['scale_pos_weight'] = class_weight.get(1, 1)
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)

        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            if self.use_class_weights:
                default_params['class_weight'] = 'balanced'
            default_params.update(params)
            return RandomForestClassifier(**default_params)

        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TDEClassifier':
        """
        Fit the classifier.

        Args:
            X: Feature DataFrame
            y: Target labels (1 for TDE, 0 for non-TDE)

        Returns:
            self
        """
        self.feature_names_ = list(X.columns)

        # Handle missing values
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Calculate class weights for imbalanced data
        class_weight = None
        if self.use_class_weights:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            if n_pos > 0:
                class_weight = {0: 1.0, 1: n_neg / n_pos}

        # Create and fit model
        self.model_ = self._create_model(class_weight)
        self.model_.fit(X_scaled, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of TDE class.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with [P(non-TDE), P(TDE)]
        """
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict TDE class using the configured threshold.

        Args:
            X: Feature DataFrame

        Returns:
            Binary predictions (1 for TDE, 0 for non-TDE)
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.model_ is None:
            raise ValueError("Model not fitted yet")

        if hasattr(self.model_, 'feature_importances_'):
            return pd.Series(
                self.model_.feature_importances_,
                index=self.feature_names_
            ).sort_values(ascending=False)

        return pd.Series(dtype=float)


# =============================================================================
# EVOLVED CLASSIFIER (placeholder for evolution)
# =============================================================================

class EvolvedTDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Evolved TDE classifier with ensemble strategy.

    Evolution history:
    - Gen 1: Default XGBoost (F1 = 0.18)
    - Gen 2: With physics features (Optimal F1 = 0.447)
    - Gen 3: Threshold optimization + tuned class weights
    - Gen 4: Ensemble (LogReg + XGBoost) with soft voting -> F1 = 0.415 CV, 0.4154 PUBLIC (BEST)
    - Gen 5: Feature selection (20 features) -> F1 = 0.552 CV, 0.3227 PUBLIC (OVERFIT!)
    - Gen 6: LightGBM -> F1 = 0.575 CV, 0.3191 PUBLIC (MORE OVERFIT!)
    - Gen 7: Revert to Gen 4 + stronger regularization -> Focus on generalization
    """

    # Gen 7: Use MORE features (less aggressive selection reduces overfitting)
    # Top 50 features instead of 20
    SELECTED_FEATURES = [
        # Top statistical features (keep)
        'g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis',
        'r_kurtosis', 'u_max', 'g_min', 'u_range', 'y_mean',
        'r_min', 'i_scatter', 'g_wmean', 'r_peak_to_baseline', 'u_std',
        'y_wmean', 'g_decay_fit_rmse', 'g_scatter', 'g_r_color_at_peak', 'g_reduced_chi_sq',
        # Add more features for robustness (Gen 7)
        'g_max', 'r_max', 'i_max', 'z_max', 'u_mean', 'g_mean', 'r_mean', 'i_mean',
        'z_mean', 'u_std', 'g_std', 'r_std', 'i_std', 'z_std', 'y_std',
        'u_skew', 'z_skew', 'y_skew', 'g_kurtosis', 'u_kurtosis', 'z_kurtosis', 'y_kurtosis',
        'g_range', 'r_range', 'i_range', 'z_range', 'y_range',
        'r_wmean', 'i_wmean', 'z_wmean', 'u_wmean'
    ]

    def __init__(self, threshold: float = 0.35, use_feature_selection: bool = True):
        self.threshold = threshold
        self.use_feature_selection = use_feature_selection
        self.optimal_threshold_ = None

        # Gen 7: Back to LR + XGBoost (XGBoost generalized better than LightGBM)
        self.lr_model_ = None
        self.xgb_model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.selected_features_ = None

        # Gen 7: Ensemble weights (LR=0.5, XGB=0.5 - more conservative)
        self.lr_weight_ = 0.5
        self.xgb_weight_ = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EvolvedTDEClassifier':
        """
        Gen 7: Fit LR + XGBoost ensemble with stronger regularization.
        Focus: Generalization over CV score.
        """
        from sklearn.linear_model import LogisticRegression

        self.feature_names_ = list(X.columns)

        # Gen 7: Less aggressive feature selection (50 features instead of 20)
        if self.use_feature_selection:
            available = [f for f in self.SELECTED_FEATURES if f in X.columns]
            self.selected_features_ = available if len(available) >= 20 else list(X.columns)
            X = X[self.selected_features_]
        else:
            self.selected_features_ = list(X.columns)

        # Preprocess
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Calculate class weight
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos = n_neg / max(n_pos, 1)

        # Model 1: Logistic Regression with STRONGER regularization (Gen 7)
        self.lr_model_ = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1,  # Gen 7: Stronger regularization (was 0.5)
            random_state=42
        )
        self.lr_model_.fit(X_scaled, y)

        # Gen 7: Back to XGBoost with more regularization
        self.xgb_model_ = xgb.XGBClassifier(
            n_estimators=100,  # Gen 7: Fewer trees (was 200)
            max_depth=2,       # Gen 7: Shallower (was 3)
            learning_rate=0.05,
            subsample=0.6,     # Gen 7: More dropout (was 0.7)
            colsample_bytree=0.6,
            min_child_weight=10,  # Gen 7: Larger min samples (was 5)
            scale_pos_weight=scale_pos,
            reg_alpha=1.0,     # Gen 7: L1 regularization
            reg_lambda=2.0,    # Gen 7: L2 regularization
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        self.xgb_model_.fit(X_scaled, y)

        # Gen 7: Fixed weights instead of optimizing (prevents overfitting to train)
        self.lr_weight_ = 0.5
        self.xgb_weight_ = 0.5
        self.threshold = 0.35  # Gen 7: Fixed threshold

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability with Gen 7 LR + XGBoost ensemble."""
        # Apply feature selection
        if self.use_feature_selection and self.selected_features_:
            available = [f for f in self.selected_features_ if f in X.columns]
            X = X[available]

        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        lr_proba = self.lr_model_.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model_.predict_proba(X_scaled)[:, 1]

        combined = self.lr_weight_ * lr_proba + self.xgb_weight_ * xgb_proba

        # Return in sklearn format [P(0), P(1)]
        return np.column_stack([1 - combined, combined])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with optimized threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# =============================================================================
# GEN 9: LOGISTIC REGRESSION ONLY (Simpler is Better)
# =============================================================================

class Gen9_LROnly(BaseEstimator, ClassifierMixin):
    """
    Gen 9: Pure Logistic Regression - simpler is better on tiny data.
    SUPERSEDED by Gen10 (threshold=0.40 is better).
    """

    def __init__(self, threshold: float = 0.35):


class Gen10_LROnly(BaseEstimator, ClassifierMixin):
    """
    Gen 10: Pure Logistic Regression with optimized threshold.

    Evolution history:
    - Gen 4: LR + XGBoost ensemble -> 0.4154 PUBLIC (was best)
    - Gen 9: Pure LogReg (C=0.05, t=0.35) -> Holdout F1 = 0.4611
    - Gen 10: Pure LogReg (C=0.05, t=0.40) -> Holdout F1 = 0.5025 (+21% vs Gen 4)

    Key insight: Higher threshold (0.40 vs 0.35) improves precision without
    losing too much recall on this imbalanced dataset.

    Holdout validation:
    - CV Mean: 0.2348
    - Holdout Mean: 0.5025 (+21% vs 0.4154 best public)
    - Gap: -0.2677 (PASS)
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen10_LROnly':
        """Fit with C=0.05 regularization and threshold=0.40."""
        from sklearn.linear_model import LogisticRegression

        self.feature_names_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=0.05,
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_scaled, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self.lr_.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


class Gen9_LROnly_Original(BaseEstimator, ClassifierMixin):
    """Gen 9 Original (kept for reference). Use Gen10_LROnly instead."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen9_LROnly':
        """
        Fit with very strong regularization (C=0.05).

        On small data, strong regularization prevents overfitting to noise.
        """
        from sklearn.linear_model import LogisticRegression

        self.feature_names_ = list(X.columns)

        # Preprocess
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Very strong regularization (C=0.05)
        # Balanced class weights handle the imbalance
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=0.05,  # Strong regularization - key for generalization
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_scaled, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self.lr_.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with fixed threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_feature_importance(self) -> pd.Series:
        """Get feature coefficients as importance scores."""
        if self.lr_ is None:
            raise ValueError("Model not fitted yet")

        return pd.Series(
            np.abs(self.lr_.coef_[0]),
            index=self.feature_names_
        ).sort_values(ascending=False)


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        score = metric_funcs[metric](y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# =============================================================================
# GEN 8: TabPFN CLASSIFIER (Foundation model for small tabular data)
# =============================================================================

class TabPFNTDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Gen 8: TabPFN-based TDE classifier.

    TabPFN is a foundation model pre-trained on synthetic tabular data.
    Designed for small datasets (<1000 samples) - perfect for MALLORN.

    Key advantages:
    - No hyperparameter tuning needed (pre-trained)
    - Excellent generalization on small data
    - Built-in uncertainty quantification
    - Should avoid overfitting that plagued Gen 5-6

    Evolution history:
    - Gen 4: LR + XGBoost ensemble -> F1 = 0.415 CV, 0.4154 PUBLIC (BEST)
    - Gen 5-6: Overfit (higher CV, worse public)
    - Gen 7: Anti-overfit measures (too conservative)
    - Gen 8: TabPFN foundation model (this) - designed for small data
    """

    def __init__(self, threshold: float = 0.10, n_ensemble: int = 4):
        """
        Args:
            threshold: Classification threshold (TabPFN outputs lower probs, use 0.10)
            n_ensemble: Number of TabPFN ensemble members (default 4 is fast)
        """
        self.threshold = threshold
        self.n_ensemble = n_ensemble
        self.model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TabPFNTDEClassifier':
        """Fit TabPFN classifier."""
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not installed. Run: pip install tabpfn")

        self.feature_names_ = list(X.columns)

        # Preprocess: impute and scale
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # TabPFN requires <= 100 features and <= 1000 samples
        # MALLORN has ~120 features, need to reduce
        n_features = X_scaled.shape[1]
        if n_features > 100:
            # Use variance-based selection (simple, no overfitting risk)
            variances = np.var(X_scaled, axis=0)
            top_100_idx = np.argsort(variances)[-100:]
            X_scaled = X_scaled[:, top_100_idx]
            self.feature_idx_ = top_100_idx
        else:
            self.feature_idx_ = np.arange(n_features)

        # Initialize TabPFN (v6 API: n_estimators)
        self.model_ = TabPFNClassifier(
            n_estimators=self.n_ensemble,
            device='cpu'  # Use CPU for stability
        )

        # Fit (TabPFN is pre-trained, this just stores the training data)
        self.model_.fit(X_scaled, y.values)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability with TabPFN."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        # Apply same feature selection
        X_scaled = X_scaled[:, self.feature_idx_]

        return self.model_.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with fixed threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


class Gen8HybridClassifier(BaseEstimator, ClassifierMixin):
    """
    Gen 8 Hybrid: TabPFN + Logistic Regression ensemble.

    Combines:
    - TabPFN: Foundation model for small data (novel)
    - LogReg: Stable, proven on this problem (from Gen 4)

    Uses FIXED weights (0.5/0.5) to avoid overfitting.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.tabpfn_ = None
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_idx_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen8HybridClassifier':
        """Fit hybrid TabPFN + LR ensemble."""
        from sklearn.linear_model import LogisticRegression

        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not installed. Run: pip install tabpfn")

        # Preprocess
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Feature reduction for TabPFN (max 100 features)
        n_features = X_scaled.shape[1]
        if n_features > 100:
            variances = np.var(X_scaled, axis=0)
            self.feature_idx_ = np.argsort(variances)[-100:]
        else:
            self.feature_idx_ = np.arange(n_features)

        X_reduced = X_scaled[:, self.feature_idx_]

        # Fit TabPFN (v6 API: n_estimators)
        self.tabpfn_ = TabPFNClassifier(
            n_estimators=4,
            device='cpu'
        )
        self.tabpfn_.fit(X_reduced, y.values)

        # Fit Logistic Regression on full features
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.5,  # Moderate regularization (like Gen 4)
            random_state=42
        )
        self.lr_.fit(X_scaled, y.values)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction with fixed 0.5/0.5 weights."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        X_reduced = X_scaled[:, self.feature_idx_]

        # Get probabilities from both models
        tabpfn_proba = self.tabpfn_.predict_proba(X_reduced)[:, 1]
        lr_proba = self.lr_.predict_proba(X_scaled)[:, 1]

        # Fixed weights (no optimization to avoid overfitting)
        combined = 0.5 * tabpfn_proba + 0.5 * lr_proba

        return np.column_stack([1 - combined, combined])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with fixed threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
