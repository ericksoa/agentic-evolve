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
    Evolved TDE classifier with ensemble strategy and feature selection.

    Evolution history:
    - Gen 1: Default XGBoost (F1 = 0.18)
    - Gen 2: With physics features (Optimal F1 = 0.447)
    - Gen 3: Threshold optimization + tuned class weights
    - Gen 4: Ensemble (LogReg + XGBoost) with soft voting -> F1 = 0.415
    - Gen 5: Feature selection (20 best features) -> F1 = 0.552 (+33%)
    """

    # Top 20 features identified via importance ranking
    SELECTED_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis',
        'r_kurtosis', 'u_max', 'g_min', 'u_range', 'y_mean',
        'r_min', 'i_scatter', 'g_wmean', 'r_peak_to_baseline', 'u_std',
        'y_wmean', 'g_decay_fit_rmse', 'g_scatter', 'g_r_color_at_peak', 'g_reduced_chi_sq'
    ]

    def __init__(self, threshold: float = 0.35, use_feature_selection: bool = True):
        self.threshold = threshold
        self.use_feature_selection = use_feature_selection
        self.optimal_threshold_ = None

        # Ensemble components
        self.lr_model_ = None
        self.xgb_model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.selected_features_ = None

        # Ensemble weights (learned)
        self.lr_weight_ = 0.5
        self.xgb_weight_ = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EvolvedTDEClassifier':
        """
        Gen 5: Fit ensemble with feature selection and adaptive weighting.
        """
        from sklearn.linear_model import LogisticRegression

        self.feature_names_ = list(X.columns)

        # Gen 5: Feature selection
        if self.use_feature_selection:
            available = [f for f in self.SELECTED_FEATURES if f in X.columns]
            self.selected_features_ = available
            X = X[available]
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

        # Model 1: Logistic Regression (stable with small data)
        self.lr_model_ = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.5,  # Some regularization
            random_state=42
        )
        self.lr_model_.fit(X_scaled, y)

        # Model 2: XGBoost (powerful but needs more data)
        self.xgb_model_ = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,  # Very shallow to prevent overfitting
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.2,
            reg_lambda=3,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model_.fit(X_scaled, y)

        # Optimize ensemble weights and threshold using training data
        lr_proba = self.lr_model_.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model_.predict_proba(X_scaled)[:, 1]

        best_f1 = 0
        best_lr_w = 0.5
        best_thresh = 0.35

        for lr_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            xgb_w = 1 - lr_w
            combined = lr_w * lr_proba + xgb_w * xgb_proba

            thresh, f1 = optimize_threshold(y.values, combined)
            if f1 > best_f1:
                best_f1 = f1
                best_lr_w = lr_w
                best_thresh = thresh

        self.lr_weight_ = best_lr_w
        self.xgb_weight_ = 1 - best_lr_w
        self.optimal_threshold_ = best_thresh
        self.threshold = best_thresh

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability with ensemble."""
        # Apply feature selection
        if self.use_feature_selection and self.selected_features_:
            X = X[self.selected_features_]

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
