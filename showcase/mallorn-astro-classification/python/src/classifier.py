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

class EvolvedTDEClassifier(TDEClassifier):
    """
    Evolved TDE classifier with optimized parameters and ensemble strategies.

    This class will be modified by the /evolve skill to discover
    better classification strategies.
    """

    def __init__(self):
        # Evolved parameters (to be optimized)
        super().__init__(
            model_type='xgboost',
            model_params={
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
            },
            threshold=0.5,  # Can be evolved
            use_class_weights=True
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EvolvedTDEClassifier':
        """
        Evolved fitting procedure.

        Can include:
        - Feature selection
        - Threshold optimization
        - Ensemble methods
        """
        # For now, use base implementation
        # Evolution will modify this
        return super().fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Evolved prediction with optimized threshold.
        """
        # Evolved threshold optimization could go here
        return super().predict(X)


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
