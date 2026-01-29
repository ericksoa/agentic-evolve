#!/usr/bin/env python3
"""
Gen1a Mutation - Ridge Classifier with Custom Threshold

Mutation Strategy: Algorithm Swap
- Replaced LogisticRegression with RidgeClassifier
- Kept same regularization philosophy but adapted for Ridge
- Added custom threshold application via decision_function
- Maintained all preprocessing and hyperparameter insights

Hypothesis: Ridge's L2 regularization might handle small sample size better
than LogisticRegression's approach, while preserving linear model simplicity.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Gen1a_Ridge(BaseEstimator, ClassifierMixin):
    """
    Gen 1a: Ridge Classifier with custom threshold.

    Mutation: Swapped LogisticRegression for RidgeClassifier
    while keeping the threshold-based prediction strategy.
    """

    def __init__(self, threshold: float = 0.40, alpha: float = 20.0):
        self.threshold = threshold
        self.alpha = alpha  # Ridge alpha roughly inverse of LR's C
        self.ridge_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen1a_Ridge':
        """Fit with Ridge regularization."""
        self.feature_names_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        self.ridge_ = RidgeClassifier(
            class_weight='balanced',
            alpha=self.alpha,
            random_state=42
        )
        self.ridge_.fit(X_scaled, y)

        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Get decision function scores."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self.ridge_.decision_function(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convert decision function to pseudo-probabilities."""
        scores = self.decision_function(X)
        # Simple sigmoid transformation for threshold compatibility
        proba_pos = 1 / (1 + np.exp(-scores))
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Alias for easy discovery
Gen1a_Candidate = Gen1a_Ridge
TDEClassifier = Gen1a_Ridge