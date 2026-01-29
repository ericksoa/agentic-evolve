#!/usr/bin/env python3
"""
Gen2a Mutation - Stronger Regularization

Mutation from gen0_a: Reduced C from 0.05 to 0.035 for stronger regularization.
This should help prevent overfitting on the small dataset with <50 positive examples.

Key change:
- C parameter: 0.05 → 0.035 (30% reduction for stronger L2 penalty)

Hypothesis: Stronger regularization will improve generalization on the tiny positive class,
potentially improving recall without sacrificing too much precision.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Gen2a_StrongerRegularization(BaseEstimator, ClassifierMixin):
    """
    Gen2a: Stronger regularization mutation from gen0_a.

    Reduced C from 0.05 to 0.035 to increase L2 penalty.
    """

    def __init__(self, threshold: float = 0.40, C: float = 0.035):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen2a_StrongerRegularization':
        """Fit with stronger regularization."""
        self.feature_names_ = list(X.columns)

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,
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


# Alias for easy discovery
Gen2a_Candidate = Gen2a_StrongerRegularization
TDEClassifier = Gen2a_StrongerRegularization