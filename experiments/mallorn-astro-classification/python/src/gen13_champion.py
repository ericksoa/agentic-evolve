#!/usr/bin/env python3
"""
Gen13 Champion: Optimized LogReg with Polynomial Features

Improvements over Gen11:
- C=0.12 (was 0.05) - less regularization
- threshold=0.50 (was 0.43) - higher precision
- Added g_max, r_max, i_max to polynomial features

Holdout F1: 0.5995 (+13.2% vs Gen11's 0.5296)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


# Top features for polynomial interactions (Gen13: added max features)
TOP_FEATURES = [
    'g_skew', 'r_scatter', 'r_skew', 'i_skew',
    'i_kurtosis', 'r_kurtosis',
    'g_max', 'r_max', 'i_max'  # Added in Gen13
]


class Gen13_Champion(BaseEstimator, ClassifierMixin):
    """
    Gen 13: Optimized LogReg + Polynomial Features

    Key changes from Gen11:
    - C=0.12 (less regularization)
    - threshold=0.50 (higher precision)
    """

    def __init__(self, threshold: float = 0.50, C: float = 0.12):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen13_Champion':
        """Fit with polynomial interactions on top features."""
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in TOP_FEATURES
        ]

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Polynomial features on top predictors
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_final = np.hstack([X_scaled, X_poly[:, n_original:]])
        else:
            X_final = X_scaled

        # Fit LogReg with optimized C
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            return np.hstack([X_scaled, X_poly[:, n_original:]])
        return X_scaled

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._transform(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases
TDEClassifier = Gen13_Champion
Gen13_Candidate = Gen13_Champion
