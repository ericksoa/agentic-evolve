#!/usr/bin/env python3
"""
Gen11 Champion: Feature Interactions with Optimized Parameters

EVOLUTION RESULT: Holdout F1 = 0.5296 (+5.4% vs Gen10's 0.5025)

Key improvements over Gen10:
1. Polynomial features (degree=2) for top 6 predictors
2. Higher threshold (0.43 vs 0.40) for better precision
3. Same strong regularization (C=0.05) to prevent overfitting

Top features used for interactions:
- g_skew, r_scatter, r_skew, i_skew, i_kurtosis, r_kurtosis

These features capture:
- Cross-band correlations (TDEs have consistent color evolution)
- Statistical moment interactions (shape consistency across bands)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen11_Champion(BaseEstimator, ClassifierMixin):
    """
    Gen 11 Champion: LogReg + Feature Interactions

    Holdout F1: 0.5296 (+5.4% improvement over Gen10)
    """

    # Top features from importance analysis
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew',
        'i_kurtosis', 'r_kurtosis'
    ]

    def __init__(self, threshold: float = 0.43, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen11_Champion':
        """Fit with polynomial feature interactions."""
        self.feature_names_ = list(X.columns)

        # Find indices of top features
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in self.TOP_FEATURES
        ]

        # Step 1: Impute missing values
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        # Step 2: Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Step 3: Create polynomial features for top predictors
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(
                degree=2,
                include_bias=False,
                interaction_only=False  # Include x^2 terms
            )
            X_poly = self.poly_.fit_transform(X_top)
            # Append new features (skip original features already in X_scaled)
            n_original = len(self.top_feature_indices_)
            X_new = X_poly[:, n_original:]
            X_final = np.hstack([X_scaled, X_new])
        else:
            X_final = X_scaled
            self.poly_ = None

        # Step 4: Fit logistic regression
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)

        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform with polynomial features."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        if self.poly_ is not None and len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new = X_poly[:, n_original:]
            return np.hstack([X_scaled, X_new])

        return X_scaled

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_final = self._transform(X)
        return self.lr_.predict_proba(X_final)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases for benchmark discovery
Gen11_Candidate = Gen11_Champion
TDEClassifier = Gen11_Champion
