#!/usr/bin/env python3
"""
Gen14a: Higher-Order Feature Interactions

MUTATION: Increased polynomial degree from 2 to 3 on top 4 features
- Parent: degree=2 on top 6 features
- Mutation: degree=3 on top 4 features
- Hypothesis: Higher-order interactions may capture TDE nonlinear patterns better

Key changes from Gen11:
1. Polynomial degree increased from 2 to 3
2. Feature set reduced from 6 to 4 most important features
3. Captures cubic interactions and x^3 terms
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen14a_Mutation(BaseEstimator, ClassifierMixin):
    """
    Gen 14a Mutation: LogReg + Higher-Order Feature Interactions

    Mutation: degree=3 on top 4 features (vs degree=2 on top 6)
    """

    # Top 4 most important features for cubic interactions
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew'
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen14a_Mutation':
        """Fit with cubic polynomial feature interactions."""
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

        # Step 3: Create cubic polynomial features for top 4 predictors
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(
                degree=3,  # MUTATION: Increased from 2 to 3
                include_bias=False,
                interaction_only=False  # Include x^2, x^3 terms
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
        """Transform with cubic polynomial features."""
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
Gen14a_Candidate = Gen14a_Mutation
TDEClassifier = Gen14a_Mutation