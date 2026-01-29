#!/usr/bin/env python3
"""
Gen11: Feature Interactions for Top Predictors

Building on Gen10's success (F1=0.5025), this variant adds polynomial
interaction features for the top predictors identified in feature importance:
- g_skew, r_scatter, r_skew, i_skew (top 4)
- i_kurtosis, r_kurtosis (top 6)

Key insight: TDE physics suggests correlations between bands (color evolution)
and between statistical moments (shape consistency). Interaction terms can
capture these cross-band patterns.

Changes from Gen10:
- Add pairwise interaction terms for top 6 features
- Add squared terms for top 4 features
- Keep strong regularization (C=0.05) to prevent overfitting from extra features
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class Gen11_Interactions(BaseEstimator, ClassifierMixin):
    """
    Gen 11: LogReg with feature interactions.

    Adds polynomial features for top predictors while keeping
    strong regularization to prevent overfitting.
    """

    # Top features from importance analysis (Gen 5)
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew',
        'i_kurtosis', 'r_kurtosis'
    ]

    def __init__(self, threshold: float = 0.40, C: float = 0.05,
                 interaction_degree: int = 2, include_bias: bool = False):
        self.threshold = threshold
        self.C = C
        self.interaction_degree = interaction_degree
        self.include_bias = include_bias
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen11_Interactions':
        """Fit with feature interactions."""
        self.feature_names_ = list(X.columns)

        # Find indices of top features that exist in the data
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

        # Step 3: Create interaction features for top predictors only
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(
                degree=self.interaction_degree,
                include_bias=self.include_bias,
                interaction_only=False  # Include x^2 terms
            )
            X_interactions = self.poly_.fit_transform(X_top)

            # Remove the original features from poly output (they're already in X_scaled)
            # poly returns: [1, x1, x2, ..., x1^2, x1*x2, x2^2, ...]
            # We want just the interaction/polynomial terms
            n_original = len(self.top_feature_indices_)
            X_new_features = X_interactions[:, n_original + (1 if self.include_bias else 0):]

            # Combine original features with new interaction features
            X_final = np.hstack([X_scaled, X_new_features])
        else:
            X_final = X_scaled
            self.poly_ = None

        # Step 4: Fit logistic regression with same strong regularization
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,  # Keep strong regularization
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)

        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with interactions."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        if self.poly_ is not None and len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_interactions = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_interactions[:, n_original + (1 if self.include_bias else 0):]
            X_final = np.hstack([X_scaled, X_new_features])
        else:
            X_final = X_scaled

        return X_final

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_final = self._transform(X)
        return self.lr_.predict_proba(X_final)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_n_features(self) -> int:
        """Return number of features after interaction expansion."""
        if self.lr_ is not None:
            return self.lr_.coef_.shape[1]
        return 0


# Aliases for benchmark discovery
Gen11_Candidate = Gen11_Interactions
TDEClassifier = Gen11_Interactions
