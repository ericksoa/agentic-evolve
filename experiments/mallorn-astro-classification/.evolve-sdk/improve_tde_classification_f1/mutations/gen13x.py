#!/usr/bin/env python3
"""
Gen13x Crossover: Best of Gen11 Champion + Weaker Regularization

CROSSOVER STRATEGY:
- From Gen11 Champion (0.5296): High threshold (0.43) + robust documentation + structure
- From Gen11b Threshold42 (0.527): Clean, compact code implementation
- From Gen11c Weaker Reg (0.5259): Weaker regularization (C=0.08) for better feature learning

HYPOTHESIS: High threshold (0.43) for precision + weaker regularization (C=0.08)
for polynomial features to express more complex patterns = best of both worlds.

Key innovations combined:
1. Polynomial features (degree=2) for top 6 predictors (all parents)
2. High threshold (0.43) for precision optimization (Champion)
3. Weaker regularization (C=0.08) to let interactions shine (Weaker Reg)
4. Clean implementation structure (Threshold42)

Top features for interactions:
- g_skew, r_scatter, r_skew, i_skew, i_kurtosis, r_kurtosis
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen13x_Crossover(BaseEstimator, ClassifierMixin):
    """
    Gen13x Crossover: High precision threshold + weaker regularization for features

    Combines:
    - Champion's high threshold (0.43) for precision
    - Weaker regularization (C=0.08) for feature expressiveness
    - Clean implementation from Threshold42
    """

    # Top features from importance analysis (consistent across all parents)
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew',
        'i_kurtosis', 'r_kurtosis'
    ]

    def __init__(self, threshold: float = 0.43, C: float = 0.08):
        """
        Initialize with optimal parameter combination from crossover.

        Args:
            threshold: High threshold for precision (from Champion)
            C: Weaker regularization for feature learning (from Weaker Reg)
        """
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen13x_Crossover':
        """Fit with polynomial feature interactions and optimized parameters."""
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
                interaction_only=False  # Include x^2 terms for completeness
            )
            X_poly = self.poly_.fit_transform(X_top)
            # Append new interaction features (skip original features already in X_scaled)
            n_original = len(self.top_feature_indices_)
            X_new = X_poly[:, n_original:]
            X_final = np.hstack([X_scaled, X_new])
        else:
            X_final = X_scaled
            self.poly_ = None

        # Step 4: Fit logistic regression with crossover parameters
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,  # Weaker regularization from Gen11c
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)

        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with polynomial interactions."""
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
        """Predict class probabilities."""
        X_final = self._transform(X)
        return self.lr_.predict_proba(X_final)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with high precision threshold from Champion."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases for benchmark discovery
Gen13_Candidate = Gen13x_Crossover
TDEClassifier = Gen13x_Crossover