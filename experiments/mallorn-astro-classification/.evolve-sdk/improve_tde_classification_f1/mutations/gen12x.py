#!/usr/bin/env python3
"""
Gen12x Hybrid: Optimized Threshold + Clean Architecture

CROSSOVER STRATEGY:
- Best parameters from Gen11_Champion (threshold=0.43, C=0.05) - highest fitness 0.5296
- Clean code structure from Gen11b/Gen11c - more maintainable
- Enhanced threshold with micro-optimization to 0.425 (between 0.42 and 0.43)
- Comprehensive documentation from Gen11_Champion

PARENT ANALYSIS:
- Gen11_Champion: Best performance but verbose code
- Gen11b_Threshold42: Clean structure, threshold=0.42 performed well
- Gen11c_WeakerReg: Shows C=0.08 hurts performance vs C=0.05

HYBRID IMPROVEMENTS:
1. Optimal parameters: threshold=0.425, C=0.05
2. Clean, efficient implementation
3. Same proven polynomial feature strategy
4. Better balance of precision/recall through fine-tuned threshold
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen12x_Hybrid(BaseEstimator, ClassifierMixin):
    """
    Gen12x Hybrid: Crossover of best performing Gen11 variants

    Combines:
    - Optimal threshold (0.425) between Gen11_Champion (0.43) and Gen11b (0.42)
    - Proven regularization (C=0.05) from champion
    - Clean architecture from Gen11b/c
    - Same polynomial feature interactions on top 6 predictors

    Expected improvement through fine-tuned threshold optimization.
    """

    # Top features for polynomial interactions (proven across all parents)
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew',
        'i_kurtosis', 'r_kurtosis'
    ]

    def __init__(self, threshold: float = 0.425, C: float = 0.05):
        """
        Initialize with hybrid-optimized parameters.

        Args:
            threshold: Decision threshold (0.425 = optimal between best parents)
            C: Regularization strength (0.05 = proven best from champion)
        """
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12x_Hybrid':
        """
        Fit hybrid model with polynomial feature interactions.

        Pipeline:
        1. Impute missing values (median strategy)
        2. Standard scaling
        3. Polynomial features (degree=2) for top 6 predictors
        4. Logistic regression with balanced classes and strong regularization
        """
        self.feature_names_ = list(X.columns)

        # Find indices of top features for polynomial expansion
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in self.TOP_FEATURES
        ]

        # Step 1: Impute missing values
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        # Step 2: Standard scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Step 3: Create polynomial features for top predictors
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(
                degree=2,
                include_bias=False,
                interaction_only=False  # Include both interactions and squares
            )
            X_poly = self.poly_.fit_transform(X_top)

            # Append new polynomial features (skip original features)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_poly[:, n_original:]
            X_final = np.hstack([X_scaled, X_new_features])
        else:
            X_final = X_scaled
            self.poly_ = None

        # Step 4: Fit logistic regression with optimal parameters
        self.lr_ = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            C=self.C,                 # Strong regularization from champion
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)

        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input data through the fitted pipeline."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        if self.poly_ is not None and len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_poly[:, n_original:]
            return np.hstack([X_scaled, X_new_features])

        return X_scaled

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        X_transformed = self._transform(X)
        return self.lr_.predict_proba(X_transformed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using optimized threshold.

        Uses threshold=0.425, optimally positioned between the best
        performing parent thresholds (0.42 and 0.43).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases for benchmark discovery
Gen12_Candidate = Gen12x_Hybrid
TDEClassifier = Gen12x_Hybrid