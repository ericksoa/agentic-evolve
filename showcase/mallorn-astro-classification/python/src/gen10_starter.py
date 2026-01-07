#!/usr/bin/env python3
"""
Gen10 Starter Solution - Pure Logistic Regression (Current Best)

This is the current best solution for the MALLORN TDE classification challenge.
Holdout F1: 0.5025 (+21% vs best public score)

Key insights from evolution:
- Simpler is better on tiny data (<50 positive examples)
- Strong regularization (C=0.05) prevents overfitting
- Higher threshold (0.40 vs 0.35) improves precision
- Use all features (no aggressive feature selection)

DO NOT:
- Use neural networks (overfit)
- Use LightGBM (overfits more than XGBoost)
- Use aggressive feature selection (<30 features)
- Optimize threshold on training data

EVOLUTION TARGETS:
- Try threshold values in range 0.38-0.45
- Try C values in range 0.03-0.10
- Try alternative linear models (Ridge, SVM-linear)
- Add interaction features for top predictors
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Gen10_LROnly(BaseEstimator, ClassifierMixin):
    """
    Gen 10: Pure Logistic Regression with optimized threshold.

    Current best solution achieving holdout F1 = 0.5025.
    """

    def __init__(self, threshold: float = 0.40, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen10_LROnly':
        """Fit with strong regularization."""
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
Gen11_Candidate = Gen10_LROnly
TDEClassifier = Gen10_LROnly
