#!/usr/bin/env python3
"""
Gen11d: Interaction Terms Only (no squared terms)

Testing if pure pairwise interactions work better than including x^2 terms.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen11d_InteractionsOnly(BaseEstimator, ClassifierMixin):
    """Gen 11d: Pairwise interactions only (no x^2)"""

    TOP_FEATURES = ['g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis', 'r_kurtosis']

    def __init__(self, threshold: float = 0.40, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen11d_InteractionsOnly':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [i for i, col in enumerate(self.feature_names_) if col in self.TOP_FEATURES]

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            # interaction_only=True means no x^2 terms, just x1*x2
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_interactions = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_interactions[:, n_original:]
            X_final = np.hstack([X_scaled, X_new_features])
        else:
            X_final = X_scaled

        self.lr_ = LogisticRegression(class_weight='balanced', C=self.C, max_iter=1000, random_state=42)
        self.lr_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_interactions = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_interactions[:, n_original:]
            return np.hstack([X_scaled, X_new_features])
        return X_scaled

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


Gen11_Candidate = Gen11d_InteractionsOnly
TDEClassifier = Gen11d_InteractionsOnly
