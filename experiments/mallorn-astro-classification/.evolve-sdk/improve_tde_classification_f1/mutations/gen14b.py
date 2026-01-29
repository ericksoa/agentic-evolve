#!/usr/bin/env python3
"""
Gen14b: Random Forest Swap from Logistic Regression

Swapping LogisticRegression with RandomForestClassifier to leverage its
natural feature interaction capabilities and potentially improve F1 score.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


class Gen14b(BaseEstimator, ClassifierMixin):
    """Gen 14b: Random Forest with feature interactions"""

    TOP_FEATURES = ['g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis', 'r_kurtosis']

    def __init__(self, threshold: float = 0.42, n_estimators: int = 100, max_depth: int = 10):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen14b':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [i for i, col in enumerate(self.feature_names_) if col in self.TOP_FEATURES]

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_interactions = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_new_features = X_interactions[:, n_original:]
            X_final = np.hstack([X_scaled, X_new_features])
        else:
            X_final = X_scaled

        self.rf_ = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.rf_.fit(X_final, y)
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
        return self.rf_.predict_proba(self._transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


Gen14_Candidate = Gen14b
TDEClassifier = Gen14b