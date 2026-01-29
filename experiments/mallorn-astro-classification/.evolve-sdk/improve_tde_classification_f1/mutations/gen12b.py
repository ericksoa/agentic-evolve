#!/usr/bin/env python3
"""
Gen12b: Manual Feature Engineering Instead of Polynomial

Swapping polynomial features for manual ratio/difference features.
Creating interpretable relationships between top features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Gen12b_ManualFeatures(BaseEstimator, ClassifierMixin):
    """Gen 12b: Manual feature engineering instead of polynomial"""

    TOP_FEATURES = ['g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis', 'r_kurtosis']

    def __init__(self, threshold: float = 0.42, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def _create_manual_features(self, X_scaled):
        """Create manual ratio and difference features from top features"""
        if len(self.top_feature_indices_) < 2:
            return X_scaled

        X_top = X_scaled[:, self.top_feature_indices_]
        manual_features = []

        # Create ratios and differences between key features
        # Focus on skewness relationships and scatter vs kurtosis
        if len(self.top_feature_indices_) >= 4:
            # Skewness ratios (different bands)
            g_skew_idx = next((i for i, idx in enumerate(self.top_feature_indices_)
                             if self.feature_names_[idx] == 'g_skew'), None)
            r_skew_idx = next((i for i, idx in enumerate(self.top_feature_indices_)
                             if self.feature_names_[idx] == 'r_skew'), None)
            i_skew_idx = next((i for i, idx in enumerate(self.top_feature_indices_)
                             if self.feature_names_[idx] == 'i_skew'), None)
            r_scatter_idx = next((i for i, idx in enumerate(self.top_feature_indices_)
                               if self.feature_names_[idx] == 'r_scatter'), None)

            if g_skew_idx is not None and r_skew_idx is not None:
                # Ratio of g and r skewness
                ratio = np.divide(X_top[:, g_skew_idx],
                                X_top[:, r_skew_idx] + 1e-8)  # Avoid division by zero
                manual_features.append(ratio.reshape(-1, 1))

            if r_skew_idx is not None and i_skew_idx is not None:
                # Difference between r and i skewness
                diff = X_top[:, r_skew_idx] - X_top[:, i_skew_idx]
                manual_features.append(diff.reshape(-1, 1))

            if r_scatter_idx is not None and g_skew_idx is not None:
                # Product of scatter and skewness (different type interaction)
                product = X_top[:, r_scatter_idx] * X_top[:, g_skew_idx]
                manual_features.append(product.reshape(-1, 1))

        # Add squared terms for the most important features
        for i in range(min(3, len(self.top_feature_indices_))):
            squared = X_top[:, i] ** 2
            manual_features.append(squared.reshape(-1, 1))

        if manual_features:
            X_new_features = np.hstack(manual_features)
            return np.hstack([X_scaled, X_new_features])

        return X_scaled

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12b_ManualFeatures':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [i for i, col in enumerate(self.feature_names_) if col in self.TOP_FEATURES]

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        X_final = self._create_manual_features(X_scaled)

        self.lr_ = LogisticRegression(class_weight='balanced', C=self.C, max_iter=1000, random_state=42)
        self.lr_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self._create_manual_features(X_scaled)

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


Gen12_Candidate = Gen12b_ManualFeatures
TDEClassifier = Gen12b_ManualFeatures