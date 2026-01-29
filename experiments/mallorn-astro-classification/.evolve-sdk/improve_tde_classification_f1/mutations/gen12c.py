#!/usr/bin/env python3
"""
Gen12c: Selective Cross-Interactions Only (No Squared Terms)

Testing if removing squared terms and keeping only cross-interactions
reduces noise while maintaining discriminative power.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from itertools import combinations


class Gen12c_SelectiveInteractions(BaseEstimator, ClassifierMixin):
    """Gen 12c: Cross-interactions only, no squared terms"""

    TOP_FEATURES = ['g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis', 'r_kurtosis']

    def __init__(self, threshold: float = 0.40, C: float = 0.08):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None
        self.interaction_pairs_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12c_SelectiveInteractions':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [i for i, col in enumerate(self.feature_names_) if col in self.TOP_FEATURES]

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        if len(self.top_feature_indices_) >= 2:
            # Create only cross-interactions (no squared terms)
            self.interaction_pairs_ = list(combinations(self.top_feature_indices_, 2))

            # Generate cross-interaction features
            X_top = X_scaled[:, self.top_feature_indices_]
            interaction_features = []

            for i, j in self.interaction_pairs_:
                # Get relative indices within the top features array
                idx_i = self.top_feature_indices_.index(i)
                idx_j = self.top_feature_indices_.index(j)
                interaction = X_top[:, idx_i] * X_top[:, idx_j]
                interaction_features.append(interaction.reshape(-1, 1))

            if interaction_features:
                X_interactions = np.hstack(interaction_features)
                X_final = np.hstack([X_scaled, X_interactions])
            else:
                X_final = X_scaled
        else:
            X_final = X_scaled

        self.lr_ = LogisticRegression(class_weight='balanced', C=self.C, max_iter=1000, random_state=42)
        self.lr_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        if self.interaction_pairs_ is not None and len(self.interaction_pairs_) > 0:
            X_top = X_scaled[:, self.top_feature_indices_]
            interaction_features = []

            for i, j in self.interaction_pairs_:
                # Get relative indices within the top features array
                idx_i = self.top_feature_indices_.index(i)
                idx_j = self.top_feature_indices_.index(j)
                interaction = X_top[:, idx_i] * X_top[:, idx_j]
                interaction_features.append(interaction.reshape(-1, 1))

            if interaction_features:
                X_interactions = np.hstack(interaction_features)
                return np.hstack([X_scaled, X_interactions])

        return X_scaled

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


Gen12_Candidate = Gen12c_SelectiveInteractions
TDEClassifier = Gen12c_SelectiveInteractions