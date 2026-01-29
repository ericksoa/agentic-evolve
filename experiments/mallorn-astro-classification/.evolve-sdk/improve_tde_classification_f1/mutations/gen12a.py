#!/usr/bin/env python3
"""
Gen12a: Selective Cross-Band Feature Interactions

MUTATION: Changed from polynomial features to selective cross-band interactions
- Instead of all degree=2 combinations, manually create meaningful cross-band features
- Focus on g-r, r-i color-like interactions and skew*kurtosis within bands
- Reduces feature dimensionality while maintaining interpretability

Parent fitness: 0.5296
Hypothesis: More targeted feature engineering may improve signal-to-noise ratio
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Gen12a(BaseEstimator, ClassifierMixin):
    """
    Gen 12a: LogReg + Selective Cross-Band Interactions

    Mutation: Replace polynomial features with targeted cross-band interactions
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
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def _create_cross_band_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Create selective cross-band interaction features."""
        if len(self.top_feature_indices_) < 4:
            return X_scaled

        # Map feature names to indices for easier reference
        feature_map = {}
        for i, feat in enumerate(self.TOP_FEATURES):
            if feat in self.feature_names_:
                idx = self.feature_names_.index(feat)
                feature_map[feat] = idx

        # Create targeted interactions
        interaction_features = []

        # Cross-band skew interactions (color-like features)
        if 'g_skew' in feature_map and 'r_skew' in feature_map:
            g_skew = X_scaled[:, feature_map['g_skew']]
            r_skew = X_scaled[:, feature_map['r_skew']]
            interaction_features.append((g_skew * r_skew).reshape(-1, 1))  # g-r skew interaction

        if 'r_skew' in feature_map and 'i_skew' in feature_map:
            r_skew = X_scaled[:, feature_map['r_skew']]
            i_skew = X_scaled[:, feature_map['i_skew']]
            interaction_features.append((r_skew * i_skew).reshape(-1, 1))  # r-i skew interaction

        # Within-band shape interactions (skew * kurtosis)
        if 'r_skew' in feature_map and 'r_kurtosis' in feature_map:
            r_skew = X_scaled[:, feature_map['r_skew']]
            r_kurtosis = X_scaled[:, feature_map['r_kurtosis']]
            interaction_features.append((r_skew * r_kurtosis).reshape(-1, 1))  # r-band shape

        if 'i_skew' in feature_map and 'i_kurtosis' in feature_map:
            i_skew = X_scaled[:, feature_map['i_skew']]
            i_kurtosis = X_scaled[:, feature_map['i_kurtosis']]
            interaction_features.append((i_skew * i_kurtosis).reshape(-1, 1))  # i-band shape

        # Scatter vs skew interaction (variability vs asymmetry)
        if 'r_scatter' in feature_map and 'g_skew' in feature_map:
            r_scatter = X_scaled[:, feature_map['r_scatter']]
            g_skew = X_scaled[:, feature_map['g_skew']]
            interaction_features.append((r_scatter * g_skew).reshape(-1, 1))

        # Combine original features with interactions
        if interaction_features:
            X_interactions = np.hstack(interaction_features)
            return np.hstack([X_scaled, X_interactions])

        return X_scaled

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12a':
        """Fit with selective cross-band interactions."""
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

        # Step 3: Create selective cross-band interactions
        X_final = self._create_cross_band_features(X_scaled)

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
        """Transform with selective cross-band interactions."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        return self._create_cross_band_features(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_final = self._transform(X)
        return self.lr_.predict_proba(X_final)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases for benchmark discovery
Gen12a_Candidate = Gen12a
TDEClassifier = Gen12a