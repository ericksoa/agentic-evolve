#!/usr/bin/env python3
"""
Gen12: Alternative Linear Models

Testing three linear model alternatives to LogReg:
1. SVM with linear kernel (hinge loss)
2. SGDClassifier (online learning)
3. RidgeClassifier (MSE-based)

All include Gen11's polynomial feature interactions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


# Top features for polynomial interactions (from Gen11)
TOP_FEATURES = [
    'g_skew', 'r_scatter', 'r_skew', 'i_skew',
    'i_kurtosis', 'r_kurtosis'
]


class Gen12_SVMLinear(BaseEstimator, ClassifierMixin):
    """
    Gen 12a: Linear SVM with polynomial interactions.

    Uses hinge loss instead of log loss (LogReg).
    Platt scaling for probability calibration.
    """

    def __init__(self, threshold: float = 0.43, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.svm_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12_SVMLinear':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in TOP_FEATURES
        ]

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Polynomial features
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_final = np.hstack([X_scaled, X_poly[:, n_original:]])
        else:
            X_final = X_scaled

        # Fit SVM
        self.svm_ = SVC(
            kernel='linear',
            C=self.C,
            class_weight='balanced',
            probability=True,
            random_state=42,
            max_iter=5000
        )
        self.svm_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            return np.hstack([X_scaled, X_poly[:, n_original:]])
        return X_scaled

    def predict_proba(self, X):
        return self.svm_.predict_proba(self._transform(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


class Gen12_SGD(BaseEstimator, ClassifierMixin):
    """
    Gen 12b: SGDClassifier with log loss.

    Online learning approach - different convergence path.
    alpha is inverse of C (regularization strength).
    """

    def __init__(self, threshold: float = 0.43, alpha: float = 0.02):
        # alpha ≈ 1/C, so alpha=0.02 ≈ C=50, but with different scale
        # For comparison: C=0.05 in LogReg → try alpha=0.01-0.1
        self.threshold = threshold
        self.alpha = alpha
        self.sgd_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12_SGD':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in TOP_FEATURES
        ]

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Polynomial features
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_final = np.hstack([X_scaled, X_poly[:, n_original:]])
        else:
            X_final = X_scaled

        # Fit SGD
        self.sgd_ = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=self.alpha,
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            tol=1e-4,
            n_jobs=-1
        )
        self.sgd_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            return np.hstack([X_scaled, X_poly[:, n_original:]])
        return X_scaled

    def predict_proba(self, X):
        return self.sgd_.predict_proba(self._transform(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


class Gen12_Ridge(BaseEstimator, ClassifierMixin):
    """
    Gen 12c: RidgeClassifier with sigmoid calibration.

    MSE-based loss instead of log loss.
    May be more robust to outliers.
    """

    def __init__(self, threshold: float = 0.43, alpha: float = 10.0):
        self.threshold = threshold
        self.alpha = alpha
        self.ridge_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12_Ridge':
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in TOP_FEATURES
        ]

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Polynomial features
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_final = np.hstack([X_scaled, X_poly[:, n_original:]])
        else:
            X_final = X_scaled

        # Fit Ridge
        self.ridge_ = RidgeClassifier(
            alpha=self.alpha,
            class_weight='balanced',
            random_state=42
        )
        self.ridge_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            return np.hstack([X_scaled, X_poly[:, n_original:]])
        return X_scaled

    def predict_proba(self, X):
        """Convert decision function to probabilities via sigmoid."""
        X_final = self._transform(X)
        decision = self.ridge_.decision_function(X_final)
        proba_positive = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba_positive, proba_positive])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases for benchmark discovery
TDEClassifier = Gen12_SVMLinear
Gen12_Candidate = Gen12_SVMLinear
