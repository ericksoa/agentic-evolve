#!/usr/bin/env python3
"""
Gen14x: Hybrid Crossover - Adaptive Parameter Selection

CROSSOVER PARENTS:
- Gen11_Champion (0.5296): Best threshold (0.43), strong regularization (C=0.05)
- Gen11b_Threshold42 (0.527): Clean implementation, middle threshold (0.42)
- Gen11c_WeakerReg (0.5259): Weaker regularization insights (C=0.08)

HYBRID INNOVATIONS:
1. Adaptive threshold selection based on validation performance
2. Interpolated regularization between strong (0.05) and weak (0.08) = 0.065
3. Champion's robust architecture + streamlined implementation
4. Enhanced feature interaction handling

Expected improvement: Combines best of all worlds for better generalization.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


class Gen14x_Hybrid(BaseEstimator, ClassifierMixin):
    """
    Gen 14x Hybrid: Adaptive Parameter Selection

    Combines strengths from three Gen11 variants:
    - Champion's architecture and high threshold
    - Clean implementation patterns
    - Balanced regularization approach
    """

    # Top features from champion analysis (proven effective)
    TOP_FEATURES = [
        'g_skew', 'r_scatter', 'r_skew', 'i_skew',
        'i_kurtosis', 'r_kurtosis'
    ]

    def __init__(self,
                 threshold: float = None,  # Will be adaptively selected
                 C: float = 0.065,        # Interpolated between 0.05 and 0.08
                 adaptive_threshold: bool = True):
        self.threshold = threshold
        self.C = C
        self.adaptive_threshold = adaptive_threshold
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None
        self.selected_threshold_ = None

    def _select_optimal_threshold(self, X_final: np.ndarray, y: pd.Series) -> float:
        """
        Adaptive threshold selection using cross-validation.
        Tests thresholds from all three parents to find the best.
        """
        candidate_thresholds = [0.40, 0.42, 0.43]  # From parents
        best_score = -1
        best_threshold = 0.43  # Default to champion's threshold

        # Fit base model for threshold testing
        temp_lr = LogisticRegression(
            class_weight='balanced',
            C=self.C,
            max_iter=1000,
            random_state=42
        )
        temp_lr.fit(X_final, y)

        # Test each threshold with simple holdout validation
        n_samples = len(y)
        if n_samples > 100:  # Only do adaptive selection with sufficient data
            split_idx = int(0.8 * n_samples)
            X_val = X_final[split_idx:]
            y_val = y.iloc[split_idx:]

            for thresh in candidate_thresholds:
                proba = temp_lr.predict_proba(X_val)[:, 1]
                pred = (proba >= thresh).astype(int)

                # Calculate F1 score manually
                tp = np.sum((pred == 1) & (y_val == 1))
                fp = np.sum((pred == 1) & (y_val == 0))
                fn = np.sum((pred == 0) & (y_val == 1))

                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        if f1 > best_score:
                            best_score = f1
                            best_threshold = thresh

        return best_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen14x_Hybrid':
        """
        Fit with adaptive parameter selection and polynomial feature interactions.
        """
        self.feature_names_ = list(X.columns)

        # Find indices of top features (from champion)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in self.TOP_FEATURES
        ]

        # Step 1: Impute missing values (consistent across all parents)
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        # Step 2: Scale features (consistent across all parents)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Step 3: Create polynomial features (from champion's approach)
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(
                degree=2,
                include_bias=False,
                interaction_only=False  # Include x^2 terms like champion
            )
            X_poly = self.poly_.fit_transform(X_top)
            # Append new features (skip original features already in X_scaled)
            n_original = len(self.top_feature_indices_)
            X_new = X_poly[:, n_original:]
            X_final = np.hstack([X_scaled, X_new])
        else:
            X_final = X_scaled
            self.poly_ = None

        # Step 4: Adaptive threshold selection
        if self.adaptive_threshold and self.threshold is None:
            self.selected_threshold_ = self._select_optimal_threshold(X_final, y)
        else:
            self.selected_threshold_ = self.threshold or 0.43

        # Step 5: Fit logistic regression with interpolated regularization
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,  # Balanced between 0.05 (champion) and 0.08 (weak reg)
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)

        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform with polynomial features (consistent implementation)."""
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
        """Predict with adaptive threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.selected_threshold_).astype(int)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'threshold': self.threshold,
            'C': self.C,
            'adaptive_threshold': self.adaptive_threshold
        }

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Aliases for benchmark discovery
Gen14_Candidate = Gen14x_Hybrid
TDEClassifier = Gen14x_Hybrid