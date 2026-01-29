#!/usr/bin/env python3
"""
Gen2X Crossover Hybrid - Dual Model Ensemble with Conservative Regularization

This hybrid combines the best aspects from three parent solutions:
- gen0_a: Winning LR parameters (C=0.05, threshold=0.40) and preprocessing
- gen1a: Ridge classifier as complementary model with sigmoid probability mapping
- gen1x: Simplified ensemble concept but without complex feature engineering

Key innovations:
1. Dual-model ensemble: Logistic Regression + Ridge Classifier
2. Conservative regularization from winning gen0_a solution
3. Weighted averaging with automatic weight optimization
4. Threshold optimization from validation split
5. Simplified approach avoiding overfitting on small dataset

Philosophy: Combine complementary linear models (LR + Ridge) while maintaining
the simplicity and strong regularization that made gen0_a successful.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


class Gen2X_DualEnsemble(BaseEstimator, ClassifierMixin):
    """
    Crossover hybrid combining:
    - Logistic Regression (from gen0_a winning parameters)
    - Ridge Classifier (from gen1a with sigmoid mapping)
    - Smart ensemble weighting (simplified from gen1x)
    """

    def __init__(self,
                 lr_C: float = 0.05,           # From gen0_a best
                 ridge_alpha: float = 20.0,     # From gen1a
                 base_threshold: float = 0.40,  # From gen0_a best
                 validation_split: float = 0.15):
        self.lr_C = lr_C
        self.ridge_alpha = ridge_alpha
        self.base_threshold = base_threshold
        self.validation_split = validation_split

        # Model components
        self.lr_model_ = None
        self.ridge_model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.lr_weight_ = 0.6  # Default weights
        self.ridge_weight_ = 0.4
        self.optimized_threshold_ = base_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen2X_DualEnsemble':
        """Fit dual ensemble with automatic weight and threshold optimization."""
        self.feature_names_ = list(X.columns)

        # Preprocessing pipeline (from gen0_a)
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Split for validation if enough data
        if len(y) > 15:  # Conservative split for small datasets
            n_val = max(2, int(len(y) * self.validation_split))
            # Ensure we have at least one positive in validation
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]

            if len(pos_indices) > 1 and len(neg_indices) > 1:
                np.random.seed(42)
                val_pos = np.random.choice(pos_indices, size=min(n_val//2, len(pos_indices)//2), replace=False)
                val_neg = np.random.choice(neg_indices, size=min(n_val//2, len(neg_indices)//2), replace=False)
                val_idx = np.concatenate([val_pos, val_neg])
                train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

                X_train = X_scaled[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X_scaled[val_idx]
                y_val = y.iloc[val_idx]
                use_validation = True
            else:
                X_train = X_scaled
                y_train = y
                X_val = X_scaled
                y_val = y
                use_validation = False
        else:
            X_train = X_scaled
            y_train = y
            X_val = X_scaled
            y_val = y
            use_validation = False

        # Train Logistic Regression (gen0_a winning configuration)
        self.lr_model_ = LogisticRegression(
            class_weight='balanced',
            C=self.lr_C,
            max_iter=1000,
            random_state=42
        )
        self.lr_model_.fit(X_train, y_train)

        # Train Ridge Classifier (gen1a approach)
        self.ridge_model_ = RidgeClassifier(
            class_weight='balanced',
            alpha=self.ridge_alpha,
            random_state=42
        )
        self.ridge_model_.fit(X_train, y_train)

        # Optimize ensemble weights and threshold if we have validation data
        if use_validation and len(np.unique(y_val)) > 1:
            self._optimize_ensemble_weights(X_val, y_val)
            self._optimize_threshold(X_val, y_val)

        return self

    def _optimize_ensemble_weights(self, X_val, y_val):
        """Optimize ensemble weights based on validation performance."""
        # Get individual model predictions
        lr_proba = self.lr_model_.predict_proba(X_val)[:, 1]

        # Convert Ridge decision function to probabilities (from gen1a)
        ridge_scores = self.ridge_model_.decision_function(X_val)
        ridge_proba = 1 / (1 + np.exp(-ridge_scores))

        # Test different weight combinations
        best_f1 = 0
        best_weights = (0.6, 0.4)

        for lr_w in np.arange(0.3, 0.8, 0.1):
            ridge_w = 1.0 - lr_w
            ensemble_proba = lr_w * lr_proba + ridge_w * ridge_proba
            val_pred = (ensemble_proba >= self.base_threshold).astype(int)
            f1 = f1_score(y_val, val_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = (lr_w, ridge_w)

        self.lr_weight_, self.ridge_weight_ = best_weights

    def _optimize_threshold(self, X_val, y_val):
        """Optimize threshold using ensemble predictions."""
        ensemble_proba = self._get_ensemble_proba(X_val)

        best_f1 = 0
        best_thresh = self.base_threshold

        # Conservative threshold range around gen0_a's winning threshold
        for thresh in np.arange(0.35, 0.50, 0.01):
            val_pred = (ensemble_proba >= thresh).astype(int)
            f1 = f1_score(y_val, val_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.optimized_threshold_ = best_thresh

    def _get_ensemble_proba(self, X_scaled):
        """Get ensemble probabilities from both models."""
        # Logistic Regression probabilities
        lr_proba = self.lr_model_.predict_proba(X_scaled)[:, 1]

        # Ridge decision function converted to probabilities (from gen1a)
        ridge_scores = self.ridge_model_.decision_function(X_scaled)
        ridge_proba = 1 / (1 + np.exp(-ridge_scores))

        # Weighted ensemble
        ensemble_proba = (self.lr_weight_ * lr_proba +
                         self.ridge_weight_ * ridge_proba)

        return ensemble_proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probabilities."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        ensemble_proba = self._get_ensemble_proba(X_scaled)

        # Return as proper probability matrix
        return np.column_stack([1 - ensemble_proba, ensemble_proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using optimized threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimized_threshold_).astype(int)


# Aliases for easy discovery
Gen2X_Candidate = Gen2X_DualEnsemble
TDEClassifier = Gen2X_DualEnsemble