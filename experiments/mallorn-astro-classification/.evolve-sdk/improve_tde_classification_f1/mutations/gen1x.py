#!/usr/bin/env python3
"""
Gen1X Crossover Hybrid - Multi-Strategy Ensemble

This hybrid combines multiple evolutionary strategies from the gen0_a baseline:
- Ensemble of Logistic Regression variants with different parameters
- Dynamic threshold selection based on validation performance
- Feature interaction generation for top predictors
- Combines Ridge and Logistic approaches

Key innovations:
1. Ensemble of 3 LR models with different C values (0.03, 0.05, 0.08)
2. Automatic threshold optimization on validation set
3. Top-k feature interactions for most important predictors
4. Weighted voting based on individual model confidence
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


class Gen1X_HybridEnsemble(BaseEstimator, ClassifierMixin):
    """
    Crossover hybrid combining multiple strategies:
    - Multi-LR ensemble with different regularization
    - Dynamic threshold optimization
    - Feature interactions for top predictors
    """

    def __init__(self,
                 thresholds: list = [0.38, 0.40, 0.42],
                 C_values: list = [0.03, 0.05, 0.08],
                 n_interactions: int = 5,
                 validation_split: float = 0.2):
        self.thresholds = thresholds
        self.C_values = C_values
        self.n_interactions = n_interactions
        self.validation_split = validation_split

        # Model components
        self.models_ = []
        self.ridge_model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.feature_selector_ = None
        self.poly_features_ = None
        self.best_threshold_ = 0.40
        self.feature_names_ = None
        self.model_weights_ = None

    def _create_interaction_features(self, X_scaled):
        """Create polynomial interactions for top features."""
        if self.n_interactions > 0 and X_scaled.shape[1] >= 2:
            # Select top features for interactions
            n_select = min(self.n_interactions, X_scaled.shape[1])
            self.feature_selector_ = SelectKBest(f_classif, k=n_select)
            X_top = self.feature_selector_.fit_transform(X_scaled, self.y_train_)

            # Create degree-2 interactions only for top features
            self.poly_features_ = PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False
            )
            X_interactions = self.poly_features_.fit_transform(X_top)

            # Combine original features with interactions
            return np.hstack([X_scaled, X_interactions])
        else:
            return X_scaled

    def _optimize_threshold(self, X_val, y_val):
        """Find best threshold using validation set."""
        best_f1 = 0
        best_thresh = 0.40

        # Get ensemble predictions on validation set
        val_probas = self._ensemble_predict_proba(X_val)

        for thresh in np.arange(0.35, 0.50, 0.01):
            val_preds = (val_probas >= thresh).astype(int)
            f1 = f1_score(y_val, val_preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

    def _ensemble_predict_proba(self, X_processed):
        """Get weighted ensemble predictions."""
        if not self.models_ or self.model_weights_ is None:
            return np.zeros(X_processed.shape[0])

        ensemble_proba = np.zeros(X_processed.shape[0])
        total_weight = 0

        for model, weight in zip(self.models_, self.model_weights_):
            proba = model.predict_proba(X_processed)[:, 1]
            ensemble_proba += weight * proba
            total_weight += weight

        # Add Ridge model contribution (converted to probability space)
        if self.ridge_model_ is not None:
            ridge_pred = self.ridge_model_.predict(X_processed)
            # Convert Ridge output to [0,1] range using sigmoid
            ridge_proba = 1 / (1 + np.exp(-ridge_pred))
            ensemble_proba += 0.3 * ridge_proba  # Fixed weight for Ridge
            total_weight += 0.3

        return ensemble_proba / total_weight if total_weight > 0 else ensemble_proba

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen1X_HybridEnsemble':
        """Fit ensemble with dynamic threshold optimization."""
        self.feature_names_ = list(X.columns)

        # Preprocessing pipeline
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Store for feature interaction creation
        self.y_train_ = y

        # Create interaction features
        X_final = self._create_interaction_features(X_scaled)

        # Split for threshold optimization
        if len(y) > 20:  # Only split if we have enough data
            n_val = max(1, int(len(y) * self.validation_split))
            # Stratified split to maintain class balance
            indices = np.arange(len(y))
            np.random.seed(42)
            val_idx = np.random.choice(indices[y == 1], size=min(n_val//2, sum(y)), replace=False)
            val_idx = np.concatenate([
                val_idx,
                np.random.choice(indices[y == 0], size=min(n_val//2, sum(y == 0)), replace=False)
            ])
            train_idx = np.setdiff1d(indices, val_idx)

            X_train_split = X_final[train_idx]
            y_train_split = y.iloc[train_idx]
            X_val = X_final[val_idx]
            y_val = y.iloc[val_idx]
        else:
            X_train_split = X_final
            y_train_split = y
            X_val = X_final
            y_val = y

        # Train ensemble of LR models with different C values
        self.models_ = []
        model_scores = []

        for C in self.C_values:
            model = LogisticRegression(
                class_weight='balanced',
                C=C,
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train_split, y_train_split)
            self.models_.append(model)

            # Evaluate model performance for weighting
            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= 0.40).astype(int)
            score = f1_score(y_val, val_pred, zero_division=0)
            model_scores.append(score)

        # Set model weights based on performance
        self.model_weights_ = np.array(model_scores)
        self.model_weights_ = self.model_weights_ / np.sum(self.model_weights_) if np.sum(model_scores) > 0 else np.ones(len(model_scores)) / len(model_scores)

        # Train Ridge model as additional diversity
        self.ridge_model_ = Ridge(alpha=1.0, random_state=42)
        self.ridge_model_.fit(X_train_split, y_train_split)

        # Optimize threshold
        self.best_threshold_ = self._optimize_threshold(X_val, y_val)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probabilities."""
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)

        # Apply same feature transformations as training
        if self.feature_selector_ is not None and self.poly_features_ is not None:
            X_top = self.feature_selector_.transform(X_scaled)
            X_interactions = self.poly_features_.transform(X_top)
            X_final = np.hstack([X_scaled, X_interactions])
        else:
            X_final = X_scaled

        ensemble_proba = self._ensemble_predict_proba(X_final)

        # Return as proper probability matrix
        return np.column_stack([1 - ensemble_proba, ensemble_proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using optimized threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold_).astype(int)


# Aliases for easy discovery
Gen1_Candidate = Gen1X_HybridEnsemble
TDEClassifier = Gen1X_HybridEnsemble