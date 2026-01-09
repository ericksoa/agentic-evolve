"""
Adaptive Ensemble Classifier - Main implementation.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings

from .analysis import DatasetAnalyzer, DatasetProfile


class AdaptiveEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Ensemble Classifier that automatically tunes itself to dataset characteristics.

    This classifier implements strategies discovered through LLM-guided evolution:
    1. Automatic threshold optimization for imbalanced data
    2. Feature selection via RFE when beneficial
    3. Diverse ensemble through different feature subsets
    4. Simple models (LogReg) that generalize well on small data

    Parameters
    ----------
    threshold : float or 'auto', default='auto'
        Decision threshold for classification. If 'auto', optimizes via CV.

    n_features : int, list of int, or 'auto', default='auto'
        Number of features to select. If 'auto', determined by dataset size.
        If list, creates ensemble with different feature counts.

    ensemble_size : int or 'auto', default='auto'
        Number of models in ensemble. If 'auto', determined by dataset size.

    base_estimator : estimator or None, default=None
        Base estimator for ensemble. If None, uses LogisticRegression.

    optimize_threshold : bool, default=True
        Whether to optimize threshold via cross-validation.

    threshold_range : tuple, default=(0.25, 0.50)
        Range for threshold search when optimize_threshold=True.

    cv : int, default=3
        Cross-validation folds for threshold optimization.

    verbose : bool, default=False
        Whether to print progress information.

    random_state : int or None, default=42
        Random state for reproducibility.

    Attributes
    ----------
    profile_ : DatasetProfile
        Learned dataset characteristics.

    optimal_threshold_ : float
        Learned optimal decision threshold.

    n_features_selected_ : list of int
        Number of features selected for each ensemble member.

    models_ : list
        Fitted ensemble models.

    feature_selectors_ : list
        Fitted RFE selectors (or None if no selection).

    scalers_ : list
        Fitted scalers for each model.

    classes_ : ndarray
        Unique classes in training data.

    Examples
    --------
    >>> from adaptive_ensemble import AdaptiveEnsembleClassifier
    >>> clf = AdaptiveEnsembleClassifier()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> print(f"Optimal threshold: {clf.optimal_threshold_}")
    """

    def __init__(
        self,
        threshold: Union[float, str] = 'auto',
        n_features: Union[int, List[int], str] = 'auto',
        ensemble_size: Union[int, str] = 'auto',
        base_estimator: Optional[BaseEstimator] = None,
        optimize_threshold: bool = True,
        threshold_range: Tuple[float, float] = (0.25, 0.50),
        cv: int = 3,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.threshold = threshold
        self.n_features = n_features
        self.ensemble_size = ensemble_size
        self.base_estimator = base_estimator
        self.optimize_threshold = optimize_threshold
        self.threshold_range = threshold_range
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state

    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[AdaptiveEnsemble] {msg}")

    def _get_base_estimator(self) -> BaseEstimator:
        """Get base estimator, using default if none provided."""
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
        )

    def _determine_feature_counts(self, n_features: int, profile: DatasetProfile) -> List[Optional[int]]:
        """Determine feature counts for ensemble members."""
        if self.n_features == 'auto':
            if profile.has_many_features:
                # Create diversity through different feature counts
                base = profile.recommended_n_features or n_features
                counts = [
                    None,  # Full features
                    max(6, base - 2),
                    base,
                ]
                # Filter to valid counts
                counts = [c if c is None or c < n_features else None for c in counts]
                return counts[:profile.recommended_ensemble_size]
            else:
                # Few features: use all features for all models
                return [None] * profile.recommended_ensemble_size
        elif isinstance(self.n_features, list):
            return self.n_features
        elif isinstance(self.n_features, int):
            return [self.n_features]
        else:
            return [None]

    def _optimize_threshold(
        self, X: np.ndarray, y: np.ndarray, profile: DatasetProfile
    ) -> float:
        """Find optimal threshold via cross-validation."""
        if not self.optimize_threshold:
            if self.threshold == 'auto':
                return profile.recommended_threshold
            return self.threshold

        self._log("Optimizing threshold...")

        # Threshold candidates
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            11
        )

        best_threshold = profile.recommended_threshold
        best_score = 0

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for thresh in thresholds:
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Quick single-model evaluation
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                model = self._get_base_estimator()
                model.fit(X_train_s, y_train)

                proba = model.predict_proba(X_val_s)[:, 1]
                pred = (proba >= thresh).astype(int)
                scores.append(f1_score(y_val, pred, zero_division=0))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = thresh

        self._log(f"Optimal threshold: {best_threshold:.3f} (F1={best_score:.4f})")
        return best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveEnsembleClassifier':
        """
        Fit the adaptive ensemble classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : AdaptiveEnsembleClassifier
            Fitted classifier.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Analyze dataset
        analyzer = DatasetAnalyzer()
        self.profile_ = analyzer.analyze(X, y)
        self._log(f"Dataset: {self.profile_.n_samples} samples, {self.profile_.n_features} features")
        self._log(f"Imbalance ratio: {self.profile_.imbalance_ratio:.2f}")

        # Store classes
        self.classes_ = np.unique(y)

        # Optimize threshold
        self.optimal_threshold_ = self._optimize_threshold(X, y, self.profile_)

        # Determine feature counts for ensemble
        feature_counts = self._determine_feature_counts(X.shape[1], self.profile_)
        self.n_features_selected_ = feature_counts
        self._log(f"Feature counts: {feature_counts}")

        # Build ensemble
        self.models_ = []
        self.scalers_ = []
        self.feature_selectors_ = []

        for i, n_feat in enumerate(feature_counts):
            self._log(f"Training model {i+1}/{len(feature_counts)} (features={n_feat or 'all'})")

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Feature selection
            if n_feat is not None and n_feat < X.shape[1]:
                rfe = RFE(
                    estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state),
                    n_features_to_select=n_feat,
                    step=1
                )
                X_selected = rfe.fit_transform(X_scaled, y)
            else:
                rfe = None
                X_selected = X_scaled

            # Train model
            model = self._get_base_estimator()
            model.fit(X_selected, y)

            self.scalers_.append(scaler)
            self.feature_selectors_.append(rfe)
            self.models_.append(model)

        self._log("Fit complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities (averaged across ensemble).
        """
        X = np.asarray(X)
        probas = []

        for scaler, rfe, model in zip(self.scalers_, self.feature_selectors_, self.models_):
            X_scaled = scaler.transform(X)
            if rfe is not None:
                X_scaled = rfe.transform(X_scaled)
            probas.append(model.predict_proba(X_scaled))

        # Average probabilities
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)

        # Binary classification with threshold
        if len(self.classes_) == 2:
            return (proba[:, 1] >= self.optimal_threshold_).astype(int)
        else:
            # Multiclass: argmax
            return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'threshold': self.threshold,
            'n_features': self.n_features,
            'ensemble_size': self.ensemble_size,
            'base_estimator': self.base_estimator,
            'optimize_threshold': self.optimize_threshold,
            'threshold_range': self.threshold_range,
            'cv': self.cv,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

    def set_params(self, **params) -> 'AdaptiveEnsembleClassifier':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> str:
        """Return human-readable summary of fitted model."""
        if not hasattr(self, 'profile_'):
            return "Model not fitted yet. Call fit() first."

        lines = [
            "AdaptiveEnsembleClassifier Summary",
            "=" * 40,
            "",
            f"Dataset: {self.profile_.n_samples} samples, {self.profile_.n_features} features",
            f"Imbalance ratio: {self.profile_.imbalance_ratio:.2f}",
            "",
            f"Learned Parameters:",
            f"  Optimal threshold: {self.optimal_threshold_:.3f}",
            f"  Ensemble size: {len(self.models_)}",
            f"  Feature counts: {self.n_features_selected_}",
            "",
            f"Strategy Used:",
            f"  Complex models: {'Yes' if self.profile_.use_complex_models else 'No (LogReg)'}",
        ]
        return "\n".join(lines)
