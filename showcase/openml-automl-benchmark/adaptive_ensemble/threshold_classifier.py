"""
Threshold-Optimized Classifier

A simpler, more generalizable approach that focuses on the single most
impactful optimization: decision threshold tuning for imbalanced data.

This is the most universally beneficial technique discovered through
our evolution experiments.
"""

import numpy as np
from typing import Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class ThresholdOptimizedClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that automatically optimizes decision threshold for F1 score.

    This is a lightweight wrapper that adds threshold optimization to any
    probabilistic classifier. It's the single most impactful optimization
    for imbalanced classification problems.

    Parameters
    ----------
    base_estimator : estimator or None, default=None
        Base classifier. If None, uses LogisticRegression with balanced weights.

    threshold_range : tuple, default=(0.20, 0.55)
        Range of thresholds to search.

    threshold_steps : int, default=15
        Number of threshold values to try.

    cv : int, default=3
        Cross-validation folds for threshold optimization.

    scale_features : bool, default=True
        Whether to standardize features before fitting.

    random_state : int or None, default=42
        Random state for reproducibility.

    Attributes
    ----------
    optimal_threshold_ : float
        Learned optimal decision threshold.

    imbalance_ratio_ : float
        Class imbalance ratio in training data.

    classes_ : ndarray
        Unique classes.

    Examples
    --------
    >>> from adaptive_ensemble import ThresholdOptimizedClassifier
    >>> clf = ThresholdOptimizedClassifier()
    >>> clf.fit(X_train, y_train)
    >>> print(f"Optimal threshold: {clf.optimal_threshold_}")
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        threshold_range: Tuple[float, float] = (0.20, 0.55),
        threshold_steps: int = 15,
        cv: int = 3,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.cv = cv
        self.scale_features = scale_features
        self.random_state = random_state

    def _get_base_estimator(self) -> BaseEstimator:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
        )

    def _compute_imbalance(self, y: np.ndarray) -> float:
        """Compute class imbalance ratio."""
        _, counts = np.unique(y, return_counts=True)
        return counts.max() / counts.min() if counts.min() > 0 else 1.0

    def _default_threshold(self, imbalance_ratio: float) -> float:
        """Compute default threshold based on imbalance."""
        if imbalance_ratio > 5.0:
            return 0.25
        elif imbalance_ratio > 3.0:
            return 0.30
        elif imbalance_ratio > 2.0:
            return 0.35
        elif imbalance_ratio > 1.5:
            return 0.40
        else:
            return 0.50

    def _optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal threshold via cross-validation."""
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.threshold_steps
        )

        default = self._default_threshold(self.imbalance_ratio_)
        best_threshold = default
        best_score = 0

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for thresh in thresholds:
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale if needed
                if self.scale_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                model = self._get_base_estimator()
                model.fit(X_train, y_train)

                proba = model.predict_proba(X_val)[:, 1]
                pred = (proba >= thresh).astype(int)
                scores.append(f1_score(y_val, pred, zero_division=0))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = thresh

        return best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThresholdOptimizedClassifier':
        """Fit the classifier with threshold optimization."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.imbalance_ratio_ = self._compute_imbalance(y)

        # Optimize threshold
        self.optimal_threshold_ = self._optimize_threshold(X, y)

        # Scale features
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        # Fit final model
        self.model_ = self._get_base_estimator()
        self.model_.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.asarray(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using optimized threshold."""
        proba = self.predict_proba(X)

        if len(self.classes_) == 2:
            return (proba[:, 1] >= self.optimal_threshold_).astype(int)
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {
            'base_estimator': self.base_estimator,
            'threshold_range': self.threshold_range,
            'threshold_steps': self.threshold_steps,
            'cv': self.cv,
            'scale_features': self.scale_features,
            'random_state': self.random_state,
        }

    def set_params(self, **params) -> 'ThresholdOptimizedClassifier':
        for key, value in params.items():
            setattr(self, key, value)
        return self
