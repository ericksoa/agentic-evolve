"""
Threshold-Optimized Classifier (v2)

Smart threshold optimization that adapts based on model uncertainty.
Key insight: Threshold optimization helps most when the model is uncertain
(high probability overlap between classes), not when it's confident.

Improvements in v2:
- Detects model uncertainty via "overlap zone" analysis
- Skips optimization when model is already confident (saves compute)
- Widens search range for high-uncertainty datasets
- Optional probability calibration
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV


class ThresholdOptimizedClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that intelligently optimizes decision threshold for a chosen metric.

    This v3 implementation detects when threshold optimization will actually help
    by analyzing the model's probability distribution. It skips optimization when
    the model is already confident (low overlap), and searches more aggressively
    when the model is uncertain (high overlap).

    Parameters
    ----------
    base_estimator : estimator or None, default=None
        Base classifier. If None, uses LogisticRegression with balanced weights.

    optimize_for : str, default='f1'
        Metric to optimize. Options:
        - 'f1': F1 score (harmonic mean of precision and recall)
        - 'f2': F2 score (emphasizes recall over precision)
        - 'f0.5': F0.5 score (emphasizes precision over recall)
        - 'recall': Recall (sensitivity, true positive rate)
        - 'precision': Precision (positive predictive value)

    threshold_range : tuple or 'auto', default='auto'
        Range of thresholds to search. If 'auto', determined by overlap analysis:
        - High overlap (>50%): wide range (0.05, 0.60)
        - Medium overlap (20-50%): normal range (0.20, 0.55)
        - Low overlap (<20%): skip optimization, use 0.50

    threshold_steps : int, default=20
        Number of threshold values to try.

    cv : int, default=3
        Cross-validation folds for threshold optimization.

    scale_features : bool, default=True
        Whether to standardize features before fitting.

    calibrate : bool, default=False
        Whether to calibrate probabilities before threshold optimization.
        Can help when model probabilities are poorly calibrated.

    skip_if_confident : bool, default=True
        If True, skip threshold optimization when model is confident
        (overlap < 20%). Saves compute without hurting performance.

    random_state : int or None, default=42
        Random state for reproducibility.

    Attributes
    ----------
    optimal_threshold_ : float
        Learned optimal decision threshold.

    overlap_pct_ : float
        Percentage of samples in the "uncertain zone" (probs 0.3-0.7).
        Higher values indicate threshold optimization will help more.

    class_separation_ : float
        Difference in mean probability between classes.
        Lower values indicate threshold optimization will help more.

    optimization_skipped_ : bool
        Whether threshold optimization was skipped (model was confident).

    imbalance_ratio_ : float
        Class imbalance ratio in training data.

    diagnostics_ : dict
        Detailed diagnostics from the fitting process.

    classes_ : ndarray
        Unique classes.

    Examples
    --------
    >>> from adaptive_ensemble import ThresholdOptimizedClassifier
    >>> clf = ThresholdOptimizedClassifier()
    >>> clf.fit(X_train, y_train)
    >>> print(f"Overlap: {clf.overlap_pct_:.1f}%")
    >>> print(f"Optimal threshold: {clf.optimal_threshold_}")
    >>> print(f"Optimization skipped: {clf.optimization_skipped_}")
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        optimize_for: str = 'f1',
        threshold_range: Union[Tuple[float, float], str] = 'auto',
        threshold_steps: int = 20,
        cv: int = 3,
        scale_features: bool = True,
        calibrate: bool = False,
        skip_if_confident: bool = True,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.optimize_for = optimize_for
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.cv = cv
        self.scale_features = scale_features
        self.calibrate = calibrate
        self.skip_if_confident = skip_if_confident
        self.random_state = random_state

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

    def _compute_imbalance(self, y: np.ndarray) -> float:
        """Compute class imbalance ratio."""
        _, counts = np.unique(y, return_counts=True)
        return counts.max() / counts.min() if counts.min() > 0 else 1.0

    def _compute_metric(self, true_labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute the selected optimization metric."""
        metric = self.optimize_for.lower()

        if metric == 'f1':
            return f1_score(true_labels, preds, zero_division=0)
        elif metric == 'f2':
            return fbeta_score(true_labels, preds, beta=2, zero_division=0)
        elif metric == 'f0.5':
            return fbeta_score(true_labels, preds, beta=0.5, zero_division=0)
        elif metric == 'recall':
            return recall_score(true_labels, preds, zero_division=0)
        elif metric == 'precision':
            return precision_score(true_labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'f2', 'f0.5', 'recall', or 'precision'.")

    def _compute_metric_at_threshold(self, probs: np.ndarray, true_labels: np.ndarray, threshold: float) -> float:
        """Compute the selected metric at a given threshold."""
        preds = (probs >= threshold).astype(int)
        return self._compute_metric(true_labels, preds)

    def _analyze_threshold_sensitivity(self, probs: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Analyze how sensitive the metric is to threshold changes.

        Returns best threshold, metric variance, and whether optimization is worthwhile.
        """
        # Test thresholds across the range
        test_thresholds = np.linspace(0.1, 0.7, 13)
        metric_scores = [self._compute_metric_at_threshold(probs, true_labels, t) for t in test_thresholds]

        best_idx = np.argmax(metric_scores)
        best_threshold = test_thresholds[best_idx]
        best_metric = metric_scores[best_idx]
        metric_at_05 = self._compute_metric_at_threshold(probs, true_labels, 0.5)

        # Compute sensitivity metrics
        metric_variance = np.var(metric_scores)
        metric_range = max(metric_scores) - min(metric_scores)
        threshold_distance = abs(best_threshold - 0.5)
        potential_gain = (best_metric - metric_at_05) / metric_at_05 if metric_at_05 > 0 else 0

        return {
            'best_threshold': best_threshold,
            'best_metric': best_metric,
            'metric_at_05': metric_at_05,
            'metric_variance': metric_variance,
            'metric_range': metric_range,
            'threshold_distance': threshold_distance,
            'potential_gain': potential_gain,
            'metric_scores': metric_scores,
            'test_thresholds': test_thresholds,
        }

    def _analyze_uncertainty(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze model uncertainty by examining probability distributions
        AND threshold sensitivity.

        Key insight: High overlap alone doesn't mean optimization helps.
        We also need the optimal threshold to be FAR from 0.5.
        """
        all_probs = []
        all_true = []

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

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
            probs = model.predict_proba(X_val)[:, 1]

            all_probs.extend(probs)
            all_true.extend(y_val)

        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        # Compute overlap: % of samples in uncertain zone (0.3-0.7)
        in_overlap = ((all_probs >= 0.3) & (all_probs <= 0.7)).sum()
        overlap_pct = in_overlap / len(all_probs) * 100

        # Compute class separation
        class0_probs = all_probs[all_true == 0]
        class1_probs = all_probs[all_true == 1]
        class_separation = abs(class1_probs.mean() - class0_probs.mean())

        # NEW: Analyze threshold sensitivity
        sensitivity = self._analyze_threshold_sensitivity(all_probs, all_true)

        # Determine strategy based on BOTH overlap AND threshold sensitivity
        # Key insight: Only optimize if (1) optimal threshold is far from 0.5 AND (2) meaningful gain
        if overlap_pct < 20:
            # Low uncertainty: model is confident, skip optimization
            recommended_range = (0.50, 0.50)
            strategy = 'skip'
        elif sensitivity['metric_range'] < 0.02:
            # Metric is flat across thresholds - optimization won't help
            recommended_range = (0.50, 0.50)
            strategy = 'skip_flat'
        elif sensitivity['potential_gain'] < 0.01:
            # Less than 1% potential gain - not worth optimizing
            recommended_range = (0.50, 0.50)
            strategy = 'skip_low_gain'
        elif sensitivity['threshold_distance'] < 0.10:
            # Optimal threshold too close to 0.5 - not worth shifting
            recommended_range = (0.50, 0.50)
            strategy = 'skip_near_default'
        elif sensitivity['threshold_distance'] > 0.15 and sensitivity['potential_gain'] > 0.05:
            # High uncertainty AND optimal threshold far from 0.5 AND meaningful gain
            # This is the credit-g pattern
            recommended_range = (0.05, 0.60)
            strategy = 'aggressive'
        elif overlap_pct > 20:
            # Medium uncertainty: normal search
            recommended_range = (0.20, 0.55)
            strategy = 'normal'
        else:
            recommended_range = (0.50, 0.50)
            strategy = 'skip'

        return {
            'overlap_pct': overlap_pct,
            'class_separation': class_separation,
            'recommended_range': recommended_range,
            'strategy': strategy,
            'probs': all_probs,
            'true_labels': all_true,
            'class0_prob_mean': class0_probs.mean(),
            'class1_prob_mean': class1_probs.mean(),
            # Sensitivity metrics
            'sensitivity': sensitivity,
            'best_threshold_estimate': sensitivity['best_threshold'],
            'metric_range': sensitivity['metric_range'],
            'potential_gain': sensitivity['potential_gain'],
        }

    def _optimize_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Find optimal threshold via cross-validation.

        Returns (optimal_threshold, best_f1_score).
        """
        if threshold_range[0] == threshold_range[1]:
            # No range to search
            return threshold_range[0], 0.0

        thresholds = np.linspace(
            threshold_range[0],
            threshold_range[1],
            self.threshold_steps
        )

        best_threshold = 0.5
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
                scores.append(self._compute_metric(y_val, pred))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = thresh

        return best_threshold, best_score

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThresholdOptimizedClassifier':
        """
        Fit the classifier with intelligent threshold optimization.

        The fitting process:
        1. Analyze model uncertainty (overlap zone analysis)
        2. Decide whether to optimize based on uncertainty level
        3. If optimizing, use appropriate search range
        4. Fit final model with optimal threshold
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.imbalance_ratio_ = self._compute_imbalance(y)

        # Step 1: Analyze uncertainty
        uncertainty = self._analyze_uncertainty(X, y)
        self.overlap_pct_ = uncertainty['overlap_pct']
        self.class_separation_ = uncertainty['class_separation']

        # Step 2: Determine threshold range
        if self.threshold_range == 'auto':
            actual_range = uncertainty['recommended_range']
        else:
            actual_range = self.threshold_range

        # Step 3: Decide whether to skip optimization
        skip_strategies = ['skip', 'skip_flat', 'skip_low_gain', 'skip_near_default']
        if self.skip_if_confident and uncertainty['strategy'] in skip_strategies:
            self.optimization_skipped_ = True
            self.optimal_threshold_ = 0.5
            best_score = uncertainty['sensitivity']['metric_at_05']
        else:
            self.optimization_skipped_ = False
            self.optimal_threshold_, best_score = self._optimize_threshold(X, y, actual_range)

        # Store diagnostics (including sensitivity analysis)
        sensitivity = uncertainty['sensitivity']
        self.diagnostics_ = {
            'strategy': uncertainty['strategy'],
            'threshold_range_used': actual_range,
            'overlap_pct': self.overlap_pct_,
            'class_separation': self.class_separation_,
            'class0_prob_mean': uncertainty['class0_prob_mean'],
            'class1_prob_mean': uncertainty['class1_prob_mean'],
            'optimization_skipped': self.optimization_skipped_,
            'cv_best_score': best_score,
            'optimize_for': self.optimize_for,
            # Sensitivity metrics (backward compat: also store as f1_range)
            'metric_range': sensitivity['metric_range'],
            'f1_range': sensitivity['metric_range'],  # backward compat
            'potential_gain': sensitivity['potential_gain'],
            'best_threshold_estimate': sensitivity['best_threshold'],
            'threshold_distance_from_05': sensitivity['threshold_distance'],
        }

        # Step 4: Scale features
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        # Step 5: Fit final model (with optional calibration)
        base_model = self._get_base_estimator()

        if self.calibrate:
            self.model_ = CalibratedClassifierCV(
                base_model,
                cv=self.cv,
                method='isotonic'
            )
        else:
            self.model_ = base_model

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
        """Get parameters for this estimator."""
        return {
            'base_estimator': self.base_estimator,
            'optimize_for': self.optimize_for,
            'threshold_range': self.threshold_range,
            'threshold_steps': self.threshold_steps,
            'cv': self.cv,
            'scale_features': self.scale_features,
            'calibrate': self.calibrate,
            'skip_if_confident': self.skip_if_confident,
            'random_state': self.random_state,
        }

    def set_params(self, **params) -> 'ThresholdOptimizedClassifier':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self) -> str:
        """Return human-readable summary of the fitted model."""
        if not hasattr(self, 'diagnostics_'):
            return "Model not fitted yet. Call fit() first."

        d = self.diagnostics_
        metric_name = self.optimize_for.upper()
        lines = [
            "ThresholdOptimizedClassifier Summary",
            "=" * 40,
            "",
            f"Optimizing for: {metric_name}",
            "",
            f"Uncertainty Analysis:",
            f"  Overlap zone: {self.overlap_pct_:.1f}%",
            f"  Class separation: {self.class_separation_:.3f}",
            f"  {metric_name} range across thresholds: {d['metric_range']:.3f}",
            f"  Potential gain: {d['potential_gain']*100:+.1f}%",
            f"  Strategy: {d['strategy']}",
            "",
            f"Optimization:",
            f"  Skipped: {self.optimization_skipped_}",
            f"  Threshold range: {d['threshold_range_used']}",
            f"  Optimal threshold: {self.optimal_threshold_:.3f}",
            f"  CV {metric_name}: {d['cv_best_score']:.3f}",
            "",
            f"Dataset:",
            f"  Imbalance ratio: {self.imbalance_ratio_:.2f}x",
        ]
        return "\n".join(lines)
