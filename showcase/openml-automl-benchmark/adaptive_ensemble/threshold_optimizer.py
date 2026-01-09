"""
ThresholdOptimizer: A wrapper that adds threshold optimization to any classifier.

This is the v9 "wrapper mode" - a simple, composable class that wraps any
sklearn-compatible binary classifier and adds threshold optimization.
"""

import numpy as np
import warnings
from typing import Optional, Union, Dict, Any

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds threshold optimization to any binary classifier.

    Takes any sklearn-compatible classifier with predict_proba() and finds
    the optimal decision threshold for your chosen metric.

    Parameters
    ----------
    estimator : estimator object
        A classifier with predict_proba() method. Will be cloned during fit.

    optimize_for : str, default='f1'
        Metric to optimize: 'f1', 'f2', 'precision', 'recall', 'balanced_accuracy'

    cv : int, default=5
        Number of cross-validation folds for threshold search.

    calibrate : bool or str, default=False
        Calibrate probabilities before threshold optimization.
        - False: No calibration
        - True or 'isotonic': Isotonic regression calibration
        - 'sigmoid' or 'platt': Platt scaling (logistic calibration)

    strategy : str, default='auto'
        When to apply threshold optimization:
        - 'auto': Use heuristics to decide if optimization will help
        - 'always': Always optimize threshold
        - 'never': Always use 0.5 (useful as baseline)

    Attributes
    ----------
    estimator_ : estimator
        The fitted wrapped estimator.

    threshold_ : float
        The optimized threshold (or 0.5 if skipped).

    classes_ : ndarray of shape (n_classes,)
        Class labels.

    optimization_skipped_ : bool
        Whether threshold optimization was skipped.

    diagnostics_ : dict
        Diagnostic information about the optimization process.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from adaptive_ensemble import ThresholdOptimizer
    >>>
    >>> # Wrap any classifier
    >>> clf = ThresholdOptimizer(RandomForestClassifier(n_estimators=100))
    >>> clf.fit(X_train, y_train)
    >>>
    >>> # Check the optimized threshold
    >>> print(f"Optimal threshold: {clf.threshold_:.3f}")
    >>>
    >>> # Make predictions
    >>> predictions = clf.predict(X_test)

    Notes
    -----
    This wrapper is designed for binary classification only. For multiclass
    problems, use the wrapped estimator directly.

    The threshold optimization uses cross-validation to avoid overfitting
    the threshold to the training data.
    """

    def __init__(
        self,
        estimator,
        optimize_for: str = 'f1',
        cv: int = 5,
        calibrate: Union[bool, str] = False,
        strategy: str = 'auto',
        safety_mode: bool = False,
        safety_margin: float = 0.02,
        compute_confidence: bool = True,
        confidence_samples: int = 100,
    ):
        self.estimator = estimator
        self.optimize_for = optimize_for
        self.cv = cv
        self.calibrate = calibrate
        self.strategy = strategy
        self.safety_mode = safety_mode
        self.safety_margin = safety_margin
        self.compute_confidence = compute_confidence
        self.confidence_samples = confidence_samples

    def fit(self, X, y):
        """
        Fit the wrapped estimator and optimize the decision threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).

        Returns
        -------
        self : ThresholdOptimizer
            Fitted estimator.
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # Check for binary classification
        if len(self.classes_) != 2:
            raise ValueError(
                f"ThresholdOptimizer only supports binary classification. "
                f"Found {len(self.classes_)} classes: {self.classes_}"
            )

        # Check estimator has predict_proba
        if not hasattr(self.estimator, 'predict_proba'):
            raise ValueError(
                f"Estimator {type(self.estimator).__name__} does not have "
                "predict_proba() method. ThresholdOptimizer requires probability "
                "predictions."
            )

        # Initialize attributes
        self.optimization_skipped_ = False
        self.diagnostics_ = {}

        # Clone estimator to avoid modifying original
        self.estimator_ = clone(self.estimator)

        # Get out-of-fold probabilities via CV
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        try:
            oof_probs = cross_val_predict(
                self.estimator_,
                X, y,
                cv=cv_splitter,
                method='predict_proba'
            )[:, 1]  # Probability of positive class
        except Exception as e:
            warnings.warn(
                f"Cross-validation failed: {e}. Using default threshold 0.5."
            )
            self.threshold_ = 0.5
            self.optimization_skipped_ = True
            self.estimator_.fit(X, y)
            return self

        # Apply calibration if requested
        if self.calibrate:
            oof_probs = self._calibrate_probabilities(X, y, oof_probs, cv_splitter)

        # Decide whether to optimize
        if self.strategy == 'never':
            self.threshold_ = 0.5
            self.optimization_skipped_ = True
            self.diagnostics_['skip_reason'] = 'strategy=never'
        elif self.strategy == 'always':
            self.threshold_ = self._find_optimal_threshold(oof_probs, y)
            self.optimization_skipped_ = False
        else:  # 'auto'
            should_optimize, reason = self._should_optimize(oof_probs, y)
            if should_optimize:
                self.threshold_ = self._find_optimal_threshold(oof_probs, y)
                self.optimization_skipped_ = False
            else:
                self.threshold_ = 0.5
                self.optimization_skipped_ = True
                self.diagnostics_['skip_reason'] = reason

        # Store optimization details
        self.diagnostics_['oof_probs_mean'] = float(np.mean(oof_probs))
        self.diagnostics_['oof_probs_std'] = float(np.std(oof_probs))
        self.diagnostics_['strategy'] = self.strategy

        # Compute confidence intervals if requested
        if self.compute_confidence and not self.optimization_skipped_:
            self.threshold_confidence_ = self._compute_confidence_intervals(
                oof_probs, y
            )
        else:
            # Default confidence for skipped optimization
            self.threshold_confidence_ = {
                'point_estimate': self.threshold_,
                'ci_low': self.threshold_,
                'ci_high': self.threshold_,
                'std': 0.0,
                'confidence': 1.0,
                'bootstrap_thresholds': [self.threshold_],
            }

        # Store for plotting
        self._oof_probs = oof_probs
        self._oof_labels = y

        # Compute operating points for v8 features
        self.operating_points_ = self._compute_operating_points(oof_probs, y)

        # Fit final estimator on all data
        if self.calibrate:
            method = 'isotonic' if self.calibrate is True else self.calibrate
            if method in ('sigmoid', 'platt'):
                method = 'sigmoid'
            self.estimator_ = CalibratedClassifierCV(
                clone(self.estimator),
                method=method,
                cv=self.cv
            )
        else:
            self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict class labels using the optimized threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['estimator_', 'threshold_'])
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)

    def predict_proba(self, X):
        """
        Return probability estimates from the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of each class.
        """
        check_is_fitted(self, ['estimator_'])
        X = check_array(X)
        return self.estimator_.predict_proba(X)

    def _find_optimal_threshold(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        n_thresholds: int = 100,
    ) -> float:
        """Find threshold that maximizes the target metric."""
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_score = -1
        best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)

            if self.optimize_for == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.optimize_for == 'f2':
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif self.optimize_for == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif self.optimize_for == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif self.optimize_for == 'balanced_accuracy':
                # balanced accuracy = (sensitivity + specificity) / 2
                tp = np.sum((y_pred == 1) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                raise ValueError(f"Unknown metric: {self.optimize_for}")

            if score > best_score:
                best_score = score
                best_threshold = thresh

        # Store scores for diagnostics
        self.diagnostics_['best_score'] = float(best_score)
        self.diagnostics_['default_score'] = float(self._score_at_threshold(probs, y_true, 0.5))
        self.diagnostics_['improvement'] = float(best_score - self.diagnostics_['default_score'])

        return best_threshold

    def _score_at_threshold(self, probs: np.ndarray, y_true: np.ndarray, threshold: float) -> float:
        """Calculate score at a specific threshold."""
        y_pred = (probs >= threshold).astype(int)
        if self.optimize_for == 'f1':
            return f1_score(y_true, y_pred, zero_division=0)
        elif self.optimize_for == 'f2':
            return fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif self.optimize_for == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif self.optimize_for == 'recall':
            return recall_score(y_true, y_pred, zero_division=0)
        else:
            return f1_score(y_true, y_pred, zero_division=0)

    def _should_optimize(self, probs: np.ndarray, y_true: np.ndarray) -> tuple:
        """
        Heuristic check: will threshold optimization likely help?

        Returns (should_optimize, reason).
        """
        # Compute some basic statistics
        pos_probs = probs[y_true == 1]
        neg_probs = probs[y_true == 0]

        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return False, "single_class_in_data"

        # Check class separation
        pos_mean = np.mean(pos_probs)
        neg_mean = np.mean(neg_probs)
        separation = pos_mean - neg_mean

        if separation > 0.6:
            # Classes are well-separated, threshold doesn't matter much
            return False, "well_separated_classes"

        # Check if probabilities cluster around 0.5
        near_boundary = np.mean((probs > 0.3) & (probs < 0.7))
        if near_boundary < 0.1:
            # Most predictions are confident, threshold won't change much
            return False, "confident_predictions"

        # Check potential gain
        best_thresh = self._find_optimal_threshold(probs, y_true)
        best_score = self.diagnostics_.get('best_score', 0)
        default_score = self.diagnostics_.get('default_score', 0)
        gain = best_score - default_score

        if gain < 0.01:
            # Less than 1% improvement
            return False, "minimal_gain"

        # Check if optimal threshold is close to 0.5
        if abs(best_thresh - 0.5) < 0.05:
            return False, "threshold_near_default"

        return True, "optimization_recommended"

    def _calibrate_probabilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probs: np.ndarray,
        cv_splitter,
    ) -> np.ndarray:
        """Apply probability calibration."""
        method = 'isotonic' if self.calibrate is True else self.calibrate
        if method in ('sigmoid', 'platt'):
            method = 'sigmoid'

        # Use CalibratedClassifierCV to get calibrated OOF predictions
        calibrated = CalibratedClassifierCV(
            clone(self.estimator),
            method=method,
            cv=cv_splitter
        )

        try:
            calibrated_probs = cross_val_predict(
                calibrated,
                X, y,
                cv=cv_splitter,
                method='predict_proba'
            )[:, 1]
            return calibrated_probs
        except Exception:
            # Fallback to original probs if calibration fails
            return probs

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'estimator': self.estimator,
            'optimize_for': self.optimize_for,
            'cv': self.cv,
            'calibrate': self.calibrate,
            'strategy': self.strategy,
            'safety_mode': self.safety_mode,
            'safety_margin': self.safety_margin,
            'compute_confidence': self.compute_confidence,
            'confidence_samples': self.confidence_samples,
        }
        if deep and hasattr(self.estimator, 'get_params'):
            estimator_params = self.estimator.get_params(deep=True)
            for key, value in estimator_params.items():
                params[f'estimator__{key}'] = value
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        estimator_params = {}
        for key, value in list(params.items()):
            if key.startswith('estimator__'):
                estimator_params[key[11:]] = value
                del params[key]

        # Set own params
        for key, value in params.items():
            setattr(self, key, value)

        # Set estimator params
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)

        return self

    def __repr__(self):
        estimator_name = type(self.estimator).__name__
        return (
            f"ThresholdOptimizer(estimator={estimator_name}(), "
            f"optimize_for='{self.optimize_for}', cv={self.cv}, "
            f"strategy='{self.strategy}')"
        )

    # =========================================================================
    # v7 Features: Trust & Transparency
    # =========================================================================

    def _compute_confidence_intervals(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict:
        """
        Compute bootstrap confidence intervals for the optimal threshold.

        Returns dict with point_estimate, ci_low, ci_high, std, confidence.
        """
        n_samples = len(probs)
        bootstrap_thresholds = []

        for _ in range(self.confidence_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_probs = probs[indices]
            boot_y = y_true[indices]

            # Find optimal threshold for this sample
            thresh = self._find_optimal_threshold_simple(boot_probs, boot_y)
            bootstrap_thresholds.append(thresh)

        bootstrap_thresholds = np.array(bootstrap_thresholds)

        # Compute statistics
        ci_low = np.percentile(bootstrap_thresholds, 2.5)
        ci_high = np.percentile(bootstrap_thresholds, 97.5)
        std = np.std(bootstrap_thresholds)

        # Confidence: how far from uncertain 0.5 (normalized to [0, 1])
        point_estimate = self.threshold_
        confidence = min(1.0, abs(point_estimate - 0.5) * 2)

        return {
            'point_estimate': point_estimate,
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'std': float(std),
            'confidence': float(confidence),
            'bootstrap_thresholds': bootstrap_thresholds.tolist(),
        }

    def _find_optimal_threshold_simple(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        n_thresholds: int = 50,
    ) -> float:
        """Find optimal threshold without storing diagnostics (for bootstrap)."""
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_score = -1
        best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            if self.optimize_for == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.optimize_for == 'f2':
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif self.optimize_for == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif self.optimize_for == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold

    def explain(self) -> str:
        """
        Return a human-readable explanation of the threshold optimization.

        Returns
        -------
        explanation : str
            Multi-line explanation of what happened and why.
        """
        if not hasattr(self, 'threshold_'):
            return "ThresholdOptimizer has not been fitted yet. Call fit() first."

        lines = []
        lines.append("=" * 60)
        lines.append("THRESHOLD OPTIMIZATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Decision summary
        if self.optimization_skipped_:
            lines.append("DECISION: SKIP threshold optimization")
            reason = self.diagnostics_.get('skip_reason', 'unknown')
            lines.append(f"Reason: {reason}")
            lines.append(f"Using default threshold: {self.threshold_:.3f}")
        else:
            lines.append("DECISION: OPTIMIZE threshold")
            lines.append(f"Optimal threshold: {self.threshold_:.3f}")
            improvement = self.diagnostics_.get('improvement', 0)
            lines.append(f"Expected improvement: {improvement:+.1%}")

        lines.append("")

        # Confidence information
        if hasattr(self, 'threshold_confidence_'):
            ci = self.threshold_confidence_
            lines.append("CONFIDENCE:")
            lines.append(f"  95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
            lines.append(f"  Std dev: {ci['std']:.3f}")

            if ci['std'] < 0.05:
                lines.append("  Confidence level: HIGH (narrow CI)")
            elif ci['std'] < 0.10:
                lines.append("  Confidence level: MEDIUM")
            else:
                lines.append("  Confidence level: LOW (wide CI - use with caution)")

        lines.append("")

        # Diagnostics
        lines.append("DIAGNOSTICS:")
        lines.append(f"  Strategy: {self.diagnostics_.get('strategy', 'unknown')}")
        lines.append(f"  Probability mean: {self.diagnostics_.get('oof_probs_mean', 0):.3f}")
        lines.append(f"  Probability std: {self.diagnostics_.get('oof_probs_std', 0):.3f}")

        if 'best_score' in self.diagnostics_:
            lines.append(f"  Best {self.optimize_for}: {self.diagnostics_['best_score']:.3f}")
            lines.append(f"  Default {self.optimize_for}: {self.diagnostics_['default_score']:.3f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def plot(self, figsize=(10, 6), show=True):
        """
        Plot the metric vs threshold curve with confidence interval.

        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size.
        show : bool, default=True
            Whether to display the plot. If False, returns the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object if show=False, otherwise None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        if not hasattr(self, 'threshold_'):
            raise RuntimeError("ThresholdOptimizer has not been fitted yet.")

        if not hasattr(self, '_oof_probs'):
            raise RuntimeError("No probability data available for plotting.")

        # Compute metric scores at different thresholds
        thresholds = np.linspace(0.01, 0.99, 100)
        scores = []
        for thresh in thresholds:
            y_pred = (self._oof_probs >= thresh).astype(int)
            if self.optimize_for == 'f1':
                score = f1_score(self._oof_labels, y_pred, zero_division=0)
            elif self.optimize_for == 'f2':
                score = fbeta_score(self._oof_labels, y_pred, beta=2, zero_division=0)
            elif self.optimize_for == 'precision':
                score = precision_score(self._oof_labels, y_pred, zero_division=0)
            elif self.optimize_for == 'recall':
                score = recall_score(self._oof_labels, y_pred, zero_division=0)
            else:
                score = f1_score(self._oof_labels, y_pred, zero_division=0)
            scores.append(score)

        scores = np.array(scores)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot metric curve
        ax.plot(thresholds, scores, 'b-', linewidth=2, label=f'{self.optimize_for.upper()} score')

        # Plot confidence interval band if available
        if hasattr(self, 'threshold_confidence_'):
            ci = self.threshold_confidence_
            ax.axvspan(ci['ci_low'], ci['ci_high'], alpha=0.2, color='blue',
                      label='95% CI')

        # Plot optimal threshold
        ax.axvline(self.threshold_, color='red', linestyle='-', linewidth=2,
                  label=f'Optimal threshold ({self.threshold_:.3f})')

        # Plot default threshold
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1,
                  label='Default (0.5)')

        # Add improvement annotation
        if 'improvement' in self.diagnostics_ and self.diagnostics_['improvement'] > 0:
            improvement = self.diagnostics_['improvement']
            ax.annotate(
                f'+{improvement:.1%}',
                xy=(self.threshold_, scores[np.argmin(np.abs(thresholds - self.threshold_))]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, color='green', fontweight='bold'
            )

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel(f'{self.optimize_for.upper()} Score', fontsize=12)
        ax.set_title('Threshold Optimization Curve', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if show:
            plt.show()
            return None
        else:
            return fig

    # =========================================================================
    # v8 Features: Operating Point Selection (Pareto Frontier)
    # =========================================================================

    def _compute_operating_points(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        n_thresholds: int = 50,
    ) -> Dict:
        """
        Compute operating points (precision, recall, etc.) at various thresholds.

        Returns dict with thresholds, metrics, and Pareto frontier mask.
        """
        from sklearn.metrics import precision_recall_fscore_support

        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        precisions = []
        recalls = []
        f1_scores = []
        f2_scores = []
        fprs = []
        specificities = []

        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)

            # Compute metrics
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

            # Compute confusion matrix elements for FPR/specificity
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            f2_scores.append(f2)
            fprs.append(fpr)
            specificities.append(specificity)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)
        f2_scores = np.array(f2_scores)
        fprs = np.array(fprs)
        specificities = np.array(specificities)

        # Find Pareto frontier (precision vs recall)
        pareto_mask = self._find_pareto_frontier(precisions, recalls)

        # Find the index of current threshold
        selected_index = np.argmin(np.abs(thresholds - self.threshold_))

        return {
            'thresholds': thresholds,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'f2_scores': f2_scores,
            'fprs': fprs,
            'specificities': specificities,
            'pareto_mask': pareto_mask,
            'selected_index': selected_index,
            'selection_method': 'threshold_optimization',
        }

    def _find_pareto_frontier(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
    ) -> np.ndarray:
        """
        Find the Pareto frontier (non-dominated points) for precision vs recall.

        Returns boolean mask where True indicates a Pareto-optimal point.
        """
        n_points = len(precisions)
        pareto_mask = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Point j dominates point i if it's better in both metrics
                    if (precisions[j] >= precisions[i] and recalls[j] >= recalls[i] and
                        (precisions[j] > precisions[i] or recalls[j] > recalls[i])):
                        pareto_mask[i] = False
                        break

        return pareto_mask

    def set_operating_point(
        self,
        min_recall: float = None,
        min_precision: float = None,
        max_fpr: float = None,
        target_f1: float = None,
        target_f2: float = None,
        threshold: float = None,
    ) -> 'ThresholdOptimizer':
        """
        Select an operating point based on business constraints.

        Parameters
        ----------
        min_recall : float, optional
            Minimum required recall. Finds best precision at this recall level.
        min_precision : float, optional
            Minimum required precision. Finds best recall at this precision level.
        max_fpr : float, optional
            Maximum allowed false positive rate.
        target_f1 : float, optional
            Target F1 score. Finds closest operating point.
        target_f2 : float, optional
            Target F2 score. Finds closest operating point.
        threshold : float, optional
            Set threshold directly (0 to 1).

        Returns
        -------
        self : ThresholdOptimizer
            Returns self for method chaining.

        Examples
        --------
        >>> clf.set_operating_point(min_recall=0.95)  # Can't miss positives
        >>> clf.set_operating_point(min_precision=0.90)  # Can't have false alarms
        >>> clf.set_operating_point(max_fpr=0.05)  # Limit false positive rate
        """
        if not hasattr(self, 'operating_points_'):
            raise RuntimeError(
                "Operating points not computed. Call fit() first."
            )

        ops = self.operating_points_

        # Direct threshold setting
        if threshold is not None:
            idx = np.argmin(np.abs(ops['thresholds'] - threshold))
            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'threshold={threshold}'
            return self

        # Constraint-based selection
        if min_recall is not None:
            # Find points meeting recall constraint
            valid = ops['recalls'] >= min_recall
            if not np.any(valid):
                warnings.warn(
                    f"No operating point meets min_recall={min_recall}. "
                    "Using highest recall point."
                )
                idx = np.argmax(ops['recalls'])
            else:
                # Best precision among valid points
                valid_idx = np.where(valid)[0]
                best_in_valid = np.argmax(ops['precisions'][valid])
                idx = valid_idx[best_in_valid]

            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'min_recall={min_recall}'
            return self

        if min_precision is not None:
            # Find points meeting precision constraint
            valid = ops['precisions'] >= min_precision
            if not np.any(valid):
                warnings.warn(
                    f"No operating point meets min_precision={min_precision}. "
                    "Using highest precision point."
                )
                idx = np.argmax(ops['precisions'])
            else:
                # Best recall among valid points
                valid_idx = np.where(valid)[0]
                best_in_valid = np.argmax(ops['recalls'][valid])
                idx = valid_idx[best_in_valid]

            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'min_precision={min_precision}'
            return self

        if max_fpr is not None:
            # Find points meeting FPR constraint
            valid = ops['fprs'] <= max_fpr
            if not np.any(valid):
                warnings.warn(
                    f"No operating point meets max_fpr={max_fpr}. "
                    "Using lowest FPR point."
                )
                idx = np.argmin(ops['fprs'])
            else:
                # Best recall among valid points
                valid_idx = np.where(valid)[0]
                best_in_valid = np.argmax(ops['recalls'][valid])
                idx = valid_idx[best_in_valid]

            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'max_fpr={max_fpr}'
            return self

        if target_f1 is not None:
            # Find closest to target F1
            idx = np.argmin(np.abs(ops['f1_scores'] - target_f1))
            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'target_f1={target_f1}'
            return self

        if target_f2 is not None:
            # Find closest to target F2
            idx = np.argmin(np.abs(ops['f2_scores'] - target_f2))
            self.threshold_ = ops['thresholds'][idx]
            ops['selected_index'] = idx
            ops['selection_method'] = f'target_f2={target_f2}'
            return self

        raise ValueError(
            "Must specify at least one constraint: min_recall, min_precision, "
            "max_fpr, target_f1, target_f2, or threshold"
        )

    def get_operating_point(self) -> Dict:
        """
        Get the current operating point details.

        Returns
        -------
        point : dict
            Dictionary with threshold, precision, recall, f1, f2, fpr, specificity.
        """
        if not hasattr(self, 'operating_points_'):
            raise RuntimeError(
                "Operating points not computed. Call fit() first."
            )

        ops = self.operating_points_
        idx = ops['selected_index']

        return {
            'threshold': float(ops['thresholds'][idx]),
            'precision': float(ops['precisions'][idx]),
            'recall': float(ops['recalls'][idx]),
            'f1': float(ops['f1_scores'][idx]),
            'f2': float(ops['f2_scores'][idx]),
            'fpr': float(ops['fprs'][idx]),
            'specificity': float(ops['specificities'][idx]),
            'is_pareto_optimal': bool(ops['pareto_mask'][idx]),
            'selection_method': ops['selection_method'],
        }

    def list_operating_points(self, pareto_only: bool = False):
        """
        List all operating points as a DataFrame.

        Parameters
        ----------
        pareto_only : bool, default=False
            If True, only return Pareto-optimal points.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with threshold, precision, recall, f1, f2, fpr columns.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for list_operating_points(). "
                "Install it with: pip install pandas"
            )

        if not hasattr(self, 'operating_points_'):
            raise RuntimeError(
                "Operating points not computed. Call fit() first."
            )

        ops = self.operating_points_

        df = pd.DataFrame({
            'threshold': ops['thresholds'],
            'precision': ops['precisions'],
            'recall': ops['recalls'],
            'f1': ops['f1_scores'],
            'f2': ops['f2_scores'],
            'fpr': ops['fprs'],
            'specificity': ops['specificities'],
            'is_pareto_optimal': ops['pareto_mask'],
        })

        if pareto_only:
            df = df[df['is_pareto_optimal']].reset_index(drop=True)

        return df

    def plot_operating_points(self, figsize=(10, 8), show=True):
        """
        Plot the Pareto frontier of operating points.

        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size.
        show : bool, default=True
            Whether to display the plot.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure if show=False.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        if not hasattr(self, 'operating_points_'):
            raise RuntimeError(
                "Operating points not computed. Call fit() first."
            )

        ops = self.operating_points_
        idx = ops['selected_index']

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all points (dominated in gray)
        dominated = ~ops['pareto_mask']
        ax.scatter(
            ops['recalls'][dominated],
            ops['precisions'][dominated],
            c='lightgray', s=30, alpha=0.5, label='Dominated'
        )

        # Plot Pareto frontier
        pareto_idx = np.where(ops['pareto_mask'])[0]
        # Sort by recall for line plot
        sorted_idx = pareto_idx[np.argsort(ops['recalls'][pareto_idx])]
        ax.plot(
            ops['recalls'][sorted_idx],
            ops['precisions'][sorted_idx],
            'b-', linewidth=2, label='Pareto Frontier'
        )
        ax.scatter(
            ops['recalls'][ops['pareto_mask']],
            ops['precisions'][ops['pareto_mask']],
            c='blue', s=50, zorder=5
        )

        # Plot current operating point
        ax.scatter(
            [ops['recalls'][idx]],
            [ops['precisions'][idx]],
            c='red', s=200, marker='*', zorder=10,
            label=f'Selected (t={ops["thresholds"][idx]:.2f})'
        )

        # Add iso-F1 curves
        for f1_target in [0.2, 0.4, 0.6, 0.8]:
            recall_range = np.linspace(0.01, 0.99, 100)
            precision_for_f1 = (f1_target * recall_range) / (2 * recall_range - f1_target)
            valid = (precision_for_f1 > 0) & (precision_for_f1 <= 1)
            ax.plot(
                recall_range[valid],
                precision_for_f1[valid],
                'k--', alpha=0.2, linewidth=0.5
            )
            # Label the curve
            if np.any(valid):
                mid_idx = len(recall_range[valid]) // 2
                ax.annotate(
                    f'F1={f1_target}',
                    xy=(recall_range[valid][mid_idx], precision_for_f1[valid][mid_idx]),
                    fontsize=8, alpha=0.5
                )

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Operating Points (Pareto Frontier)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if show:
            plt.show()
            return None
        else:
            return fig
