"""
Dataset analysis for adaptive strategy selection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DatasetProfile:
    """Profile of dataset characteristics."""
    n_samples: int
    n_features: int
    n_classes: int
    imbalance_ratio: float
    is_small: bool  # < 1000 samples
    is_very_small: bool  # < 500 samples
    is_imbalanced: bool  # ratio > 1.5
    is_highly_imbalanced: bool  # ratio > 3.0
    has_many_features: bool  # > 15 features

    # Recommended strategies
    recommended_threshold: float
    recommended_n_features: Optional[int]
    recommended_ensemble_size: int
    use_complex_models: bool


class DatasetAnalyzer:
    """
    Analyzes dataset characteristics and recommends strategies.

    Based on empirical findings from LLM-guided evolution experiments:
    - Small datasets benefit from simpler models
    - Imbalanced datasets need threshold tuning
    - High-dimensional datasets need feature selection
    """

    def __init__(self):
        self.profile_: Optional[DatasetProfile] = None

    def analyze(self, X: np.ndarray, y: np.ndarray) -> DatasetProfile:
        """
        Analyze dataset and create profile with recommendations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        DatasetProfile
            Profile with characteristics and recommendations
        """
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        # Calculate imbalance ratio (majority / minority)
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

        # Determine characteristics
        is_small = n_samples < 1000
        is_very_small = n_samples < 500
        is_imbalanced = imbalance_ratio > 1.5
        is_highly_imbalanced = imbalance_ratio > 3.0
        has_many_features = n_features > 15

        # Recommend threshold based on imbalance
        if is_highly_imbalanced:
            # Very imbalanced: lower threshold to catch more positives
            recommended_threshold = 0.30
        elif is_imbalanced:
            # Moderately imbalanced: slightly lower threshold
            recommended_threshold = 0.35
        else:
            # Balanced: default threshold
            recommended_threshold = 0.50

        # Recommend feature count based on dataset size and features
        if has_many_features:
            if is_very_small:
                recommended_n_features = min(6, n_features)
            elif is_small:
                recommended_n_features = min(8, n_features)
            else:
                recommended_n_features = min(12, n_features)
        else:
            recommended_n_features = None  # Keep all features

        # Recommend ensemble size based on dataset size
        if is_very_small:
            recommended_ensemble_size = 3
        elif is_small:
            recommended_ensemble_size = 3
        else:
            recommended_ensemble_size = 5

        # Determine if complex models are appropriate
        use_complex_models = n_samples >= 2000

        self.profile_ = DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            imbalance_ratio=imbalance_ratio,
            is_small=is_small,
            is_very_small=is_very_small,
            is_imbalanced=is_imbalanced,
            is_highly_imbalanced=is_highly_imbalanced,
            has_many_features=has_many_features,
            recommended_threshold=recommended_threshold,
            recommended_n_features=recommended_n_features,
            recommended_ensemble_size=recommended_ensemble_size,
            use_complex_models=use_complex_models,
        )

        return self.profile_

    def summary(self) -> str:
        """Return human-readable summary of analysis."""
        if self.profile_ is None:
            return "No analysis performed yet. Call analyze() first."

        p = self.profile_
        lines = [
            f"Dataset Profile:",
            f"  Samples: {p.n_samples} ({'small' if p.is_small else 'medium/large'})",
            f"  Features: {p.n_features} ({'many' if p.has_many_features else 'few'})",
            f"  Classes: {p.n_classes}",
            f"  Imbalance ratio: {p.imbalance_ratio:.2f} ({'imbalanced' if p.is_imbalanced else 'balanced'})",
            f"",
            f"Recommended Strategy:",
            f"  Threshold: {p.recommended_threshold}",
            f"  Feature selection: {p.recommended_n_features or 'None (keep all)'}",
            f"  Ensemble size: {p.recommended_ensemble_size}",
            f"  Complex models: {'Yes' if p.use_complex_models else 'No (use LogReg)'}",
        ]
        return "\n".join(lines)
