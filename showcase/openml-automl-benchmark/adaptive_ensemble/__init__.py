"""
Adaptive Ensemble Classifier

A sklearn-compatible classifier that automatically adapts to dataset characteristics:
- Detects class imbalance and optimizes decision threshold
- Applies feature selection based on feature count
- Builds diverse ensemble based on dataset size
- Uses simple models that generalize well on small data

Usage:
    from adaptive_ensemble import AdaptiveEnsembleClassifier

    clf = AdaptiveEnsembleClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Access learned parameters
    print(f"Optimal threshold: {clf.optimal_threshold_}")
    print(f"Selected features: {clf.n_features_selected_}")
"""

from .classifier import AdaptiveEnsembleClassifier
from .threshold_classifier import ThresholdOptimizedClassifier
from .analysis import DatasetAnalyzer

# Meta-learning components (optional import, may not be available)
try:
    from .meta_learning import MetaFeatureExtractor, MetaLearningDetector
    _HAS_META_LEARNING = True
except ImportError:
    _HAS_META_LEARNING = False
    MetaFeatureExtractor = None
    MetaLearningDetector = None

__version__ = "0.1.0"
__all__ = [
    "AdaptiveEnsembleClassifier",
    "ThresholdOptimizedClassifier",
    "DatasetAnalyzer",
    "MetaFeatureExtractor",
    "MetaLearningDetector",
]
