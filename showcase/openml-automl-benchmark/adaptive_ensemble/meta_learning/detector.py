"""
Meta-learning detector for predicting threshold optimization benefit.

Uses a trained meta-model to predict whether threshold optimization
will improve performance on a given dataset.
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .extractor import MetaFeatureExtractor


class MetaLearningDetector:
    """
    Predicts whether threshold optimization will help for a dataset.

    Uses a trained meta-model (GradientBoostingClassifier) that takes
    dataset meta-features and predicts P(will_help).

    Parameters
    ----------
    model_path : str or None, default=None
        Path to pre-trained model. If None, uses default pretrained model.
    threshold : float, default=0.5
        Decision threshold for should_optimize().
    extractor : MetaFeatureExtractor or None, default=None
        Feature extractor to use. If None, creates default extractor.

    Attributes
    ----------
    model_ : GradientBoostingClassifier
        The trained meta-model.
    scaler_ : StandardScaler
        Feature scaler for normalization.
    feature_importances_ : dict
        Feature importance scores from the model.

    Examples
    --------
    >>> detector = MetaLearningDetector.load_pretrained()
    >>> features = extractor.extract(X, y)
    >>> p_help = detector.predict_proba(features)
    >>> print(f"P(will help): {p_help:.2%}")
    >>> if detector.should_optimize(features):
    ...     print("Threshold optimization recommended!")
    """

    # Default pretrained model path (relative to this file)
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), 'pretrained', 'detector_v1.pkl'
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        extractor: Optional[MetaFeatureExtractor] = None,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.extractor = extractor or MetaFeatureExtractor()

        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.feature_importances_ = None

        if model_path:
            self.load(model_path)

    def fit(
        self,
        training_data: Dict[str, np.ndarray],
        verbose: bool = False,
    ) -> 'MetaLearningDetector':
        """
        Train the meta-detector on collected training data.

        Parameters
        ----------
        training_data : dict
            Dictionary with keys:
            - 'X': Feature matrix of shape (n_datasets, n_features)
            - 'y': Binary labels (1 = optimization helped, 0 = didn't help)
            - 'feature_names': List of feature names (optional)
        verbose : bool, default=False
            Whether to print training progress.

        Returns
        -------
        self : MetaLearningDetector
            Fitted detector.
        """
        X = np.asarray(training_data['X'])
        y = np.asarray(training_data['y'])

        if 'feature_names' in training_data:
            self.feature_names_ = training_data['feature_names']
        else:
            self.feature_names_ = self.extractor.get_feature_names()

        if verbose:
            print(f"Training meta-detector on {len(X)} datasets...")
            print(f"Positive rate: {y.mean():.1%}")

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Train GradientBoostingClassifier
        # Using modest settings to avoid overfitting on small meta-dataset
        self.model_ = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model_.fit(X_scaled, y)

        # Store feature importances
        if hasattr(self.model_, 'feature_importances_'):
            importances = self.model_.feature_importances_
            self.feature_importances_ = dict(zip(self.feature_names_, importances))

        if verbose:
            print(f"Training complete. Top 5 important features:")
            top_features = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for name, imp in top_features:
                print(f"  {name}: {imp:.3f}")

        return self

    def predict_proba(self, features: Union[Dict[str, float], np.ndarray]) -> float:
        """
        Predict probability that threshold optimization will help.

        Parameters
        ----------
        features : dict or np.ndarray
            Meta-features, either as dict from extractor.extract()
            or as array from extractor.to_array().

        Returns
        -------
        probability : float
            P(threshold optimization will help) in [0, 1].
        """
        if self.model_ is None:
            raise RuntimeError("Detector not trained. Call fit() or load_pretrained().")

        # Convert dict to array if needed, filtering to expected features
        if isinstance(features, dict):
            # Filter to only the features the model was trained on
            if hasattr(self, 'feature_names_') and self.feature_names_:
                filtered_features = {k: features[k] for k in self.feature_names_ if k in features}
                X = np.array([filtered_features[k] for k in self.feature_names_]).reshape(1, -1)
            else:
                X = self.extractor.to_array(features).reshape(1, -1)
        else:
            X = np.asarray(features).reshape(1, -1)

        # Scale
        X_scaled = self.scaler_.transform(X)

        # Predict
        proba = self.model_.predict_proba(X_scaled)[0, 1]
        return float(proba)

    def should_optimize(
        self,
        features: Union[Dict[str, float], np.ndarray],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Binary decision: should we optimize threshold for this dataset?

        Parameters
        ----------
        features : dict or np.ndarray
            Meta-features from extractor.
        threshold : float or None
            Decision threshold. If None, uses self.threshold.

        Returns
        -------
        should_optimize : bool
            True if threshold optimization is recommended.
        """
        threshold = threshold if threshold is not None else self.threshold
        return self.predict_proba(features) >= threshold

    def predict_with_confidence(
        self,
        features: Union[Dict[str, float], np.ndarray],
    ) -> Dict[str, float]:
        """
        Predict with confidence information.

        Returns
        -------
        result : dict
            - 'probability': P(will_help)
            - 'recommendation': 'optimize' or 'skip'
            - 'confidence': how far from 0.5 (0 = uncertain, 0.5 = certain)
        """
        prob = self.predict_proba(features)
        recommendation = 'optimize' if prob >= self.threshold else 'skip'
        confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]

        return {
            'probability': prob,
            'recommendation': recommendation,
            'confidence': confidence,
        }

    def save(self, path: str) -> None:
        """Save detector to file."""
        state = {
            'model': self.model_,
            'scaler': self.scaler_,
            'feature_names': self.feature_names_,
            'feature_importances': self.feature_importances_,
            'threshold': self.threshold,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> 'MetaLearningDetector':
        """Load detector from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.model_ = state['model']
        self.scaler_ = state['scaler']
        self.feature_names_ = state['feature_names']
        self.feature_importances_ = state['feature_importances']
        self.threshold = state.get('threshold', 0.5)

        return self

    @classmethod
    def load_pretrained(cls) -> 'MetaLearningDetector':
        """
        Load the default pre-trained meta-detector.

        Returns
        -------
        detector : MetaLearningDetector
            Pre-trained detector ready for predictions.

        Raises
        ------
        FileNotFoundError
            If pretrained model doesn't exist.
        """
        detector = cls()
        if os.path.exists(cls.DEFAULT_MODEL_PATH):
            detector.load(cls.DEFAULT_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"Pretrained model not found at {cls.DEFAULT_MODEL_PATH}. "
                "Run the training script first: python scripts/train_meta_detector.py"
            )
        return detector

    @classmethod
    def is_pretrained_available(cls) -> bool:
        """Check if pretrained model is available."""
        return os.path.exists(cls.DEFAULT_MODEL_PATH)

    @classmethod
    def create_fallback(cls) -> 'MetaLearningDetector':
        """
        Create a rule-based fallback detector.

        This is used when no pretrained model is available. It mimics
        the hard-coded rules from ThresholdOptimizedClassifier but
        exposes them through the same API.
        """
        detector = cls()
        detector._use_fallback = True
        return detector

    def _fallback_predict(self, features: Dict[str, float]) -> float:
        """
        Fallback prediction using hard-coded rules.

        Mimics the logic from ThresholdOptimizedClassifier._analyze_uncertainty
        """
        # Extract key features
        overlap_pct = features.get('overlap_pct', 0)
        f1_range = features.get('f1_range', 0)
        potential_gain = features.get('potential_gain', 0)
        threshold_distance = features.get('threshold_distance', 0)

        # Apply rules from threshold_classifier.py
        if overlap_pct < 20:
            return 0.1  # Low probability
        if f1_range < 0.02:
            return 0.15  # Flat metric, unlikely to help
        if potential_gain < 0.01:
            return 0.2  # Low gain
        if threshold_distance < 0.10:
            return 0.25  # Threshold too close to 0.5

        # Compute a score based on gain and distance
        # Higher gain and larger distance = more likely to help
        score = 0.5 + (potential_gain * 2) + (threshold_distance * 1.5)
        return min(max(score, 0), 1)  # Clamp to [0, 1]


# Convenience function for quick predictions
def should_optimize_threshold(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Quick utility to check if threshold optimization will help.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.

    Returns
    -------
    result : dict
        - 'should_optimize': bool
        - 'probability': float
        - 'confidence': float
        - 'meta_features': dict of extracted features
    """
    extractor = MetaFeatureExtractor()
    features = extractor.extract(X, y)

    try:
        detector = MetaLearningDetector.load_pretrained()
    except FileNotFoundError:
        detector = MetaLearningDetector.create_fallback()

    if hasattr(detector, '_use_fallback') and detector._use_fallback:
        prob = detector._fallback_predict(features)
        confidence = abs(prob - 0.5) * 2
    else:
        result = detector.predict_with_confidence(features)
        prob = result['probability']
        confidence = result['confidence']

    return {
        'should_optimize': prob >= 0.5,
        'probability': prob,
        'confidence': confidence,
        'meta_features': features,
    }
