"""
Meta-feature extraction for predicting threshold optimization benefit.

Extracts ~20 meta-features from a dataset in ~0.5-2 seconds using a quick
probe model (3-fold CV with LogisticRegression).
"""

import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats


class MetaFeatureExtractor:
    """
    Extract meta-features from (X, y) for meta-learning.

    Uses a quick probe model (LogisticRegression with 3-fold CV) to compute
    probability distribution features. Total extraction time is typically
    0.5-2 seconds per dataset.

    Parameters
    ----------
    cv : int, default=3
        Cross-validation folds for probe model.
    random_state : int, default=42
        Random state for reproducibility.

    Examples
    --------
    >>> extractor = MetaFeatureExtractor()
    >>> features = extractor.extract(X, y)
    >>> print(features['overlap_pct'])
    45.2
    >>> print(features['potential_gain'])
    0.15
    """

    def __init__(self, cv: int = 3, random_state: int = 42):
        self.cv = cv
        self.random_state = random_state

    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Extract meta-features from dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        features : dict
            Dictionary of meta-feature names to values.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        features = {}

        # ========== Dataset Features ==========
        features.update(self._extract_dataset_features(X, y))

        # ========== Probability Distribution Features ==========
        # Run quick probe model to get probability distribution
        probs, true_labels = self._run_probe_model(X, y)
        features.update(self._extract_prob_features(probs, true_labels))

        # ========== Threshold Sensitivity Features ==========
        features.update(self._extract_sensitivity_features(probs, true_labels))

        # ========== Derived/Interaction Features ==========
        features.update(self._extract_derived_features(features))

        return features

    def _extract_dataset_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Extract basic dataset characteristics."""
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        # Imbalance ratio
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else 1.0

        # Minority class ratio
        minority_ratio = counts.min() / counts.sum()

        # Feature statistics
        # Handle constant features
        with np.errstate(all='ignore'):
            feature_means = np.nanmean(X, axis=0)
            feature_stds = np.nanstd(X, axis=0)
            # Replace NaN with 0
            feature_means = np.nan_to_num(feature_means, nan=0.0)
            feature_stds = np.nan_to_num(feature_stds, nan=0.0)

        # Log-transform for scale
        log_n_samples = np.log10(n_samples + 1)
        log_n_features = np.log10(n_features + 1)

        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'imbalance_ratio': imbalance_ratio,
            'minority_ratio': minority_ratio,
            'log_n_samples': log_n_samples,
            'log_n_features': log_n_features,
            'samples_per_feature': n_samples / max(n_features, 1),
            'feature_mean_of_means': np.mean(feature_means),
            'feature_mean_of_stds': np.mean(feature_stds),
        }

    def _run_probe_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Run a quick probe model to get probability distribution.

        Uses LogisticRegression with 3-fold CV for speed (~0.5-2s).
        """
        all_probs = []
        all_true = []

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Quick probe model
            model = LogisticRegression(
                C=0.5,
                class_weight='balanced',
                max_iter=500,
                random_state=self.random_state,
                solver='lbfgs',
            )

            try:
                model.fit(X_train_scaled, y_train)
                probs = model.predict_proba(X_val_scaled)
                # Handle binary and multiclass
                if probs.shape[1] == 2:
                    probs = probs[:, 1]  # Probability of positive class
                else:
                    # For multiclass, use max probability
                    probs = np.max(probs, axis=1)
            except Exception:
                # If model fails, use uniform probabilities
                probs = np.full(len(y_val), 0.5)

            all_probs.extend(probs)
            all_true.extend(y_val)

        return np.array(all_probs), np.array(all_true)

    def _extract_prob_features(self, probs: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Extract probability distribution features."""
        # Only compute for binary classification
        unique_labels = np.unique(true_labels)
        if len(unique_labels) != 2:
            return self._get_default_prob_features()

        # Overlap zone: % of samples with uncertain predictions (0.3-0.7)
        in_overlap = ((probs >= 0.3) & (probs <= 0.7)).sum()
        overlap_pct = in_overlap / len(probs) * 100

        # Class separation
        class0_probs = probs[true_labels == unique_labels[0]]
        class1_probs = probs[true_labels == unique_labels[1]]
        class_separation = abs(class1_probs.mean() - class0_probs.mean())

        # Probability statistics
        prob_mean = probs.mean()
        prob_std = probs.std()
        prob_skewness = stats.skew(probs) if len(probs) > 2 else 0.0
        prob_kurtosis = stats.kurtosis(probs) if len(probs) > 2 else 0.0

        # Probability percentiles
        prob_p10 = np.percentile(probs, 10)
        prob_p25 = np.percentile(probs, 25)
        prob_p50 = np.percentile(probs, 50)
        prob_p75 = np.percentile(probs, 75)
        prob_p90 = np.percentile(probs, 90)

        # Interquartile range
        prob_iqr = prob_p75 - prob_p25

        return {
            'overlap_pct': overlap_pct,
            'class_separation': class_separation,
            'prob_mean': prob_mean,
            'prob_std': prob_std,
            'prob_skewness': prob_skewness,
            'prob_kurtosis': prob_kurtosis,
            'prob_p10': prob_p10,
            'prob_p25': prob_p25,
            'prob_p50': prob_p50,
            'prob_p75': prob_p75,
            'prob_p90': prob_p90,
            'prob_iqr': prob_iqr,
        }

    def _get_default_prob_features(self) -> Dict[str, float]:
        """Return default values for probability features (for non-binary case)."""
        return {
            'overlap_pct': 0.0,
            'class_separation': 1.0,
            'prob_mean': 0.5,
            'prob_std': 0.0,
            'prob_skewness': 0.0,
            'prob_kurtosis': 0.0,
            'prob_p10': 0.5,
            'prob_p25': 0.5,
            'prob_p50': 0.5,
            'prob_p75': 0.5,
            'prob_p90': 0.5,
            'prob_iqr': 0.0,
        }

    def _extract_sensitivity_features(self, probs: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Extract threshold sensitivity features."""
        # Only compute for binary classification
        unique_labels = np.unique(true_labels)
        if len(unique_labels) != 2:
            return self._get_default_sensitivity_features()

        # Test thresholds across the range
        test_thresholds = np.linspace(0.1, 0.7, 13)
        f1_scores = []

        for thresh in test_thresholds:
            preds = (probs >= thresh).astype(int)
            # Map predictions to actual labels
            pred_labels = np.where(preds == 1, unique_labels[1], unique_labels[0])
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            f1_scores.append(f1)

        f1_scores = np.array(f1_scores)

        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = test_thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        # F1 at default threshold
        preds_05 = (probs >= 0.5).astype(int)
        pred_labels_05 = np.where(preds_05 == 1, unique_labels[1], unique_labels[0])
        f1_at_05 = f1_score(true_labels, pred_labels_05, zero_division=0)

        # Sensitivity metrics
        f1_range = f1_scores.max() - f1_scores.min()
        f1_std = f1_scores.std()
        threshold_distance = abs(best_threshold - 0.5)

        # Potential gain from optimization
        potential_gain = (best_f1 - f1_at_05) / f1_at_05 if f1_at_05 > 0 else 0.0

        # How much F1 drops when moving away from optimal
        # Average F1 drop at edges vs optimal
        edge_f1 = (f1_scores[0] + f1_scores[-1]) / 2
        f1_drop_at_edges = best_f1 - edge_f1

        return {
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'f1_at_05': f1_at_05,
            'f1_range': f1_range,
            'f1_std': f1_std,
            'threshold_distance': threshold_distance,
            'potential_gain': potential_gain,
            'f1_drop_at_edges': f1_drop_at_edges,
        }

    def _get_default_sensitivity_features(self) -> Dict[str, float]:
        """Return default values for sensitivity features (for non-binary case)."""
        return {
            'best_threshold': 0.5,
            'best_f1': 0.0,
            'f1_at_05': 0.0,
            'f1_range': 0.0,
            'f1_std': 0.0,
            'threshold_distance': 0.0,
            'potential_gain': 0.0,
            'f1_drop_at_edges': 0.0,
        }

    def _extract_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract derived/interaction features."""
        # Interactions that may indicate threshold optimization benefit
        derived = {}

        # Imbalance * overlap (high values suggest benefit)
        derived['imbalance_x_overlap'] = features['imbalance_ratio'] * features['overlap_pct'] / 100

        # Class separation * F1 range (low separation + high range = benefit)
        derived['separation_x_range'] = features['class_separation'] * features['f1_range']

        # Threshold distance * potential gain (key indicator)
        derived['distance_x_gain'] = features['threshold_distance'] * features['potential_gain']

        # Overlap normalized by samples (high overlap on small data = risky)
        derived['overlap_per_sample'] = features['overlap_pct'] / np.log10(features['n_samples'] + 1)

        # Feature density (samples per feature)
        derived['feature_density'] = np.log10(features['samples_per_feature'] + 1)

        return derived

    def get_feature_names(self) -> list:
        """Return list of all meta-feature names in order."""
        return [
            # Dataset features
            'n_samples', 'n_features', 'n_classes', 'imbalance_ratio',
            'minority_ratio', 'log_n_samples', 'log_n_features',
            'samples_per_feature', 'feature_mean_of_means', 'feature_mean_of_stds',
            # Probability features
            'overlap_pct', 'class_separation', 'prob_mean', 'prob_std',
            'prob_skewness', 'prob_kurtosis', 'prob_p10', 'prob_p25',
            'prob_p50', 'prob_p75', 'prob_p90', 'prob_iqr',
            # Sensitivity features
            'best_threshold', 'best_f1', 'f1_at_05', 'f1_range', 'f1_std',
            'threshold_distance', 'potential_gain', 'f1_drop_at_edges',
            # Derived features
            'imbalance_x_overlap', 'separation_x_range', 'distance_x_gain',
            'overlap_per_sample', 'feature_density',
        ]

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array in consistent order."""
        return np.array([features[name] for name in self.get_feature_names()])
