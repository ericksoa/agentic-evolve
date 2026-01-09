"""
Tests for adaptive_ensemble classifiers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from adaptive_ensemble import (
    ThresholdOptimizedClassifier,
    AdaptiveEnsembleClassifier,
    DatasetAnalyzer,
)


@pytest.fixture
def balanced_data():
    """Generate balanced binary classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


class TestThresholdOptimizedClassifier:
    """Tests for ThresholdOptimizedClassifier."""

    def test_fit_predict(self, balanced_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, balanced_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_optimal_threshold_learned(self, imbalanced_data):
        """Test that optimal threshold is learned."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'optimal_threshold_')
        assert 0.0 <= clf.optimal_threshold_ <= 1.0

    def test_imbalance_ratio_computed(self, imbalanced_data):
        """Test that imbalance ratio is computed."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'imbalance_ratio_')
        assert clf.imbalance_ratio_ > 1.0

    def test_sklearn_compatible(self, balanced_data):
        """Test sklearn compatibility (get_params, set_params)."""
        clf = ThresholdOptimizedClassifier(cv=5, random_state=42)

        params = clf.get_params()
        assert params['cv'] == 5
        assert params['random_state'] == 42

        clf.set_params(cv=3)
        assert clf.cv == 3


class TestAdaptiveEnsembleClassifier:
    """Tests for AdaptiveEnsembleClassifier."""

    def test_fit_predict(self, balanced_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, balanced_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_created(self, balanced_data):
        """Test that ensemble models are created."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'models_')
        assert len(clf.models_) > 0

    def test_summary(self, balanced_data):
        """Test summary method."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "AdaptiveEnsembleClassifier" in summary
        assert "threshold" in summary.lower()


class TestDatasetAnalyzer:
    """Tests for DatasetAnalyzer."""

    def test_analyze_balanced(self, balanced_data):
        """Test analysis of balanced data."""
        X_train, X_test, y_train, y_test = balanced_data
        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(X_train, y_train)

        assert profile.n_samples == len(y_train)
        assert profile.n_features == X_train.shape[1]
        assert profile.imbalance_ratio >= 1.0

    def test_analyze_imbalanced(self, imbalanced_data):
        """Test analysis of imbalanced data."""
        X_train, X_test, y_train, y_test = imbalanced_data
        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(X_train, y_train)

        assert profile.is_imbalanced
        assert profile.recommended_threshold < 0.5

    def test_summary(self, balanced_data):
        """Test summary method."""
        X_train, X_test, y_train, y_test = balanced_data
        analyzer = DatasetAnalyzer()
        analyzer.analyze(X_train, y_train)

        summary = analyzer.summary()
        assert "Dataset Profile" in summary
        assert "Recommended Strategy" in summary
