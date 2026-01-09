"""
Unit tests for the meta-learning module.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification


class TestMetaFeatureExtractor:
    """Tests for MetaFeatureExtractor class."""

    def test_extract_basic(self):
        """Test basic feature extraction."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)

        # Check all expected features are present
        expected_keys = extractor.get_feature_names()
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_extract_dataset_features(self):
        """Test dataset-level features."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)

        assert features['n_samples'] == 500
        assert features['n_features'] == 10
        assert features['n_classes'] == 2
        assert features['imbalance_ratio'] >= 1.0

    def test_extract_imbalanced(self):
        """Test extraction on imbalanced data."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            weights=[0.8, 0.2],
            random_state=42
        )
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)

        # Imbalance ratio should reflect the class distribution
        assert features['imbalance_ratio'] > 2.0
        assert features['minority_ratio'] < 0.3

    def test_extract_sensitivity_features(self):
        """Test threshold sensitivity features."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)

        # Sensitivity features should be valid
        assert 0 <= features['best_threshold'] <= 1
        assert 0 <= features['best_f1'] <= 1
        assert 0 <= features['f1_at_05'] <= 1
        assert features['f1_range'] >= 0
        assert features['threshold_distance'] >= 0

    def test_to_array(self):
        """Test conversion to numpy array."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(n_samples=300, n_features=5, random_state=42)
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)
        arr = extractor.to_array(features)

        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(extractor.get_feature_names())
        assert not np.any(np.isnan(arr))

    def test_extract_small_dataset(self):
        """Test extraction on small dataset."""
        from adaptive_ensemble.meta_learning import MetaFeatureExtractor

        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        extractor = MetaFeatureExtractor(cv=2)  # Fewer folds for small data
        features = extractor.extract(X, y)

        assert features['n_samples'] == 50
        assert features['n_features'] == 5


class TestMetaLearningDetector:
    """Tests for MetaLearningDetector class."""

    def test_fit_and_predict(self):
        """Test training and prediction."""
        from adaptive_ensemble.meta_learning import MetaLearningDetector, MetaFeatureExtractor

        # Create synthetic training data
        extractor = MetaFeatureExtractor()
        np.random.seed(42)

        X_meta = []
        y_meta = []
        for i in range(20):
            weight = np.random.uniform(0.4, 0.6)
            X, y = make_classification(
                n_samples=np.random.randint(200, 1000),
                n_features=np.random.randint(5, 20),
                weights=[weight, 1 - weight],
                random_state=42 + i,
            )
            features = extractor.extract(X, y)
            X_meta.append(extractor.to_array(features))
            y_meta.append(np.random.randint(0, 2))

        training_data = {
            'X': np.array(X_meta),
            'y': np.array(y_meta),
            'feature_names': extractor.get_feature_names(),
        }

        # Train detector
        detector = MetaLearningDetector()
        detector.fit(training_data, verbose=False)

        # Test prediction
        X_test, y_test = make_classification(n_samples=500, n_features=10, random_state=99)
        features = extractor.extract(X_test, y_test)
        prob = detector.predict_proba(features)

        assert 0 <= prob <= 1

    def test_should_optimize(self):
        """Test should_optimize method."""
        from adaptive_ensemble.meta_learning import MetaLearningDetector, MetaFeatureExtractor

        # Create and train detector with synthetic data
        extractor = MetaFeatureExtractor()
        np.random.seed(42)

        X_meta = []
        y_meta = []
        for i in range(20):
            X, y = make_classification(
                n_samples=np.random.randint(200, 1000),
                n_features=np.random.randint(5, 20),
                random_state=42 + i,
            )
            features = extractor.extract(X, y)
            X_meta.append(extractor.to_array(features))
            y_meta.append(1 if i % 2 == 0 else 0)

        training_data = {
            'X': np.array(X_meta),
            'y': np.array(y_meta),
            'feature_names': extractor.get_feature_names(),
        }

        detector = MetaLearningDetector(threshold=0.5)
        detector.fit(training_data, verbose=False)

        # Test
        X_test, y_test = make_classification(n_samples=500, n_features=10, random_state=99)
        features = extractor.extract(X_test, y_test)
        result = detector.should_optimize(features)

        assert isinstance(result, (bool, np.bool_))

    def test_predict_with_confidence(self):
        """Test predict_with_confidence method."""
        from adaptive_ensemble.meta_learning import MetaLearningDetector, MetaFeatureExtractor

        # Create and train detector
        extractor = MetaFeatureExtractor()
        np.random.seed(42)

        X_meta = []
        y_meta = []
        for i in range(20):
            X, y = make_classification(
                n_samples=np.random.randint(200, 1000),
                n_features=np.random.randint(5, 20),
                random_state=42 + i,
            )
            features = extractor.extract(X, y)
            X_meta.append(extractor.to_array(features))
            y_meta.append(1 if i % 2 == 0 else 0)

        training_data = {
            'X': np.array(X_meta),
            'y': np.array(y_meta),
            'feature_names': extractor.get_feature_names(),
        }

        detector = MetaLearningDetector()
        detector.fit(training_data, verbose=False)

        # Test
        X_test, y_test = make_classification(n_samples=500, n_features=10, random_state=99)
        features = extractor.extract(X_test, y_test)
        result = detector.predict_with_confidence(features)

        assert 'probability' in result
        assert 'recommendation' in result
        assert 'confidence' in result
        assert result['recommendation'] in ['optimize', 'skip']
        assert 0 <= result['confidence'] <= 1

    def test_save_and_load(self, tmp_path):
        """Test saving and loading detector."""
        from adaptive_ensemble.meta_learning import MetaLearningDetector, MetaFeatureExtractor

        # Create and train detector
        extractor = MetaFeatureExtractor()
        np.random.seed(42)

        X_meta = []
        y_meta = []
        for i in range(15):
            X, y = make_classification(
                n_samples=np.random.randint(200, 500),
                n_features=np.random.randint(5, 15),
                random_state=42 + i,
            )
            features = extractor.extract(X, y)
            X_meta.append(extractor.to_array(features))
            y_meta.append(1 if i % 2 == 0 else 0)

        training_data = {
            'X': np.array(X_meta),
            'y': np.array(y_meta),
            'feature_names': extractor.get_feature_names(),
        }

        detector = MetaLearningDetector()
        detector.fit(training_data, verbose=False)

        # Save
        save_path = tmp_path / "detector.pkl"
        detector.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded_detector = MetaLearningDetector()
        loaded_detector.load(str(save_path))

        # Compare predictions
        X_test, y_test = make_classification(n_samples=300, n_features=10, random_state=99)
        features = extractor.extract(X_test, y_test)

        original_prob = detector.predict_proba(features)
        loaded_prob = loaded_detector.predict_proba(features)

        assert np.isclose(original_prob, loaded_prob)

    def test_fallback_detector(self):
        """Test fallback detector when no pretrained model exists."""
        from adaptive_ensemble.meta_learning import MetaLearningDetector, MetaFeatureExtractor

        detector = MetaLearningDetector.create_fallback()
        extractor = MetaFeatureExtractor()

        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        features = extractor.extract(X, y)

        prob = detector._fallback_predict(features)
        assert 0 <= prob <= 1


class TestThresholdClassifierWithMetaDetector:
    """Tests for ThresholdOptimizedClassifier with use_meta_detector=True."""

    def test_meta_detector_parameter(self):
        """Test that meta_detector parameter is accepted."""
        from adaptive_ensemble import ThresholdOptimizedClassifier

        clf = ThresholdOptimizedClassifier(use_meta_detector=True)
        assert clf.use_meta_detector == True
        assert clf.meta_detector_threshold == 0.5

    def test_meta_detector_in_get_params(self):
        """Test that meta_detector appears in get_params."""
        from adaptive_ensemble import ThresholdOptimizedClassifier

        clf = ThresholdOptimizedClassifier(
            use_meta_detector=True,
            meta_detector_threshold=0.6,
        )
        params = clf.get_params()

        assert 'use_meta_detector' in params
        assert 'meta_detector_threshold' in params
        assert params['use_meta_detector'] == True
        assert params['meta_detector_threshold'] == 0.6

    def test_fit_with_meta_detector_fallback(self):
        """Test fitting with meta_detector (should use fallback since no pretrained model)."""
        from adaptive_ensemble import ThresholdOptimizedClassifier

        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.7, 0.3],
            random_state=42,
        )

        clf = ThresholdOptimizedClassifier(use_meta_detector=True)
        clf.fit(X, y)

        # Should complete without error
        assert hasattr(clf, 'optimal_threshold_')
        assert hasattr(clf, 'diagnostics_')

        # Make predictions
        predictions = clf.predict(X)
        assert len(predictions) == len(X)


class TestTrainingUtilities:
    """Tests for training utilities."""

    def test_evaluate_threshold_benefit(self):
        """Test threshold benefit evaluation."""
        from adaptive_ensemble.meta_learning.training import evaluate_threshold_benefit

        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.7, 0.3],
            random_state=42,
        )

        result = evaluate_threshold_benefit(X, y, cv=3)

        assert 'f1_default' in result
        assert 'f1_optimized' in result
        assert 'optimal_threshold' in result
        assert 'gain' in result
        assert 'will_help' in result
        assert isinstance(result['will_help'], (bool, np.bool_))

    def test_collect_training_data(self):
        """Test collecting training data from multiple datasets."""
        from adaptive_ensemble.meta_learning.training import collect_training_data

        # Create test datasets
        datasets = []
        for i in range(3):
            X, y = make_classification(
                n_samples=200 + i * 100,
                n_features=5 + i * 2,
                random_state=42 + i,
            )
            datasets.append((X, y, f"test_dataset_{i}"))

        training_data = collect_training_data(datasets, cv=3, verbose=False)

        assert 'X' in training_data
        assert 'y' in training_data
        assert 'feature_names' in training_data
        assert 'dataset_names' in training_data
        assert len(training_data['X']) == 3
        assert len(training_data['y']) == 3


class TestShouldOptimizeThreshold:
    """Tests for the convenience function."""

    def test_should_optimize_threshold(self):
        """Test the convenience function."""
        from adaptive_ensemble.meta_learning.detector import should_optimize_threshold

        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.7, 0.3],
            random_state=42,
        )

        result = should_optimize_threshold(X, y)

        assert 'should_optimize' in result
        assert 'probability' in result
        assert 'confidence' in result
        assert 'meta_features' in result
        assert isinstance(result['should_optimize'], bool)
