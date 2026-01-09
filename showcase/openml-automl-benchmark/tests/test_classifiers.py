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

    def test_overlap_detection(self, balanced_data):
        """Test that overlap percentage is computed."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'overlap_pct_')
        assert 0.0 <= clf.overlap_pct_ <= 100.0

    def test_class_separation(self, balanced_data):
        """Test that class separation is computed."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'class_separation_')
        assert 0.0 <= clf.class_separation_ <= 1.0

    def test_diagnostics(self, balanced_data):
        """Test that diagnostics dict is populated."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'diagnostics_')
        assert 'strategy' in clf.diagnostics_
        assert 'overlap_pct' in clf.diagnostics_
        assert 'threshold_range_used' in clf.diagnostics_

    def test_auto_threshold_range(self, balanced_data):
        """Test auto threshold range selection."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
        clf.fit(X_train, y_train)

        assert clf.diagnostics_['strategy'] in ['aggressive', 'normal', 'skip', 'skip_flat', 'skip_low_gain', 'skip_near_default']

    def test_sensitivity_metrics(self, balanced_data):
        """Test that sensitivity metrics are computed."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert 'f1_range' in clf.diagnostics_
        assert 'potential_gain' in clf.diagnostics_
        assert 'threshold_distance_from_05' in clf.diagnostics_

    def test_skip_if_confident(self, balanced_data):
        """Test skip_if_confident parameter."""
        X_train, X_test, y_train, y_test = balanced_data

        # With skip enabled (default)
        clf1 = ThresholdOptimizedClassifier(skip_if_confident=True, random_state=42)
        clf1.fit(X_train, y_train)

        # With skip disabled
        clf2 = ThresholdOptimizedClassifier(skip_if_confident=False, random_state=42)
        clf2.fit(X_train, y_train)

        assert hasattr(clf1, 'optimization_skipped_')
        assert hasattr(clf2, 'optimization_skipped_')
        # When skip is disabled, optimization should never be skipped
        assert clf2.optimization_skipped_ == False

    def test_calibration(self, balanced_data):
        """Test calibration option."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(calibrate=True, random_state=42)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_summary(self, balanced_data):
        """Test summary method."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "ThresholdOptimizedClassifier" in summary
        assert "Overlap zone" in summary
        assert "Strategy" in summary

    def test_optimize_for_f1(self, imbalanced_data):
        """Test optimize_for='f1' (default)."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(optimize_for='f1', random_state=42)
        clf.fit(X_train, y_train)

        assert clf.optimize_for == 'f1'
        assert clf.diagnostics_['optimize_for'] == 'f1'
        assert 'Optimizing for: F1' in clf.summary()

    def test_optimize_for_f2(self, imbalanced_data):
        """Test optimize_for='f2' (emphasizes recall)."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(optimize_for='f2', random_state=42)
        clf.fit(X_train, y_train)

        assert clf.optimize_for == 'f2'
        assert clf.diagnostics_['optimize_for'] == 'f2'
        assert 'Optimizing for: F2' in clf.summary()

    def test_optimize_for_recall(self, imbalanced_data):
        """Test optimize_for='recall'."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(optimize_for='recall', random_state=42)
        clf.fit(X_train, y_train)

        assert clf.optimize_for == 'recall'
        assert clf.diagnostics_['optimize_for'] == 'recall'
        assert 'Optimizing for: RECALL' in clf.summary()

    def test_optimize_for_precision(self, imbalanced_data):
        """Test optimize_for='precision'."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(optimize_for='precision', random_state=42)
        clf.fit(X_train, y_train)

        assert clf.optimize_for == 'precision'
        assert clf.diagnostics_['optimize_for'] == 'precision'
        assert 'Optimizing for: PRECISION' in clf.summary()

    def test_optimize_for_in_get_params(self, balanced_data):
        """Test that optimize_for is included in get_params."""
        clf = ThresholdOptimizedClassifier(optimize_for='f2', random_state=42)
        params = clf.get_params()

        assert 'optimize_for' in params
        assert params['optimize_for'] == 'f2'

    def test_optimize_for_invalid(self, balanced_data):
        """Test that invalid optimize_for raises ValueError."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(optimize_for='invalid_metric', random_state=42)

        with pytest.raises(ValueError, match="Unknown metric"):
            clf.fit(X_train, y_train)

    def test_auto_model_small_dataset(self, balanced_data):
        """Test auto model selection for small datasets."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(base_estimator='auto', random_state=42)
        clf.fit(X_train, y_train)

        # Small dataset (<2000 samples) should use LogisticRegression
        assert 'n_samples' in clf.diagnostics_
        assert clf.diagnostics_['n_samples'] < 2000
        assert clf.diagnostics_['auto_model'] is not None
        assert 'logreg' in clf.diagnostics_['auto_model']

    def test_auto_model_in_summary(self, balanced_data):
        """Test that auto model selection is shown in summary."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(base_estimator='auto', random_state=42)
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "Base model:" in summary

    def test_auto_model_large_dataset(self):
        """Test auto model selection for larger datasets."""
        # Create a larger dataset (>2000 samples)
        X, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ThresholdOptimizedClassifier(base_estimator='auto', random_state=42)
        clf.fit(X_train, y_train)

        # Dataset has 2400 samples, should use XGBoost/LightGBM if available
        assert clf.diagnostics_['auto_model'] is not None
        # Either xgboost, lightgbm or logreg fallback
        auto_model = clf.diagnostics_['auto_model']
        assert 'xgboost' in auto_model or 'lightgbm' in auto_model or 'logreg' in auto_model

    def test_multiclass_support(self):
        """Test that multiclass data is handled gracefully."""
        # Create multiclass dataset
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf = ThresholdOptimizedClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Check that warning was issued
            assert len(w) == 1
            assert "Multiclass detected" in str(w[0].message)

        # Check predictions work
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1, 2})

        # Check diagnostics
        assert clf.diagnostics_['strategy'] == 'multiclass'
        assert clf.diagnostics_['n_classes'] == 3
        assert clf.optimal_threshold_ is None

    def test_multiclass_summary(self):
        """Test that multiclass summary is correct."""
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=3,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = ThresholdOptimizedClassifier(random_state=42)
            clf.fit(X, y)

        summary = clf.summary()
        assert "Multiclass (4 classes)" in summary
        assert "argmax prediction" in summary

    def test_multiclass_proba(self):
        """Test that multiclass probabilities are correct shape."""
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=3,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = ThresholdOptimizedClassifier(random_state=42)
            clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_tune_base_model(self, imbalanced_data):
        """Test hyperparameter tuning for base model."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(tune_base_model=True, random_state=42)
        clf.fit(X_train, y_train)

        # Check tuning info is stored in diagnostics
        assert 'tuning' in clf.diagnostics_
        tuning = clf.diagnostics_['tuning']
        assert tuning is not None
        assert tuning['status'] == 'completed'
        assert 'best_params' in tuning
        assert 'best_score' in tuning
        assert 'model_type' in tuning
        assert tuning['model_type'] == 'LogisticRegression'

        # Model should still work for predictions
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_tune_base_model_in_summary(self, imbalanced_data):
        """Test that tuning info is shown in summary."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(tune_base_model=True, random_state=42)
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "Hyperparameter Tuning:" in summary
        assert "Best params:" in summary
        assert "CV score:" in summary

    def test_tune_base_model_disabled(self, imbalanced_data):
        """Test that tuning is disabled by default."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Tuning should be None when disabled
        assert clf.diagnostics_['tuning'] is None

    def test_tune_base_model_in_get_params(self, balanced_data):
        """Test that tune_base_model is included in get_params."""
        clf = ThresholdOptimizedClassifier(tune_base_model=True, random_state=42)
        params = clf.get_params()

        assert 'tune_base_model' in params
        assert params['tune_base_model'] is True

    def test_tune_base_model_with_auto(self):
        """Test tuning with auto model selection."""
        # Small dataset so it uses LogisticRegression
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ThresholdOptimizedClassifier(
            base_estimator='auto',
            tune_base_model=True,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Check that tuning worked with auto model
        assert clf.diagnostics_['tuning']['status'] == 'completed'
        assert clf.diagnostics_['auto_model'] is not None

    def test_cost_matrix_basic(self, imbalanced_data):
        """Test cost-sensitive optimization."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            cost_matrix={'fp': 1, 'fn': 10},  # FN costs 10x more
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Check that cost_matrix is stored in diagnostics
        assert clf.diagnostics_['cost_matrix'] == {'fp': 1, 'fn': 10}
        assert clf.diagnostics_['optimize_for'] == 'cost'

        # Model should still work for predictions
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_cost_matrix_threshold_shift(self, imbalanced_data):
        """Test that cost matrix affects threshold."""
        X_train, X_test, y_train, y_test = imbalanced_data

        # High FN cost should lower threshold (catch more positives)
        clf_high_fn = ThresholdOptimizedClassifier(
            cost_matrix={'fp': 1, 'fn': 100},
            skip_if_confident=False,
            threshold_range=(0.1, 0.9),
            random_state=42
        )
        clf_high_fn.fit(X_train, y_train)

        # High FP cost should raise threshold (be more selective)
        clf_high_fp = ThresholdOptimizedClassifier(
            cost_matrix={'fp': 100, 'fn': 1},
            skip_if_confident=False,
            threshold_range=(0.1, 0.9),
            random_state=42
        )
        clf_high_fp.fit(X_train, y_train)

        # High FN cost should result in lower threshold
        assert clf_high_fn.optimal_threshold_ < clf_high_fp.optimal_threshold_

    def test_cost_matrix_in_summary(self, imbalanced_data):
        """Test that cost_matrix is shown in summary."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            cost_matrix={'fp': 1, 'fn': 5},
            random_state=42
        )
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "COST" in summary
        assert "fp=1" in summary
        assert "fn=5" in summary
        assert "CV total cost" in summary

    def test_cost_matrix_in_get_params(self, balanced_data):
        """Test that cost_matrix is included in get_params."""
        clf = ThresholdOptimizedClassifier(
            cost_matrix={'fp': 2, 'fn': 3},
            random_state=42
        )
        params = clf.get_params()

        assert 'cost_matrix' in params
        assert params['cost_matrix'] == {'fp': 2, 'fn': 3}

    def test_calibration_isotonic(self, imbalanced_data):
        """Test isotonic calibration."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            calibrate='isotonic',
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Should work and produce predictions
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

        # Probabilities should be in [0, 1]
        proba = clf.predict_proba(X_test)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_calibration_sigmoid(self, imbalanced_data):
        """Test sigmoid (Platt) calibration."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            calibrate='sigmoid',
            random_state=42
        )
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_calibration_platt_alias(self, imbalanced_data):
        """Test that 'platt' is an alias for 'sigmoid'."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            calibrate='platt',
            random_state=42
        )
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_calibration_true_defaults_to_isotonic(self, imbalanced_data):
        """Test that calibrate=True defaults to isotonic."""
        X_train, X_test, y_train, y_test = imbalanced_data

        clf_true = ThresholdOptimizedClassifier(calibrate=True, random_state=42)
        clf_isotonic = ThresholdOptimizedClassifier(calibrate='isotonic', random_state=42)

        clf_true.fit(X_train, y_train)
        clf_isotonic.fit(X_train, y_train)

        # Both should produce predictions
        pred_true = clf_true.predict(X_test)
        pred_isotonic = clf_isotonic.predict(X_test)
        assert len(pred_true) == len(pred_isotonic)

    def test_calibration_invalid(self, balanced_data):
        """Test that invalid calibration method raises ValueError."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(
            calibrate='invalid_method',
            random_state=42
        )

        with pytest.raises(ValueError, match="Unknown calibration method"):
            clf.fit(X_train, y_train)

    def test_ensemble_thresholds_basic(self, imbalanced_data):
        """Test ensemble threshold optimization."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            ensemble_thresholds=5,
            skip_if_confident=False,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Check that ensemble was created
        assert clf.threshold_ensemble_ is not None
        assert len(clf.threshold_ensemble_) == 5
        assert clf.diagnostics_['threshold_ensemble'] is not None

        # Should still produce predictions
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_ensemble_thresholds_in_summary(self, imbalanced_data):
        """Test that ensemble thresholds are shown in summary."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(
            ensemble_thresholds=3,
            skip_if_confident=False,
            random_state=42
        )
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert "Ensemble Thresholds:" in summary
        assert "Count: 3" in summary
        assert "Mean:" in summary
        assert "Std:" in summary

    def test_ensemble_thresholds_none(self, imbalanced_data):
        """Test that ensemble is None by default."""
        X_train, X_test, y_train, y_test = imbalanced_data
        clf = ThresholdOptimizedClassifier(random_state=42)
        clf.fit(X_train, y_train)

        assert clf.threshold_ensemble_ is None

    def test_ensemble_thresholds_in_get_params(self, balanced_data):
        """Test that ensemble_thresholds is included in get_params."""
        clf = ThresholdOptimizedClassifier(
            ensemble_thresholds=5,
            random_state=42
        )
        params = clf.get_params()

        assert 'ensemble_thresholds' in params
        assert params['ensemble_thresholds'] == 5

    def test_ensemble_thresholds_skipped_when_confident(self, balanced_data):
        """Test that ensemble is skipped when optimization is skipped."""
        X_train, X_test, y_train, y_test = balanced_data
        clf = ThresholdOptimizedClassifier(
            ensemble_thresholds=5,
            skip_if_confident=True,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # If optimization was skipped, ensemble should be None
        if clf.optimization_skipped_:
            assert clf.threshold_ensemble_ is None


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
