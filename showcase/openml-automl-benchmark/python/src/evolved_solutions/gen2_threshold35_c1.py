"""
Gen 2d: Threshold 0.35 + C=1.0.

Testing different regularization with best threshold.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_domain=True, add_bins=True, n_bins=4):
        self.add_domain = add_domain
        self.add_bins = add_bins
        self.n_bins = n_bins
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        if self.add_bins:
            for i in range(X.shape[1]):
                col = X[:, i]
                col_valid = col[~np.isnan(col)]
                if len(col_valid) > self.n_bins:
                    try:
                        percentiles = np.linspace(0, 100, self.n_bins + 1)
                        self.bin_edges_[i] = np.percentile(col_valid, percentiles)
                    except:
                        pass
        return self

    def transform(self, X):
        features = [X]

        if self.add_domain:
            if X.shape[1] >= 8:
                glucose = X[:, 1]
                bmi = X[:, 5]
                age = X[:, 7]

                glucose_normal = (glucose < 140).astype(float).reshape(-1, 1)
                glucose_prediabetic = ((glucose >= 140) & (glucose < 200)).astype(float).reshape(-1, 1)
                glucose_diabetic = (glucose >= 200).astype(float).reshape(-1, 1)

                bmi_underweight = (bmi < 18.5).astype(float).reshape(-1, 1)
                bmi_normal = ((bmi >= 18.5) & (bmi < 25)).astype(float).reshape(-1, 1)
                bmi_overweight = ((bmi >= 25) & (bmi < 30)).astype(float).reshape(-1, 1)
                bmi_obese = (bmi >= 30).astype(float).reshape(-1, 1)

                age_young = (age < 30).astype(float).reshape(-1, 1)
                age_middle = ((age >= 30) & (age < 50)).astype(float).reshape(-1, 1)
                age_senior = (age >= 50).astype(float).reshape(-1, 1)

                features.extend([
                    glucose_normal, glucose_prediabetic, glucose_diabetic,
                    bmi_underweight, bmi_normal, bmi_overweight, bmi_obese,
                    age_young, age_middle, age_senior
                ])

        if self.add_bins:
            bin_features = []
            for i, edges in self.bin_edges_.items():
                if i < X.shape[1]:
                    col = X[:, i]
                    binned = np.digitize(col, edges[1:-1]).reshape(-1, 1)
                    bin_features.append(binned)

            if bin_features:
                features.extend(bin_features)

        result = np.hstack(features)
        result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)

        return result


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.base_estimator = LogisticRegression(
            C=1.0,  # Less regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)


def create_pipeline():
    """Threshold 0.35 + C=1.0."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_engineer', DiabetesFeatureEngineer(add_domain=True, add_bins=True, n_bins=4)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(
            estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            n_features_to_select=10,
            step=1
        )),
        ('classifier', ThresholdClassifier(threshold=0.35))
    ])
