"""
Gen 3b: Threshold 0.37 - fine-tuned.
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

        if self.add_domain and X.shape[1] >= 8:
            glucose, bmi, age = X[:, 1], X[:, 5], X[:, 7]

            features.extend([
                (glucose < 140).astype(float).reshape(-1, 1),
                ((glucose >= 140) & (glucose < 200)).astype(float).reshape(-1, 1),
                (glucose >= 200).astype(float).reshape(-1, 1),
                (bmi < 18.5).astype(float).reshape(-1, 1),
                ((bmi >= 18.5) & (bmi < 25)).astype(float).reshape(-1, 1),
                ((bmi >= 25) & (bmi < 30)).astype(float).reshape(-1, 1),
                (bmi >= 30).astype(float).reshape(-1, 1),
                (age < 30).astype(float).reshape(-1, 1),
                ((age >= 30) & (age < 50)).astype(float).reshape(-1, 1),
                (age >= 50).astype(float).reshape(-1, 1)
            ])

        if self.add_bins:
            for i, edges in self.bin_edges_.items():
                if i < X.shape[1]:
                    features.append(np.digitize(X[:, i], edges[1:-1]).reshape(-1, 1))

        result = np.hstack(features)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.37):
        self.threshold = threshold
        self.base_estimator = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_engineer', DiabetesFeatureEngineer(add_domain=True, add_bins=True, n_bins=4)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=10, step=1)),
        ('classifier', ThresholdClassifier(threshold=0.37))
    ])
