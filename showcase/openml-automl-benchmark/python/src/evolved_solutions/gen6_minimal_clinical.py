"""
Gen 6b: Minimal Clinical Features

Only the 3-4 most predictive clinical indicators based on medical knowledge:
1. Glucose status (the most important predictor)
2. BMI/obesity status
3. Age risk
4. Family history (pedigree)

Hypothesis: Less is more on small data.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class MinimalClinicalFeatures(BaseEstimator, TransformerMixin):
    """Only the clinically most important features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        glucose = X[:, 1]
        bmi = X[:, 5]
        pedigree = X[:, 6]
        age = X[:, 7]

        features = [X]  # Original 8 features

        # Only 4 derived features - the essentials
        features.extend([
            # Glucose risk level (most predictive)
            (glucose >= 140).astype(float).reshape(-1, 1),

            # Obesity (second most predictive)
            (bmi >= 30).astype(float).reshape(-1, 1),

            # Age risk (third)
            (age >= 45).astype(float).reshape(-1, 1),

            # Family history (fourth)
            (pedigree > 0.5).astype(float).reshape(-1, 1),
        ])

        result = np.hstack(features)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.base_estimator = LogisticRegression(
            C=0.5, class_weight='balanced', max_iter=1000, random_state=42
        )
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
        ('clinical_features', MinimalClinicalFeatures()),
        ('scaler', StandardScaler()),
        ('classifier', ThresholdClassifier(threshold=0.35))
    ])
