"""
Gen 6a: Medical Domain + RFE8 + Threshold 0.35

Combining:
- Medical domain feature engineering
- RFE to 8 features (our winning feature count)
- Threshold 0.35 (our winning threshold)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class MedicalDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Focused medical domain features - only the most important ones."""

    def __init__(self):
        self.glucose_mean_ = None
        self.insulin_mean_ = None

    def fit(self, X, y=None):
        self.glucose_mean_ = np.nanmean(X[:, 1])
        self.insulin_mean_ = np.nanmean(X[:, 4])
        return self

    def transform(self, X):
        preg, glucose, bp, skin, insulin, bmi, pedigree, age = [X[:, i] for i in range(8)]
        features = [X]

        # Key clinical glucose thresholds
        features.extend([
            ((glucose >= 140) & (glucose < 200)).astype(float).reshape(-1, 1),  # Prediabetic
            (glucose >= 200).astype(float).reshape(-1, 1),  # Diabetic
        ])

        # Key BMI thresholds
        features.extend([
            (bmi >= 30).astype(float).reshape(-1, 1),  # Obese
            (bmi >= 35).astype(float).reshape(-1, 1),  # Severely obese
        ])

        # Key age thresholds
        features.append((age >= 45).astype(float).reshape(-1, 1))  # ADA screening

        # HOMA-IR proxy (insulin resistance)
        insulin_safe = np.where(insulin > 0, insulin, self.insulin_mean_)
        glucose_safe = np.where(glucose > 0, glucose, self.glucose_mean_)
        homa_proxy = np.clip((glucose_safe * insulin_safe) / 405.0, 0, 20)
        features.append(homa_proxy.reshape(-1, 1))

        # Key interactions
        features.extend([
            ((bmi >= 30) & (glucose >= 140)).astype(float).reshape(-1, 1),  # Metabolic double risk
            ((pedigree > 0.5) & (glucose >= 120)).astype(float).reshape(-1, 1),  # Genetic + glucose
        ])

        # Simplified FINDRISC
        findrisc = (
            (age >= 45).astype(float) * 2 +
            (bmi >= 30).astype(float) * 3 +
            (glucose >= 140).astype(float) * 5 +
            (pedigree > 0.5).astype(float) * 3
        )
        features.append(findrisc.reshape(-1, 1))

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
        ('medical_features', MedicalDomainFeatureEngineer()),
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            n_features_to_select=8,
            step=1
        )),
        ('classifier', ThresholdClassifier(threshold=0.35))
    ])
