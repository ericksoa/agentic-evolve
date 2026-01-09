"""
Gen 5b: Medical Domain Features + XGBoost

XGBoost can capture non-linear interactions that LogReg misses.
Combined with domain-expert features.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from xgboost import XGBClassifier


class MedicalDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Medical domain feature engineering for diabetes prediction."""

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

        # Clinical glucose thresholds
        features.extend([
            (glucose < 140).astype(float).reshape(-1, 1),
            ((glucose >= 140) & (glucose < 200)).astype(float).reshape(-1, 1),
            (glucose >= 200).astype(float).reshape(-1, 1),
        ])

        # BMI categories
        features.extend([
            ((bmi >= 25) & (bmi < 30)).astype(float).reshape(-1, 1),
            (bmi >= 30).astype(float).reshape(-1, 1),
            (bmi >= 35).astype(float).reshape(-1, 1),
        ])

        # Age risk stratification
        features.extend([
            (age >= 45).astype(float).reshape(-1, 1),
            (age >= 55).astype(float).reshape(-1, 1),
        ])

        # Insulin resistance markers
        insulin_safe = np.where(insulin > 0, insulin, self.insulin_mean_)
        glucose_safe = np.where(glucose > 0, glucose, self.glucose_mean_)
        homa_proxy = np.clip((glucose_safe * insulin_safe) / 405.0, 0, 20)
        features.append(homa_proxy.reshape(-1, 1))

        # Pregnancy risk
        features.extend([
            (preg >= 4).astype(float).reshape(-1, 1),
            ((preg > 0) & (glucose >= 140)).astype(float).reshape(-1, 1),
        ])

        # Critical interactions
        features.extend([
            ((bmi >= 30) & (glucose >= 140)).astype(float).reshape(-1, 1),
            ((pedigree > 0.5) & (glucose >= 120)).astype(float).reshape(-1, 1),
            ((age >= 40) & (bmi >= 28) & (glucose >= 120)).astype(float).reshape(-1, 1),
        ])

        # FINDRISC proxy
        findrisc = (
            (age >= 45).astype(float) * 2 + (bmi >= 25).astype(float) * 1 +
            (bmi >= 30).astype(float) * 2 + (glucose >= 140).astype(float) * 5 +
            (pedigree > 0.5).astype(float) * 3
        )
        features.append(findrisc.reshape(-1, 1))

        result = np.hstack(features)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)


class ThresholdXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost with optimized threshold."""

    def __init__(self, threshold=0.40):
        self.threshold = threshold
        self.base_estimator = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.8,  # Approx class ratio
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
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
        ('classifier', ThresholdXGBClassifier(threshold=0.40))
    ])
