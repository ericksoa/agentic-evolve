"""
Gen 8b: 3-model ensemble with threshold 0.34
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=4):
        self.n_bins = n_bins
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        for i in range(X.shape[1]):
            col = X[:, i]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) > self.n_bins:
                try:
                    self.bin_edges_[i] = np.percentile(col_valid, np.linspace(0, 100, self.n_bins + 1))
                except:
                    pass
        return self

    def transform(self, X):
        features = [X]
        if X.shape[1] >= 8:
            glucose, bmi, age = X[:, 1], X[:, 5], X[:, 7]
            features.extend([
                (glucose < 140).astype(float).reshape(-1, 1),
                ((glucose >= 140) & (glucose < 200)).astype(float).reshape(-1, 1),
                (glucose >= 200).astype(float).reshape(-1, 1),
                ((bmi >= 18.5) & (bmi < 25)).astype(float).reshape(-1, 1),
                ((bmi >= 25) & (bmi < 30)).astype(float).reshape(-1, 1),
                (bmi >= 30).astype(float).reshape(-1, 1),
                (age < 30).astype(float).reshape(-1, 1),
                ((age >= 30) & (age < 50)).astype(float).reshape(-1, 1),
                (age >= 50).astype(float).reshape(-1, 1),
            ])
        for i, edges in self.bin_edges_.items():
            if i < X.shape[1]:
                features.append(np.digitize(X[:, i], edges[1:-1]).reshape(-1, 1))
        return np.nan_to_num(np.hstack(features), nan=0, posinf=0, neginf=0)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.34):
        self.threshold = threshold
        self.models_, self.fes_, self.scalers_, self.rfes_ = [], [], [], []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_, self.fes_, self.scalers_, self.rfes_ = [], [], [], []

        for n_feat in [None, 8, 10]:
            fe = DiabetesFeatureEngineer(n_bins=4)
            X_fe = fe.fit_transform(X)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_fe)
            rfe = None
            if n_feat:
                rfe = RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=n_feat)
                X_s = rfe.fit_transform(X_s, y)
            model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
            model.fit(X_s, y)
            self.fes_.append(fe)
            self.scalers_.append(scaler)
            self.rfes_.append(rfe)
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        probas = []
        for i in range(len(self.models_)):
            X_s = self.scalers_[i].transform(self.fes_[i].transform(X))
            if self.rfes_[i]:
                X_s = self.rfes_[i].transform(X_s)
            probas.append(self.models_[i].predict_proba(X_s))
        return np.mean(probas, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', EnsembleClassifier(threshold=0.34))
    ])
