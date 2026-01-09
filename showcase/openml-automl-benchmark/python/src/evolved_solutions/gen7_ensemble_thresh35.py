"""
Gen 7b: Ensemble with threshold 0.35

Same as Gen7a but with threshold 0.35 (our original winning threshold).
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
                    percentiles = np.linspace(0, 100, self.n_bins + 1)
                    self.bin_edges_[i] = np.percentile(col_valid, percentiles)
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

        result = np.hstack(features)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.models_ = []
        self.feature_engineers_ = []
        self.scalers_ = []
        self.rfes_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Model 1: Full features
        fe1 = DiabetesFeatureEngineer(n_bins=4)
        X1 = fe1.fit_transform(X)
        scaler1 = StandardScaler()
        X1_scaled = scaler1.fit_transform(X1)
        model1 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model1.fit(X1_scaled, y)
        self.feature_engineers_.append(fe1)
        self.scalers_.append(scaler1)
        self.rfes_.append(None)
        self.models_.append(model1)

        # Model 2: RFE 8
        fe2 = DiabetesFeatureEngineer(n_bins=4)
        X2 = fe2.fit_transform(X)
        scaler2 = StandardScaler()
        X2_scaled = scaler2.fit_transform(X2)
        rfe2 = RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=8)
        X2_rfe = rfe2.fit_transform(X2_scaled, y)
        model2 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model2.fit(X2_rfe, y)
        self.feature_engineers_.append(fe2)
        self.scalers_.append(scaler2)
        self.rfes_.append(rfe2)
        self.models_.append(model2)

        # Model 3: RFE 10
        fe3 = DiabetesFeatureEngineer(n_bins=4)
        X3 = fe3.fit_transform(X)
        scaler3 = StandardScaler()
        X3_scaled = scaler3.fit_transform(X3)
        rfe3 = RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=10)
        X3_rfe = rfe3.fit_transform(X3_scaled, y)
        model3 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model3.fit(X3_rfe, y)
        self.feature_engineers_.append(fe3)
        self.scalers_.append(scaler3)
        self.rfes_.append(rfe3)
        self.models_.append(model3)

        return self

    def _transform_for_model(self, X, idx):
        X_fe = self.feature_engineers_[idx].transform(X)
        X_scaled = self.scalers_[idx].transform(X_fe)
        if self.rfes_[idx] is not None:
            X_scaled = self.rfes_[idx].transform(X_scaled)
        return X_scaled

    def predict_proba(self, X):
        probas = [self.models_[i].predict_proba(self._transform_for_model(X, i)) for i in range(len(self.models_))]
        return np.mean(probas, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', EnsembleClassifier(threshold=0.35))
    ])
