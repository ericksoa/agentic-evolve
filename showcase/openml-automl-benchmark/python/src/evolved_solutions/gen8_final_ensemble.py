"""
Gen 8: Final Optimized Ensemble

Building on Gen7b with:
1. 5 diverse models instead of 3
2. Include RFE variants from 6 to 12 features
3. Threshold 0.35 (proven winner)
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


class FinalEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    5-model ensemble for maximum diversity:
    1. Full features, C=0.5
    2. RFE 6 features
    3. RFE 8 features
    4. RFE 10 features
    5. Full features, C=1.0
    """

    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.models_ = []
        self.feature_engineers_ = []
        self.scalers_ = []
        self.rfes_ = []
        self.classes_ = None

    def _add_model(self, X, y, n_features=None, C=0.5):
        fe = DiabetesFeatureEngineer(n_bins=4)
        X_fe = fe.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fe)

        rfe = None
        if n_features is not None:
            rfe = RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=n_features)
            X_scaled = rfe.fit_transform(X_scaled, y)

        model = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        self.feature_engineers_.append(fe)
        self.scalers_.append(scaler)
        self.rfes_.append(rfe)
        self.models_.append(model)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        self.feature_engineers_ = []
        self.scalers_ = []
        self.rfes_ = []

        # 5 diverse models
        self._add_model(X, y, n_features=None, C=0.5)   # Full features
        self._add_model(X, y, n_features=6, C=0.5)      # RFE 6
        self._add_model(X, y, n_features=8, C=0.5)      # RFE 8 (previous winner)
        self._add_model(X, y, n_features=10, C=0.5)     # RFE 10
        self._add_model(X, y, n_features=None, C=1.0)   # Full features, different C

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
        ('classifier', FinalEnsembleClassifier(threshold=0.35))
    ])
