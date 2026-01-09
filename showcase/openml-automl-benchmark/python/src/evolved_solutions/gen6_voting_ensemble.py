"""
Gen 6c: Voting Ensemble

Soft voting ensemble of our best approaches:
1. LogReg with threshold 0.35
2. LogReg with RFE8 + threshold 0.35
3. LogReg with medical features

This mimics Auto-sklearn's ensemble selection but simpler.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Standard domain + bins feature engineering."""

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


class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Soft voting ensemble with threshold optimization.

    Combines probabilities from multiple LogReg models with different
    feature sets, then applies optimized threshold.
    """

    def __init__(self, threshold=0.38):
        self.threshold = threshold
        self.models_ = []
        self.feature_engineers_ = []
        self.scalers_ = []
        self.rfe_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Model 1: Standard features
        fe1 = DiabetesFeatureEngineer(n_bins=4)
        X1 = fe1.fit_transform(X)
        scaler1 = StandardScaler()
        X1_scaled = scaler1.fit_transform(X1)
        model1 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model1.fit(X1_scaled, y)

        self.feature_engineers_.append(fe1)
        self.scalers_.append(scaler1)
        self.models_.append(model1)

        # Model 2: RFE to 8 features
        fe2 = DiabetesFeatureEngineer(n_bins=4)
        X2 = fe2.fit_transform(X)
        scaler2 = StandardScaler()
        X2_scaled = scaler2.fit_transform(X2)
        rfe = RFE(LogisticRegression(C=1.0, max_iter=1000, random_state=42), n_features_to_select=8)
        X2_rfe = rfe.fit_transform(X2_scaled, y)
        model2 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model2.fit(X2_rfe, y)

        self.feature_engineers_.append(fe2)
        self.scalers_.append(scaler2)
        self.rfe_ = rfe
        self.models_.append(model2)

        # Model 3: Different regularization
        fe3 = DiabetesFeatureEngineer(n_bins=4)
        X3 = fe3.fit_transform(X)
        scaler3 = StandardScaler()
        X3_scaled = scaler3.fit_transform(X3)
        model3 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        model3.fit(X3_scaled, y)

        self.feature_engineers_.append(fe3)
        self.scalers_.append(scaler3)
        self.models_.append(model3)

        return self

    def predict_proba(self, X):
        # Model 1: Standard
        X1 = self.feature_engineers_[0].transform(X)
        X1_scaled = self.scalers_[0].transform(X1)
        proba1 = self.models_[0].predict_proba(X1_scaled)

        # Model 2: RFE
        X2 = self.feature_engineers_[1].transform(X)
        X2_scaled = self.scalers_[1].transform(X2)
        X2_rfe = self.rfe_.transform(X2_scaled)
        proba2 = self.models_[1].predict_proba(X2_rfe)

        # Model 3: Different C
        X3 = self.feature_engineers_[2].transform(X)
        X3_scaled = self.scalers_[2].transform(X3)
        proba3 = self.models_[2].predict_proba(X3_scaled)

        # Weighted average (equal weights)
        avg_proba = (proba1 + proba2 + proba3) / 3.0

        return avg_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)


def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', VotingEnsembleClassifier(threshold=0.38))
    ])
