"""
Baseline solution for diabetes evolution.

This represents our current best: Feature Selection (RFE) with 10 features + LogReg.
Holdout F1: 0.690

The evolution should try to improve on this by:
1. Combining feature selection with SMOTE
2. Adding calibration
3. Fine-tuning parameters
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add domain-specific features for diabetes prediction.

    Features added:
    - Domain features: BMI categories, glucose thresholds
    - Binned features: Quantile-based bins for key predictors
    """

    def __init__(self, add_domain=True, add_bins=True, n_bins=4):
        self.add_domain = add_domain
        self.add_bins = add_bins
        self.n_bins = n_bins
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        if self.add_bins:
            # Fit quantile bins on each column
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
            # Assuming standard diabetes dataset columns:
            # 0: pregnancies, 1: glucose, 2: blood_pressure, 3: skin_thickness,
            # 4: insulin, 5: bmi, 6: diabetes_pedigree, 7: age

            if X.shape[1] >= 8:
                glucose = X[:, 1]
                bmi = X[:, 5]
                age = X[:, 7]

                # Domain features
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


def create_pipeline():
    """
    Create the diabetes classification pipeline.

    Current best configuration:
    - Domain + Bins feature engineering
    - RFE feature selection to 10 features
    - LogReg with C=0.5, balanced weights

    Returns sklearn Pipeline.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_engineer', DiabetesFeatureEngineer(add_domain=True, add_bins=True, n_bins=4)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(
            estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            n_features_to_select=10,
            step=1
        )),
        ('classifier', LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])


# For testing
if __name__ == "__main__":
    import openml
    from sklearn.model_selection import cross_val_score

    # Load data
    dataset = openml.datasets.get_dataset(37)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    X = X.select_dtypes(include=[np.number]).values

    # Test pipeline
    pipeline = create_pipeline()
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    print(f"5-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
