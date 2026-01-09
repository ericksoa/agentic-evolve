"""
Gen 5: Medical Domain Expert Feature Engineering

This solution applies clinical diabetes knowledge, not just statistical transformations.

Clinical knowledge applied:
1. WHO/ADA diagnostic thresholds for glucose
2. Clinical BMI categories and obesity risk curves
3. Known diabetes risk factor interactions
4. Insulin resistance markers
5. Pregnancy-related diabetes risk (gestational diabetes history)
6. Age-stratified risk adjustments
7. FINDRISC-inspired risk components

Dataset columns (Pima Indians Diabetes):
0: preg - Number of pregnancies
1: plas - Plasma glucose (2-hour OGTT, mg/dL)
2: pres - Diastolic blood pressure (mm Hg)
3: skin - Triceps skin fold thickness (mm) - proxy for body fat
4: insu - 2-hour serum insulin (μU/ml)
5: mass - BMI (kg/m²)
6: pedi - Diabetes pedigree function (genetic risk)
7: age  - Age (years)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class MedicalDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering based on clinical diabetes knowledge.

    This goes beyond statistical binning to apply actual medical thresholds
    and known risk factor interactions.
    """

    def __init__(self):
        self.glucose_mean_ = None
        self.insulin_mean_ = None
        self.age_mean_ = None

    def fit(self, X, y=None):
        # Store means for imputation in interaction features
        self.glucose_mean_ = np.nanmean(X[:, 1])
        self.insulin_mean_ = np.nanmean(X[:, 4])
        self.age_mean_ = np.nanmean(X[:, 7])
        return self

    def transform(self, X):
        # Extract columns with clinical names
        preg = X[:, 0]
        glucose = X[:, 1]
        bp = X[:, 2]
        skin = X[:, 3]
        insulin = X[:, 4]
        bmi = X[:, 5]
        pedigree = X[:, 6]
        age = X[:, 7]

        features = [X]  # Keep original features

        # =================================================================
        # 1. CLINICAL GLUCOSE THRESHOLDS (WHO/ADA criteria for OGTT)
        # =================================================================
        # Normal: <140 mg/dL, Prediabetes: 140-199, Diabetes: ≥200
        glucose_normal = (glucose < 140).astype(float)
        glucose_prediabetic = ((glucose >= 140) & (glucose < 200)).astype(float)
        glucose_diabetic = (glucose >= 200).astype(float)

        # Impaired fasting glucose zones (assuming some fasting component)
        glucose_high_normal = ((glucose >= 100) & (glucose < 140)).astype(float)

        features.extend([
            glucose_normal.reshape(-1, 1),
            glucose_prediabetic.reshape(-1, 1),
            glucose_diabetic.reshape(-1, 1),
            glucose_high_normal.reshape(-1, 1),
        ])

        # =================================================================
        # 2. CLINICAL BMI CATEGORIES (WHO classification)
        # =================================================================
        # Underweight: <18.5, Normal: 18.5-25, Overweight: 25-30
        # Obese I: 30-35, Obese II: 35-40, Obese III: ≥40
        bmi_underweight = (bmi < 18.5).astype(float)
        bmi_normal = ((bmi >= 18.5) & (bmi < 25)).astype(float)
        bmi_overweight = ((bmi >= 25) & (bmi < 30)).astype(float)
        bmi_obese_1 = ((bmi >= 30) & (bmi < 35)).astype(float)
        bmi_obese_2 = ((bmi >= 35) & (bmi < 40)).astype(float)
        bmi_obese_3 = (bmi >= 40).astype(float)

        # Combined severe obesity (clinical cutoff for high risk)
        bmi_severe_obese = (bmi >= 35).astype(float)

        features.extend([
            bmi_normal.reshape(-1, 1),
            bmi_overweight.reshape(-1, 1),
            bmi_obese_1.reshape(-1, 1),
            bmi_severe_obese.reshape(-1, 1),
        ])

        # =================================================================
        # 3. AGE-BASED RISK STRATIFICATION
        # =================================================================
        # ADA recommends screening at 45+, risk increases significantly
        age_young = (age < 30).astype(float)  # Lower risk baseline
        age_middle = ((age >= 30) & (age < 45)).astype(float)  # Increasing risk
        age_screening = (age >= 45).astype(float)  # ADA screening threshold
        age_high_risk = (age >= 55).astype(float)  # High risk

        features.extend([
            age_young.reshape(-1, 1),
            age_screening.reshape(-1, 1),
            age_high_risk.reshape(-1, 1),
        ])

        # =================================================================
        # 4. INSULIN RESISTANCE MARKERS
        # =================================================================
        # HOMA-IR proxy: Glucose × Insulin / 405 (clinical formula)
        # Higher values indicate insulin resistance
        insulin_safe = np.where(insulin > 0, insulin, self.insulin_mean_)
        glucose_safe = np.where(glucose > 0, glucose, self.glucose_mean_)

        homa_proxy = (glucose_safe * insulin_safe) / 405.0
        homa_high = (homa_proxy > 2.5).astype(float)  # Clinical cutoff for IR

        # Glucose/Insulin ratio - low ratio suggests insulin resistance
        gi_ratio = glucose_safe / np.maximum(insulin_safe, 1)
        gi_ratio_low = (gi_ratio < 6).astype(float)  # Suggests hyperinsulinemia

        features.extend([
            np.clip(homa_proxy, 0, 20).reshape(-1, 1),  # Capped HOMA proxy
            homa_high.reshape(-1, 1),
            gi_ratio_low.reshape(-1, 1),
        ])

        # =================================================================
        # 5. PREGNANCY-RELATED RISK (Gestational Diabetes History)
        # =================================================================
        # Multiple pregnancies, especially with high glucose, suggests GDM history
        # GDM increases T2D risk 7-fold
        high_parity = (preg >= 4).astype(float)
        any_pregnancy = (preg > 0).astype(float)

        # Young + high parity = higher chance of GDM history
        young_multiparous = ((age < 35) & (preg >= 3)).astype(float)

        # Pregnancy + elevated glucose = likely GDM
        preg_with_high_glucose = ((preg > 0) & (glucose >= 140)).astype(float)

        features.extend([
            high_parity.reshape(-1, 1),
            young_multiparous.reshape(-1, 1),
            preg_with_high_glucose.reshape(-1, 1),
        ])

        # =================================================================
        # 6. METABOLIC SYNDROME INDICATORS
        # =================================================================
        # Blood pressure component (hypertension)
        bp_elevated = (bp >= 85).astype(float)  # Elevated diastolic
        bp_high = (bp >= 90).astype(float)  # Stage 1 hypertension

        # Central obesity proxy (skin fold + BMI)
        skin_safe = np.where(skin > 0, skin, np.nanmean(skin))
        central_obesity_proxy = ((bmi >= 30) & (skin_safe > 30)).astype(float)

        features.extend([
            bp_elevated.reshape(-1, 1),
            central_obesity_proxy.reshape(-1, 1),
        ])

        # =================================================================
        # 7. CRITICAL INTERACTION FEATURES (Clinical Knowledge)
        # =================================================================

        # Glucose × Age interaction: High glucose is MORE concerning in younger patients
        # (suggests earlier onset, more aggressive disease)
        glucose_age_risk = ((glucose_prediabetic.astype(bool)) | (glucose_diabetic.astype(bool))) & (age < 40)

        # BMI × Glucose: Obese + high glucose = very high risk
        metabolic_double_risk = ((bmi >= 30) & (glucose >= 140)).astype(float)

        # Pedigree × Glucose: Family history + elevated glucose
        genetic_glucose_risk = ((pedigree > 0.5) & (glucose >= 120)).astype(float)

        # Age × BMI × Glucose triple interaction (metabolic syndrome proxy)
        triple_risk = ((age >= 40) & (bmi >= 28) & (glucose >= 120)).astype(float)

        features.extend([
            glucose_age_risk.astype(float).reshape(-1, 1),
            metabolic_double_risk.reshape(-1, 1),
            genetic_glucose_risk.reshape(-1, 1),
            triple_risk.reshape(-1, 1),
        ])

        # =================================================================
        # 8. FINDRISC-INSPIRED COMPOSITE SCORE
        # =================================================================
        # Simplified FINDRISC components we can approximate
        findrisc_proxy = (
            (age >= 45).astype(float) * 2 +
            (age >= 55).astype(float) * 1 +  # Additional for 55+
            (bmi >= 25).astype(float) * 1 +
            (bmi >= 30).astype(float) * 2 +  # Additional for obese
            (glucose >= 140).astype(float) * 5 +  # History of high glucose
            (bp >= 90).astype(float) * 2 +
            (pedigree > 0.5).astype(float) * 3  # Family history proxy
        )

        findrisc_high = (findrisc_proxy >= 7).astype(float)
        findrisc_very_high = (findrisc_proxy >= 12).astype(float)

        features.extend([
            findrisc_proxy.reshape(-1, 1),
            findrisc_high.reshape(-1, 1),
            findrisc_very_high.reshape(-1, 1),
        ])

        # =================================================================
        # 9. NON-LINEAR TRANSFORMATIONS (Clinically Motivated)
        # =================================================================
        # Log transforms for skewed clinical measurements
        glucose_log = np.log1p(glucose_safe)
        insulin_log = np.log1p(insulin_safe)
        pedigree_log = np.log1p(pedigree * 10)  # Scale up pedigree

        features.extend([
            glucose_log.reshape(-1, 1),
            insulin_log.reshape(-1, 1),
            pedigree_log.reshape(-1, 1),
        ])

        # Combine all features
        result = np.hstack(features)
        result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)

        return result


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Classifier with optimized decision threshold for imbalanced data."""

    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.base_estimator = LogisticRegression(
            C=0.5,
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
    """
    Medical domain expert pipeline.

    Uses clinical diabetes knowledge for feature engineering,
    not just statistical transformations.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('medical_features', MedicalDomainFeatureEngineer()),
        ('scaler', StandardScaler()),
        ('classifier', ThresholdClassifier(threshold=0.35))
    ])
