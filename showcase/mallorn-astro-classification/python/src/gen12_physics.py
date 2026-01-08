#!/usr/bin/env python3
"""
Gen12: Enhanced Physics-Based Features

Adds new TDE physics features beyond Gen11:
1. Cross-band alpha consistency (normalized std)
2. Color temperature evolution (early - late)
3. Baseline variability (MAD of quiescent periods)
4. Rise/decay consistency across bands
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Optional


# Top features for polynomial interactions (from Gen11)
TOP_FEATURES = [
    'g_skew', 'r_scatter', 'r_skew', 'i_skew',
    'i_kurtosis', 'r_kurtosis'
]


def extract_physics_features(light_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Extract enhanced physics features for TDE classification.

    These features go beyond Gen2's evolved features to capture
    more nuanced TDE signatures.
    """
    features = {}
    bands = ['g', 'r', 'i']

    # Cache band data
    band_data_cache = {}
    for band in bands:
        bd = light_curve[light_curve['band'] == band].sort_values('mjd')
        if len(bd) >= 5:
            band_data_cache[band] = bd

    # --- 1. Cross-band alpha consistency ---
    alpha_values = []
    decay_times = []
    rise_times = []

    for band in bands:
        if band not in band_data_cache:
            continue

        bd = band_data_cache[band]
        flux = bd['flux'].values
        mjd = bd['mjd'].values

        peak_idx = np.argmax(flux)
        peak_flux = flux[peak_idx]
        peak_mjd = mjd[peak_idx]

        rise_time = peak_mjd - mjd[0]
        decay_time = mjd[-1] - peak_mjd

        rise_times.append(rise_time)
        decay_times.append(decay_time)

        # Fit power-law to decay
        if peak_idx < len(flux) - 3 and peak_flux > 0 and decay_time > 10:
            post_peak_mjd = mjd[peak_idx:] - peak_mjd
            post_peak_flux = flux[peak_idx:]
            norm_flux = post_peak_flux / peak_flux

            try:
                def power_law(t, t0, alpha):
                    return (1 + t / (t0 + 1e-5)) ** alpha

                if norm_flux[-1] < 0.9:  # At least 10% decline
                    popt, _ = curve_fit(
                        power_law,
                        post_peak_mjd[1:] + 0.1,
                        norm_flux[1:],
                        p0=[30, -1.5],
                        bounds=([1, -5], [500, 0]),
                        maxfev=500
                    )
                    alpha_values.append(popt[1])
            except:
                pass

    # Alpha consistency metrics
    if len(alpha_values) >= 2:
        mean_alpha = np.mean(alpha_values)
        std_alpha = np.std(alpha_values)
        median_alpha = np.median(alpha_values)

        # Normalized consistency (lower = more consistent)
        features['alpha_consistency'] = std_alpha / (abs(mean_alpha) + 1e-5)

        # Agreement score (how many bands agree with median)
        agreement = sum(abs(a - median_alpha) < 0.3 for a in alpha_values)
        features['alpha_agreement'] = agreement / len(alpha_values)

        # TDE signature strength (closer to -5/3 is better)
        features['tde_signature'] = 1.0 / (abs(mean_alpha - (-5/3)) + 0.1)

    # Rise/decay consistency
    if len(rise_times) >= 2 and len(decay_times) >= 2:
        features['rise_consistency'] = 1.0 - (np.std(rise_times) / (np.mean(rise_times) + 1e-5))
        features['decay_consistency'] = 1.0 - (np.std(decay_times) / (np.mean(decay_times) + 1e-5))

    # --- 2. Color temperature evolution (early vs late) ---
    color_pairs = [('g', 'r'), ('r', 'i')]

    for b1, b2 in color_pairs:
        if b1 not in band_data_cache or b2 not in band_data_cache:
            continue

        bd1 = band_data_cache[b1]
        bd2 = band_data_cache[b2]

        mjd1, flux1 = bd1['mjd'].values, bd1['flux'].values
        mjd2, flux2 = bd2['mjd'].values, bd2['flux'].values

        # Early phase (first 25% of observations)
        n_early = max(3, len(mjd1) // 4)
        early_flux1 = np.median(flux1[:n_early])

        # Find corresponding early flux in b2
        early_mjd_range = mjd1[n_early-1]
        early_mask = mjd2 <= early_mjd_range
        if early_mask.sum() >= 2:
            early_flux2 = np.median(flux2[early_mask])

            if early_flux1 > 0 and early_flux2 > 0:
                early_color = -2.5 * np.log10(early_flux1 / early_flux2)
            else:
                early_color = None
        else:
            early_color = None

        # Late phase (last 25% of observations)
        late_start = max(0, len(mjd1) - len(mjd1) // 4)
        late_flux1 = np.median(flux1[late_start:])

        late_mjd_start = mjd1[late_start]
        late_mask = mjd2 >= late_mjd_start
        if late_mask.sum() >= 2:
            late_flux2 = np.median(flux2[late_mask])

            if late_flux1 > 0 and late_flux2 > 0:
                late_color = -2.5 * np.log10(late_flux1 / late_flux2)
            else:
                late_color = None
        else:
            late_color = None

        # Temperature evolution (TDEs cool: early should be bluer)
        if early_color is not None and late_color is not None:
            features[f'{b1}{b2}_temp_evolution'] = late_color - early_color

    # --- 3. Baseline variability (TDEs from quiescent hosts) ---
    for band in bands:
        if band not in band_data_cache:
            continue

        bd = band_data_cache[band]
        flux = bd['flux'].values

        if len(flux) >= 10:
            # First and last 15% as baseline
            n_baseline = max(2, len(flux) // 7)  # ~15%
            baseline_flux = np.concatenate([flux[:n_baseline], flux[-n_baseline:]])

            # Median Absolute Deviation (robust variability measure)
            mad = np.median(np.abs(baseline_flux - np.median(baseline_flux)))
            features[f'{band}_baseline_mad'] = mad

            # Baseline relative to peak
            peak_flux = np.max(flux)
            if peak_flux > 0:
                features[f'{band}_baseline_ratio'] = np.median(baseline_flux) / peak_flux

    # Combined baseline stability
    mad_values = [features.get(f'{b}_baseline_mad', np.nan) for b in bands]
    mad_values = [v for v in mad_values if not np.isnan(v)]
    if mad_values:
        features['baseline_stability'] = 1.0 / (np.mean(mad_values) + 1e-5)

    # --- 4. Flux ratio consistency ---
    # TDEs should have consistent flux ratios across bands (spectral shape)
    mean_fluxes = []
    for band in bands:
        if band in band_data_cache:
            mean_fluxes.append(band_data_cache[band]['flux'].mean())

    if len(mean_fluxes) >= 2:
        # Normalized std of flux ratios
        mean_fluxes = np.array(mean_fluxes)
        features['spectral_consistency'] = 1.0 - (np.std(mean_fluxes) / (np.mean(mean_fluxes) + 1e-5))

    return features


class Gen12_Physics(BaseEstimator, ClassifierMixin):
    """
    Gen 12: LogReg + Gen11 features + enhanced physics features.

    Combines Gen11's polynomial interactions with new physics-based
    features designed to capture TDE signatures more precisely.
    """

    def __init__(self, threshold: float = 0.43, C: float = 0.05):
        self.threshold = threshold
        self.C = C
        self.lr_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.poly_ = None
        self.feature_names_ = None
        self.top_feature_indices_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Gen12_Physics':
        """Fit with Gen11 features + polynomial interactions."""
        self.feature_names_ = list(X.columns)
        self.top_feature_indices_ = [
            i for i, col in enumerate(self.feature_names_)
            if col in TOP_FEATURES
        ]

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imputed)

        # Polynomial features on top predictors
        if len(self.top_feature_indices_) >= 2:
            X_top = X_scaled[:, self.top_feature_indices_]
            self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly_.fit_transform(X_top)
            n_original = len(self.top_feature_indices_)
            X_final = np.hstack([X_scaled, X_poly[:, n_original:]])
        else:
            X_final = X_scaled

        # Fit LogReg
        self.lr_ = LogisticRegression(
            class_weight='balanced',
            C=self.C,
            max_iter=1000,
            random_state=42
        )
        self.lr_.fit(X_final, y)
        return self

    def _transform(self, X):
        X_imputed = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imputed)
        if self.poly_ is not None:
            X_top = X_scaled[:, self.top_feature_indices_]
            X_poly = self.poly_.transform(X_top)
            n_original = len(self.top_feature_indices_)
            return np.hstack([X_scaled, X_poly[:, n_original:]])
        return X_scaled

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._transform(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


# Aliases
TDEClassifier = Gen12_Physics
Gen12_Candidate = Gen12_Physics
