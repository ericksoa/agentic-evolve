"""
Feature extraction for light curve classification.

This module contains feature extraction functions that can be evolved
to discover better representations for TDE identification.

The LSST bands are: u, g, r, i, z, y
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple


# =============================================================================
# BASELINE FEATURES (hand-crafted, to be improved by evolution)
# =============================================================================

def extract_baseline_features(light_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Extract baseline features from a light curve.

    Args:
        light_curve: DataFrame with columns [mjd, band, flux, flux_err]

    Returns:
        Dictionary of feature name -> value
    """
    features = {}

    # Per-band statistics
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    for band in bands:
        band_data = light_curve[light_curve['band'] == band]

        if len(band_data) < 2:
            # Not enough data in this band
            features[f'{band}_mean'] = np.nan
            features[f'{band}_std'] = np.nan
            features[f'{band}_max'] = np.nan
            features[f'{band}_min'] = np.nan
            features[f'{band}_range'] = np.nan
            features[f'{band}_n_obs'] = 0
            continue

        flux = band_data['flux'].values
        flux_err = band_data['flux_err'].values
        mjd = band_data['mjd'].values

        # Basic statistics
        features[f'{band}_mean'] = np.mean(flux)
        features[f'{band}_std'] = np.std(flux)
        features[f'{band}_max'] = np.max(flux)
        features[f'{band}_min'] = np.min(flux)
        features[f'{band}_range'] = np.max(flux) - np.min(flux)
        features[f'{band}_n_obs'] = len(flux)

        # Weighted mean (by inverse variance)
        weights = 1.0 / (flux_err ** 2 + 1e-10)
        features[f'{band}_wmean'] = np.average(flux, weights=weights)

        # Skewness and kurtosis
        if len(flux) >= 3:
            features[f'{band}_skew'] = stats.skew(flux)
            features[f'{band}_kurtosis'] = stats.kurtosis(flux)
        else:
            features[f'{band}_skew'] = np.nan
            features[f'{band}_kurtosis'] = np.nan

        # Time-based features
        if len(mjd) >= 2:
            # Rise/decay characteristics
            peak_idx = np.argmax(flux)
            features[f'{band}_time_to_peak'] = mjd[peak_idx] - mjd[0]
            features[f'{band}_time_from_peak'] = mjd[-1] - mjd[peak_idx]

            # Simple linear trend
            if len(mjd) >= 2:
                slope, intercept = np.polyfit(mjd - mjd[0], flux, 1)
                features[f'{band}_slope'] = slope

    # Cross-band features (colors)
    color_pairs = [('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'g')]
    for b1, b2 in color_pairs:
        if not np.isnan(features.get(f'{b1}_mean', np.nan)) and \
           not np.isnan(features.get(f'{b2}_mean', np.nan)):
            # Color at mean
            features[f'{b1}_{b2}_color'] = features[f'{b1}_mean'] - features[f'{b2}_mean']
            # Color evolution (difference in slopes)
            if f'{b1}_slope' in features and f'{b2}_slope' in features:
                features[f'{b1}_{b2}_color_slope'] = (
                    features.get(f'{b1}_slope', 0) - features.get(f'{b2}_slope', 0)
                )

    # Global features
    all_flux = light_curve['flux'].values
    all_mjd = light_curve['mjd'].values

    features['total_observations'] = len(all_flux)
    features['observation_span'] = all_mjd.max() - all_mjd.min() if len(all_mjd) > 0 else 0
    features['global_max'] = np.max(all_flux) if len(all_flux) > 0 else np.nan
    features['global_amplitude'] = np.ptp(all_flux) if len(all_flux) > 0 else np.nan

    return features


# =============================================================================
# EVOLVED FEATURES (placeholder - to be replaced by evolution)
# =============================================================================

def extract_evolved_features(light_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Evolved feature extraction function - Gen 4.

    Evolution history:
    - Gen 1: Baseline features (F1 = 0.276)
    - Gen 2: Added TDE physics features (power-law, colors, asymmetry) -> F1 = 0.368
    - Gen 3: Threshold optimization
    - Gen 4: Ensemble with optimized features -> F1 = 0.415 (+50% improvement)

    Args:
        light_curve: DataFrame with columns [mjd, band, flux, flux_err]

    Returns:
        Dictionary of feature name -> value
    """
    # Start with baseline features
    features = extract_baseline_features(light_curve)

    # =========================================================================
    # GEN 2: TDE-SPECIFIC PHYSICS FEATURES
    # =========================================================================

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    band_data_cache = {}

    # Cache band data for reuse
    for band in bands:
        bd = light_curve[light_curve['band'] == band].sort_values('mjd')
        if len(bd) >= 3:
            band_data_cache[band] = bd

    # --- 1. Power-law decay fitting (TDE signature: alpha ~ -5/3) ---
    for band in ['g', 'r', 'i']:
        if band not in band_data_cache:
            continue

        band_data = band_data_cache[band]
        flux = band_data['flux'].values
        flux_err = band_data['flux_err'].values
        mjd = band_data['mjd'].values

        peak_idx = np.argmax(flux)
        peak_flux = flux[peak_idx]
        peak_mjd = mjd[peak_idx]

        # Rise time and decay time
        features[f'{band}_rise_time'] = peak_mjd - mjd[0]
        features[f'{band}_decay_time'] = mjd[-1] - peak_mjd

        # Asymmetry ratio (TDEs have fast rise, slow decay)
        if features[f'{band}_decay_time'] > 0:
            features[f'{band}_asymmetry'] = features[f'{band}_rise_time'] / features[f'{band}_decay_time']
        else:
            features[f'{band}_asymmetry'] = np.nan

        # Power-law decay fitting
        if peak_idx < len(flux) - 3 and peak_flux > 0:
            post_peak_mjd = mjd[peak_idx:] - peak_mjd
            post_peak_flux = flux[peak_idx:]
            norm_flux = post_peak_flux / peak_flux

            try:
                def power_law(t, t0, alpha):
                    return (1 + t / (t0 + 1e-5)) ** alpha

                if norm_flux[-1] < 0.8:  # At least 20% decline
                    popt, pcov = curve_fit(
                        power_law,
                        post_peak_mjd[1:] + 0.1,
                        norm_flux[1:],
                        p0=[30, -1.5],
                        bounds=([1, -5], [500, 0]),
                        maxfev=1000
                    )
                    features[f'{band}_decay_t0'] = popt[0]
                    features[f'{band}_decay_alpha'] = popt[1]
                    features[f'{band}_tde_alpha_diff'] = abs(popt[1] - (-5/3))

                    # Fit quality (lower is better fit)
                    predicted = power_law(post_peak_mjd[1:] + 0.1, *popt)
                    residuals = norm_flux[1:] - predicted
                    features[f'{band}_decay_fit_rmse'] = np.sqrt(np.mean(residuals**2))
            except:
                pass

    # --- 2. Color evolution (TDEs are blue and evolve) ---
    color_pairs = [('g', 'r'), ('r', 'i'), ('u', 'g'), ('i', 'z')]

    for b1, b2 in color_pairs:
        if b1 not in band_data_cache or b2 not in band_data_cache:
            continue

        bd1 = band_data_cache[b1]
        bd2 = band_data_cache[b2]

        # Color at peak (using peak of bluer band)
        peak_mjd_b1 = bd1.loc[bd1['flux'].idxmax(), 'mjd']
        peak_flux_b1 = bd1['flux'].max()

        # Find flux in b2 closest to b1 peak time
        time_diff = np.abs(bd2['mjd'].values - peak_mjd_b1)
        closest_idx = np.argmin(time_diff)
        if time_diff[closest_idx] < 10:  # Within 10 days
            flux_b2_at_peak = bd2['flux'].iloc[closest_idx]
            if flux_b2_at_peak > 0 and peak_flux_b1 > 0:
                # Flux ratio (magnitude-like color)
                features[f'{b1}_{b2}_color_at_peak'] = -2.5 * np.log10(peak_flux_b1 / flux_b2_at_peak)

        # Color evolution rate (slope of color vs time)
        # Interpolate to common times
        common_mjds = np.intersect1d(
            np.round(bd1['mjd'].values),
            np.round(bd2['mjd'].values)
        )
        if len(common_mjds) >= 3:
            colors = []
            times = []
            for mjd_val in common_mjds:
                f1 = bd1.loc[np.abs(bd1['mjd'] - mjd_val) < 1, 'flux'].mean()
                f2 = bd2.loc[np.abs(bd2['mjd'] - mjd_val) < 1, 'flux'].mean()
                if f1 > 0 and f2 > 0:
                    colors.append(-2.5 * np.log10(f1 / f2))
                    times.append(mjd_val)

            if len(colors) >= 3:
                slope, _ = np.polyfit(times, colors, 1)
                features[f'{b1}_{b2}_color_slope'] = slope

    # --- 3. Smoothness metrics (TDEs are smooth, SNe have bumps) ---
    for band in ['g', 'r', 'i']:
        if band not in band_data_cache:
            continue

        bd = band_data_cache[band]
        flux = bd['flux'].values
        flux_err = bd['flux_err'].values
        mjd = bd['mjd'].values

        if len(flux) >= 5:
            # Fit polynomial and measure residuals
            try:
                # Normalize time
                t_norm = (mjd - mjd[0]) / (mjd[-1] - mjd[0] + 1e-5)

                # Fit quadratic (smooth curves should fit well)
                coeffs = np.polyfit(t_norm, flux, 2)
                predicted = np.polyval(coeffs, t_norm)
                residuals = flux - predicted

                # Reduced chi-squared (smoothness measure)
                chi_sq = np.sum((residuals / (flux_err + 1e-5))**2)
                dof = len(flux) - 3
                features[f'{band}_reduced_chi_sq'] = chi_sq / max(dof, 1)

                # Scatter relative to trend
                features[f'{band}_scatter'] = np.std(residuals) / (np.mean(np.abs(flux)) + 1e-5)

            except:
                pass

    # --- 4. Global TDE indicators ---

    # Combine alpha values across bands (TDE signature consistency)
    alpha_values = []
    for band in ['g', 'r', 'i']:
        key = f'{band}_decay_alpha'
        if key in features and not np.isnan(features[key]):
            alpha_values.append(features[key])

    if len(alpha_values) >= 2:
        features['mean_decay_alpha'] = np.mean(alpha_values)
        features['std_decay_alpha'] = np.std(alpha_values)
        features['tde_alpha_score'] = np.mean([abs(a - (-5/3)) for a in alpha_values])

    # Blue excess indicator (TDEs are blue)
    if 'g_mean' in features and 'r_mean' in features:
        g_mean = features.get('g_mean', 0)
        r_mean = features.get('r_mean', 0)
        if r_mean > 0 and g_mean > 0:
            features['blue_excess'] = g_mean / r_mean  # >1 means blue

    # Peak brightness relative to baseline
    for band in ['g', 'r']:
        if band in band_data_cache:
            bd = band_data_cache[band]
            flux = bd['flux'].values
            if len(flux) >= 10:
                # Use first/last 10% as baseline
                n_baseline = max(2, len(flux) // 10)
                baseline = np.median(np.concatenate([flux[:n_baseline], flux[-n_baseline:]]))
                peak = np.max(flux)
                if baseline > 0:
                    features[f'{band}_peak_to_baseline'] = peak / baseline

    return features


# =============================================================================
# FEATURE AGGREGATION
# =============================================================================

def extract_features_for_object(
    object_id: str,
    light_curves: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    use_evolved: bool = True
) -> Dict[str, float]:
    """
    Extract all features for a single astronomical object.

    Args:
        object_id: Unique identifier for the object
        light_curves: DataFrame with light curve data
        metadata: Optional metadata (redshift, etc.)
        use_evolved: Whether to use evolved features

    Returns:
        Dictionary of feature name -> value
    """
    # Filter to this object
    obj_lc = light_curves[light_curves['object_id'] == object_id].copy()

    # Extract light curve features
    if use_evolved:
        features = extract_evolved_features(obj_lc)
    else:
        features = extract_baseline_features(obj_lc)

    # Add metadata features if available
    if metadata is not None and object_id in metadata.index:
        obj_meta = metadata.loc[object_id]
        if 'photo_z' in obj_meta:
            features['photo_z'] = obj_meta['photo_z']
        if 'photo_z_err' in obj_meta:
            features['photo_z_err'] = obj_meta['photo_z_err']

    return features


def extract_features_batch(
    light_curves: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    use_evolved: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract features for all objects in the dataset.

    Args:
        light_curves: DataFrame with all light curve data
        metadata: Optional metadata DataFrame
        use_evolved: Whether to use evolved features
        verbose: Show progress bar

    Returns:
        DataFrame with one row per object, columns are features
    """
    from tqdm import tqdm

    object_ids = light_curves['object_id'].unique()

    all_features = []
    iterator = tqdm(object_ids, desc="Extracting features") if verbose else object_ids

    for obj_id in iterator:
        features = extract_features_for_object(
            obj_id, light_curves, metadata, use_evolved
        )
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features).set_index('object_id')
