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
    Evolved feature extraction function.

    This function will be replaced/augmented by the /evolve skill
    to discover novel feature representations.

    Args:
        light_curve: DataFrame with columns [mjd, band, flux, flux_err]

    Returns:
        Dictionary of feature name -> value
    """
    # Start with baseline features
    features = extract_baseline_features(light_curve)

    # Add evolved features here
    # Example evolved features that could help identify TDEs:

    # TDE-specific: they have characteristic -5/3 power law decay
    # Try to fit this and use fit quality as a feature

    bands_with_data = []
    for band in ['g', 'r', 'i']:
        band_data = light_curve[light_curve['band'] == band]
        if len(band_data) >= 5:
            bands_with_data.append((band, band_data))

    for band, band_data in bands_with_data:
        flux = band_data['flux'].values
        mjd = band_data['mjd'].values

        # Find peak and fit decay
        peak_idx = np.argmax(flux)

        if peak_idx < len(flux) - 3:
            # Have post-peak data to fit
            post_peak_mjd = mjd[peak_idx:] - mjd[peak_idx]
            post_peak_flux = flux[peak_idx:]

            # Normalize flux
            peak_flux = flux[peak_idx]
            if peak_flux > 0:
                norm_flux = post_peak_flux / peak_flux

                # Try power law fit: F(t) = (1 + t/t0)^alpha
                # TDEs have alpha ~ -5/3
                try:
                    def power_law(t, t0, alpha):
                        return (1 + t / (t0 + 1e-5)) ** alpha

                    # Only fit if we have enough decline
                    if norm_flux[-1] < 0.8:  # At least 20% decline
                        popt, _ = curve_fit(
                            power_law,
                            post_peak_mjd[1:] + 0.1,  # Avoid t=0
                            norm_flux[1:],
                            p0=[30, -1.5],
                            bounds=([1, -5], [500, 0]),
                            maxfev=1000
                        )
                        features[f'{band}_decay_t0'] = popt[0]
                        features[f'{band}_decay_alpha'] = popt[1]

                        # How close to TDE -5/3 power law?
                        features[f'{band}_tde_alpha_diff'] = abs(popt[1] - (-5/3))
                except:
                    features[f'{band}_decay_t0'] = np.nan
                    features[f'{band}_decay_alpha'] = np.nan
                    features[f'{band}_tde_alpha_diff'] = np.nan

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
