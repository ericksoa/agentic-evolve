"""
Data loading utilities for MALLORN competition.

Handles the multi-split data format from Kaggle:
- split_XX/train_full_lightcurves.csv
- split_XX/test_full_lightcurves.csv

Column naming (based on MALLORN paper):
- object_id: Object identifier
- mjd: Modified Julian Date
- flux: Flux in microjanskys
- flux_err: Flux error in microjanskys
- band: Filter/band (u, g, r, i, z, y)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Standard column names we'll use internally
# MALLORN data has: object_id, Time (MJD), Flux, Flux_err, Filter
COLUMN_MAPPING = {
    # Possible names in raw data -> our standard name
    'object_id': 'object_id',
    'Object ID': 'object_id',
    'objectid': 'object_id',
    'id': 'object_id',

    'mjd': 'mjd',
    'MJD': 'mjd',
    'Date': 'mjd',
    'date': 'mjd',
    'time': 'mjd',
    'Time (MJD)': 'mjd',  # MALLORN format

    'flux': 'flux',
    'Flux': 'flux',

    'flux_err': 'flux_err',
    'Flux_err': 'flux_err',  # MALLORN format
    'flux_error': 'flux_err',
    'Flux Error': 'flux_err',
    'fluxerr': 'flux_err',

    'band': 'band',
    'Band': 'band',
    'filter': 'band',
    'Filter': 'band',  # MALLORN format
    'passband': 'band',
}

# Metadata column mappings
# MALLORN train_log.csv has: object_id, Z, Z_err, EBV, SpecType, English Translation, split, target
METADATA_MAPPING = {
    'object_id': 'object_id',
    'Object ID': 'object_id',

    'redshift': 'redshift',
    'Redshift': 'redshift',
    'z': 'redshift',
    'Z': 'redshift',  # MALLORN format
    'spec_z': 'redshift',

    'photo_z': 'photo_z',
    'photoz': 'photo_z',
    'photometric_redshift': 'photo_z',

    'photo_z_err': 'photo_z_err',
    'photoz_err': 'photo_z_err',
    'Z_err': 'photo_z_err',  # MALLORN format

    'ebv': 'ebv',
    'E(B-V)': 'ebv',
    'EBV': 'ebv',  # MALLORN format
    'extinction': 'ebv',

    'target': 'target',
    'Target': 'target',
    'class': 'target',
    'Class': 'target',
    'spectral_type': 'spectral_type',
    'SpecType': 'spectral_type',  # MALLORN format
    'type': 'target',

    'split': 'split',
}


def standardize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns to standard names."""
    rename_dict = {}
    for col in df.columns:
        if col in mapping:
            rename_dict[col] = mapping[col]
    return df.rename(columns=rename_dict)


def find_data_files(data_dir: str) -> Dict[str, List[Path]]:
    """
    Find all data files in the data directory.

    Returns dict with keys:
    - 'train_lc': list of training light curve files
    - 'test_lc': list of test light curve files
    - 'train_meta': list of training metadata files
    - 'test_meta': list of test metadata files
    """
    data_path = Path(data_dir)

    files = {
        'train_lc': [],
        'test_lc': [],
        'train_meta': [],
        'test_meta': [],
    }

    # Look for split directories
    for split_dir in sorted(data_path.glob('split_*')):
        train_lc = split_dir / 'train_full_lightcurves.csv'
        test_lc = split_dir / 'test_full_lightcurves.csv'
        train_meta = split_dir / 'train_metadata.csv'
        test_meta = split_dir / 'test_metadata.csv'

        if train_lc.exists():
            files['train_lc'].append(train_lc)
        if test_lc.exists():
            files['test_lc'].append(test_lc)
        if train_meta.exists():
            files['train_meta'].append(train_meta)
        if test_meta.exists():
            files['test_meta'].append(test_meta)

    # Also look for non-split files
    for pattern, key in [
        ('train*.csv', 'train_lc'),
        ('test*.csv', 'test_lc'),
        ('*metadata*.csv', 'train_meta'),
    ]:
        for f in data_path.glob(pattern):
            if 'light' in f.name.lower() or 'lc' in f.name.lower():
                if 'train' in f.name.lower():
                    files['train_lc'].append(f)
                elif 'test' in f.name.lower():
                    files['test_lc'].append(f)

    return files


def load_single_split(
    data_dir: str,
    split_num: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load data from a single split.

    MALLORN data structure:
    - train_log.csv: Global metadata with 'split' column
    - test_log.csv: Global test metadata
    - split_XX/train_full_lightcurves.csv: Training light curves
    - split_XX/test_full_lightcurves.csv: Test light curves

    Returns:
        (train_lc, test_lc, train_meta, test_meta)
    """
    data_path = Path(data_dir)
    split_name = f'split_{split_num:02d}'
    split_dir = data_path / split_name

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Load light curves
    train_lc = pd.read_csv(split_dir / 'train_full_lightcurves.csv')
    test_lc = pd.read_csv(split_dir / 'test_full_lightcurves.csv')

    train_lc = standardize_columns(train_lc, COLUMN_MAPPING)
    test_lc = standardize_columns(test_lc, COLUMN_MAPPING)

    # Load global metadata and filter by split
    train_meta = None
    test_meta = None

    train_log_path = data_path / 'train_log.csv'
    if train_log_path.exists():
        full_train_meta = pd.read_csv(train_log_path)
        full_train_meta = standardize_columns(full_train_meta, METADATA_MAPPING)

        # Filter to this split's objects
        if 'split' in full_train_meta.columns:
            train_meta = full_train_meta[full_train_meta['split'] == split_name].copy()
        else:
            # No split column - filter by objects in the light curves
            train_objects = train_lc['object_id'].unique()
            train_meta = full_train_meta[full_train_meta['object_id'].isin(train_objects)].copy()

    test_log_path = data_path / 'test_log.csv'
    if test_log_path.exists():
        full_test_meta = pd.read_csv(test_log_path)
        full_test_meta = standardize_columns(full_test_meta, METADATA_MAPPING)

        # Filter by objects in the light curves
        test_objects = test_lc['object_id'].unique()
        test_meta = full_test_meta[full_test_meta['object_id'].isin(test_objects)].copy()

    return train_lc, test_lc, train_meta, test_meta


def load_all_splits(
    data_dir: str,
    max_splits: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Load data from all available splits.

    Returns:
        List of (train_lc, test_lc, train_meta, test_meta) tuples
    """
    data_path = Path(data_dir)
    splits = []

    split_num = 1
    while True:
        if max_splits and split_num > max_splits:
            break

        split_dir = data_path / f'split_{split_num:02d}'
        if not split_dir.exists():
            break

        try:
            split_data = load_single_split(data_dir, split_num)
            splits.append(split_data)
        except Exception as e:
            print(f"Warning: Failed to load split {split_num}: {e}")

        split_num += 1

    return splits


def load_combined_training_data(
    data_dir: str,
    max_splits: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine training data from all splits.

    For the MALLORN challenge, each split represents a different
    train/test partition. We can use all training data combined
    for feature engineering.

    Returns:
        (combined_light_curves, combined_metadata)
    """
    splits = load_all_splits(data_dir, max_splits)

    all_lc = []
    all_meta = []

    for train_lc, _, train_meta, _ in splits:
        all_lc.append(train_lc)
        if train_meta is not None:
            all_meta.append(train_meta)

    combined_lc = pd.concat(all_lc, ignore_index=True).drop_duplicates()
    combined_meta = pd.concat(all_meta, ignore_index=True).drop_duplicates() if all_meta else None

    return combined_lc, combined_meta


def get_data_summary(data_dir: str) -> Dict:
    """Get summary statistics about the available data."""
    files = find_data_files(data_dir)

    summary = {
        'n_train_files': len(files['train_lc']),
        'n_test_files': len(files['test_lc']),
        'train_files': [str(f) for f in files['train_lc'][:5]],  # First 5
        'test_files': [str(f) for f in files['test_lc'][:5]],
    }

    # Try to load first split for more details
    if files['train_lc']:
        try:
            df = pd.read_csv(files['train_lc'][0], nrows=1000)
            summary['columns'] = list(df.columns)
            summary['sample_rows'] = len(df)
        except Exception as e:
            summary['load_error'] = str(e)

    return summary


def prepare_labels(
    metadata: pd.DataFrame,
    target_class: str = 'TDE'
) -> pd.Series:
    """
    Convert target column to binary labels.

    Args:
        metadata: DataFrame with 'target' column
        target_class: Positive class name

    Returns:
        Binary series (1 for target_class, 0 otherwise)
    """
    if 'target' not in metadata.columns:
        raise ValueError("Metadata must have 'target' column")

    labels = (metadata['target'] == target_class).astype(int)

    if 'object_id' in metadata.columns:
        labels.index = metadata['object_id']

    return labels


if __name__ == '__main__':
    # Quick test
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data'

    print("Data Summary:")
    print("-" * 40)
    summary = get_data_summary(data_dir)
    for k, v in summary.items():
        print(f"  {k}: {v}")
