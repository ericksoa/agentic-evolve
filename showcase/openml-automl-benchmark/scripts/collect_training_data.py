#!/usr/bin/env python3
"""
Collect training data for the meta-learning detector.

Downloads binary classification datasets from OpenML and evaluates
whether threshold optimization helps on each one.

Usage:
    python scripts/collect_training_data.py [--max-datasets N] [--output PATH]
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    import openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False
    print("Warning: openml not installed. Install with: pip install openml")

from adaptive_ensemble.meta_learning.training import (
    collect_training_data,
    save_training_data,
)


def get_binary_datasets_from_openml(
    max_datasets: int = 100,
    min_samples: int = 100,
    max_samples: int = 50000,
    min_features: int = 2,
    max_features: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Fetch binary classification datasets from OpenML.

    Returns list of (X, y, name) tuples.
    """
    if not HAS_OPENML:
        raise ImportError("openml is required. Install with: pip install openml")

    print("Fetching dataset list from OpenML...")

    # Get list of datasets
    datasets = openml.datasets.list_datasets(output_format='dataframe')

    # Filter for binary classification
    binary_datasets = datasets[
        (datasets['NumberOfClasses'] == 2) &
        (datasets['NumberOfInstances'] >= min_samples) &
        (datasets['NumberOfInstances'] <= max_samples) &
        (datasets['NumberOfFeatures'] >= min_features) &
        (datasets['NumberOfFeatures'] <= max_features)
    ]

    # Sort by popularity (number of runs) and take top N
    if 'NumberOfDownloads' in binary_datasets.columns:
        binary_datasets = binary_datasets.sort_values('NumberOfDownloads', ascending=False)
    else:
        binary_datasets = binary_datasets.sort_values('did')

    binary_datasets = binary_datasets.head(max_datasets * 2)  # Get extra in case some fail

    print(f"Found {len(binary_datasets)} candidate binary datasets")

    loaded_datasets = []
    for _, row in binary_datasets.iterrows():
        if len(loaded_datasets) >= max_datasets:
            break

        dataset_id = row['did']
        name = row['name']

        try:
            print(f"  Loading {name} (id={dataset_id})...", end=' ')

            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format='array'
            )

            # Convert to numpy
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)

            # Handle NaN values
            if np.any(np.isnan(X)):
                # Simple imputation: replace NaN with column mean
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[np.isnan(X[:, i]), i] = col_means[i]

            # Encode labels to 0/1 if needed
            unique_labels = np.unique(y)
            if len(unique_labels) != 2:
                print("skipped (not binary)")
                continue

            if not np.all(np.isin(unique_labels, [0, 1])):
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                y = np.array([label_map[label] for label in y])

            print(f"OK ({X.shape[0]} samples, {X.shape[1]} features)")
            loaded_datasets.append((X, y, name))

        except Exception as e:
            print(f"error: {e}")
            continue

    print(f"\nSuccessfully loaded {len(loaded_datasets)} datasets")
    return loaded_datasets


def get_fallback_datasets() -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Generate synthetic datasets for testing when OpenML is not available.
    """
    from sklearn.datasets import make_classification

    print("Generating synthetic datasets for testing...")
    datasets = []

    configs = [
        # (n_samples, n_features, weights, name)
        (500, 10, [0.7, 0.3], "synth_imbalanced_small"),
        (1000, 20, [0.6, 0.4], "synth_moderate_medium"),
        (2000, 15, [0.5, 0.5], "synth_balanced_medium"),
        (500, 5, [0.8, 0.2], "synth_highly_imbalanced"),
        (1500, 30, [0.65, 0.35], "synth_many_features"),
        (800, 8, [0.55, 0.45], "synth_nearly_balanced"),
        (300, 12, [0.75, 0.25], "synth_small_imbalanced"),
        (2500, 25, [0.6, 0.4], "synth_larger"),
    ]

    for i, (n_samples, n_features, weights, name) in enumerate(configs):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features // 2, 10),
            n_redundant=min(n_features // 4, 5),
            weights=weights,
            random_state=42 + i,
        )
        datasets.append((X, y, name))
        print(f"  Created {name}: {n_samples} samples, {n_features} features")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Collect meta-learning training data")
    parser.add_argument('--max-datasets', type=int, default=50,
                        help="Maximum number of datasets to use")
    parser.add_argument('--output', type=str, default='meta_training_data.pkl',
                        help="Output file path")
    parser.add_argument('--use-synthetic', action='store_true',
                        help="Use synthetic datasets instead of OpenML")
    parser.add_argument('--cv', type=int, default=5,
                        help="CV folds for evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("Meta-Learning Training Data Collection")
    print("=" * 60)

    # Load datasets
    if args.use_synthetic or not HAS_OPENML:
        datasets = get_fallback_datasets()
    else:
        datasets = get_binary_datasets_from_openml(max_datasets=args.max_datasets)

    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    # Collect training data
    print("\n" + "=" * 60)
    print("Evaluating threshold optimization benefit...")
    print("=" * 60)

    training_data = collect_training_data(
        datasets,
        cv=args.cv,
        significance_threshold=0.01,
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total datasets: {len(training_data['y'])}")
    print(f"Datasets where optimization helped: {training_data['y'].sum()}")
    print(f"Positive rate: {training_data['y'].mean():.1%}")

    # Save
    save_training_data(training_data, args.output)
    print(f"\nTraining data saved to: {args.output}")
    print(f"CSV version saved to: {args.output.replace('.pkl', '.csv')}")


if __name__ == '__main__':
    main()
