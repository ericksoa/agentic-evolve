#!/usr/bin/env python3
"""
Generate Kaggle submission for MALLORN competition.

Usage:
    python submit.py                     # Generate submission with best model
    python submit.py --model evolved     # Use evolved classifier
    python submit.py --threshold 0.3     # Custom classification threshold
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_batch
from classifier import TDEClassifier, EvolvedTDEClassifier


def load_test_data(data_dir: str = '../data'):
    """Load test data for submission."""
    data_path = Path(data_dir)
    test_lc = pd.read_csv(data_path / 'test.csv')
    test_meta = pd.read_csv(data_path / 'test_metadata.csv')
    return test_lc, test_meta


def load_train_data(data_dir: str = '../data'):
    """Load training data."""
    data_path = Path(data_dir)
    train_lc = pd.read_csv(data_path / 'train.csv')
    train_meta = pd.read_csv(data_path / 'train_metadata.csv')
    return train_lc, train_meta


def generate_submission(
    model_type: str = 'xgboost',
    threshold: float = 0.5,
    use_evolved: bool = False,
    data_dir: str = '../data',
    output_file: str = 'submission.csv'
):
    """
    Generate Kaggle submission file.

    Args:
        model_type: Type of classifier to use
        threshold: Classification threshold
        use_evolved: Use evolved classifier
        data_dir: Path to data directory
        output_file: Output CSV filename
    """
    print("=" * 60)
    print("Generating MALLORN Submission")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    train_lc, train_meta = load_train_data(data_dir)

    print("Loading test data...")
    test_lc, test_meta = load_test_data(data_dir)

    # Prepare training labels
    y_train = (train_meta['target'] == 'TDE').astype(int)
    y_train.index = train_meta['object_id']

    # Extract features
    print("\nExtracting training features...")
    X_train = extract_features_batch(
        train_lc,
        metadata=train_meta.set_index('object_id'),
        use_evolved=use_evolved,
        verbose=True
    )

    print("Extracting test features...")
    X_test = extract_features_batch(
        test_lc,
        metadata=test_meta.set_index('object_id') if 'object_id' in test_meta.columns else None,
        use_evolved=use_evolved,
        verbose=True
    )

    # Align training data
    common_ids = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_ids]
    y_train = y_train.loc[common_ids]

    # Create classifier
    print(f"\nTraining classifier (type={model_type}, threshold={threshold})...")
    if use_evolved:
        classifier = EvolvedTDEClassifier()
    else:
        classifier = TDEClassifier(
            model_type=model_type,
            threshold=threshold
        )

    classifier.fit(X_train, y_train)

    # Predict on test set
    print("Generating predictions...")
    test_proba = classifier.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    # Map predictions to class names
    class_map = {0: 'non-TDE', 1: 'TDE'}
    predictions = pd.Series(test_pred, index=X_test.index).map(class_map)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'object_id': X_test.index,
        'target': predictions.values
    })

    # Save submission
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to: {output_file}")

    # Summary statistics
    n_tde = (test_pred == 1).sum()
    print(f"\nPrediction summary:")
    print(f"  Total objects: {len(test_pred)}")
    print(f"  Predicted TDE: {n_tde} ({100 * n_tde / len(test_pred):.1f}%)")
    print(f"  Predicted non-TDE: {len(test_pred) - n_tde}")

    return submission


def main():
    parser = argparse.ArgumentParser(description='Generate MALLORN Submission')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest', 'lightgbm', 'evolved'],
                        help='Model type to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output filename')

    args = parser.parse_args()

    use_evolved = args.model == 'evolved'
    model_type = 'xgboost' if use_evolved else args.model

    generate_submission(
        model_type=model_type,
        threshold=args.threshold,
        use_evolved=use_evolved,
        data_dir=args.data_dir,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
