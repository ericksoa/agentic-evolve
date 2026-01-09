#!/usr/bin/env python3
"""
Train the meta-learning detector from collected training data.

Usage:
    python scripts/train_meta_detector.py [--input PATH] [--output PATH]
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_ensemble.meta_learning.training import (
    load_training_data,
    train_and_evaluate,
)
from adaptive_ensemble.meta_learning.detector import MetaLearningDetector


def main():
    parser = argparse.ArgumentParser(description="Train meta-learning detector")
    parser.add_argument('--input', type=str, default='meta_training_data.pkl',
                        help="Input training data file")
    parser.add_argument('--output', type=str, default=None,
                        help="Output model path (default: pretrained location)")
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Fraction of data to hold out for testing")
    parser.add_argument('--skip-eval', action='store_true',
                        help="Skip held-out evaluation, use all data for training")
    args = parser.parse_args()

    print("=" * 60)
    print("Meta-Learning Detector Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading training data from {args.input}...")
    try:
        training_data = load_training_data(args.input)
    except FileNotFoundError:
        print(f"Error: Training data not found at {args.input}")
        print("Run collect_training_data.py first to generate training data.")
        return 1

    print(f"Loaded {len(training_data['y'])} datasets")
    print(f"Features: {len(training_data['feature_names'])}")
    print(f"Positive rate: {training_data['y'].mean():.1%}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = MetaLearningDetector.DEFAULT_MODEL_PATH

    if args.skip_eval:
        # Train on all data
        print("\n" + "=" * 60)
        print("Training on all data (no held-out evaluation)...")
        print("=" * 60)

        detector = MetaLearningDetector()
        detector.fit(training_data, verbose=True)

        # Save
        detector.save(output_path)
        print(f"\nModel saved to: {output_path}")

    else:
        # Train with held-out evaluation
        print("\n" + "=" * 60)
        print("Training with held-out evaluation...")
        print("=" * 60)

        results = train_and_evaluate(
            training_data,
            test_size=args.test_size,
            verbose=True,
        )

        # Check performance
        precision = results['precision']
        recall = results['recall']

        print("\n" + "=" * 60)
        print("Performance Check")
        print("=" * 60)

        success = True
        if precision < 0.7:
            print(f"WARNING: Precision {precision:.1%} < 70% target")
            success = False
        else:
            print(f"OK: Precision {precision:.1%} >= 70% target")

        if recall < 0.6:
            print(f"WARNING: Recall {recall:.1%} < 60% target")
            success = False
        else:
            print(f"OK: Recall {recall:.1%} >= 60% target")

        # Retrain on full data and save
        print("\n" + "=" * 60)
        print("Retraining on full data for final model...")
        print("=" * 60)

        detector = MetaLearningDetector()
        detector.fit(training_data, verbose=True)

        # Save
        detector.save(output_path)
        print(f"\nModel saved to: {output_path}")

        if success:
            print("\n SUCCESS: Meta-detector meets performance targets!")
        else:
            print("\n WARNING: Meta-detector below some targets. Consider collecting more data.")

    # Show top features
    if detector.feature_importances_:
        print("\n" + "=" * 60)
        print("Top 10 Important Features")
        print("=" * 60)
        top_features = sorted(
            detector.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for name, imp in top_features:
            print(f"  {name:30s}: {imp:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
