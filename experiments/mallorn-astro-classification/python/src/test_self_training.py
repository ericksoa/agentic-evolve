#!/usr/bin/env python3
"""
Test self-training on MALLORN data.

Evaluates the semi-supervised approach using test data pseudo-labels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from features import extract_features_batch
from data_loader import load_single_split
from gen12_self_training import Gen12_SelfTraining

# Holdout splits for validation
CV_SPLITS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
HOLDOUT_SPLITS = [14, 17, 20]


def evaluate_on_split(split_num: int, use_self_training: bool = True) -> dict:
    """Evaluate classifier on a single split."""
    # Load data
    train_lc, test_lc, train_meta, test_meta = load_single_split("../../data", split_num)

    if train_meta is None:
        return {"error": "No training data"}

    # Extract features
    X_train = extract_features_batch(
        train_lc,
        metadata=train_meta.set_index('object_id'),
        use_evolved=True,
        verbose=False
    )

    # Get labels
    y_train = train_meta.set_index('object_id')['target']

    # Align
    common_ids = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_ids]
    y_train = y_train.loc[common_ids]

    # CV evaluation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        clf = Gen12_SelfTraining(
            threshold=0.43,
            C=0.05,
            confidence_pos=0.90,
            confidence_neg=0.10,
            data_dir="../../data",
            use_test_data=use_self_training
        )

        try:
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            fold_scores.append(f1)
        except Exception as e:
            print(f"  Error: {e}")
            fold_scores.append(0.0)

    return {
        "f1_mean": np.mean(fold_scores),
        "f1_std": np.std(fold_scores),
        "n_pos": (y_train == 1).sum()
    }


def main():
    print("=" * 60)
    print("Self-Training Evaluation")
    print("=" * 60)

    # Evaluate with self-training
    print("\n=== WITH Self-Training ===")
    holdout_scores_st = []
    for split_num in HOLDOUT_SPLITS:
        result = evaluate_on_split(split_num, use_self_training=True)
        print(f"Split {split_num:02d}: F1={result['f1_mean']:.4f}")
        holdout_scores_st.append(result['f1_mean'])

    # Evaluate without self-training (baseline)
    print("\n=== WITHOUT Self-Training (Gen11 baseline) ===")
    holdout_scores_base = []
    for split_num in HOLDOUT_SPLITS:
        result = evaluate_on_split(split_num, use_self_training=False)
        print(f"Split {split_num:02d}: F1={result['f1_mean']:.4f}")
        holdout_scores_base.append(result['f1_mean'])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Self-Training Holdout F1: {np.mean(holdout_scores_st):.4f}")
    print(f"Baseline Holdout F1:      {np.mean(holdout_scores_base):.4f}")
    improvement = np.mean(holdout_scores_st) - np.mean(holdout_scores_base)
    print(f"Improvement:              {improvement:+.4f}")


if __name__ == "__main__":
    main()
