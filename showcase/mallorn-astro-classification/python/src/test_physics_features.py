#!/usr/bin/env python3
"""
Test Gen12 physics-based features on MALLORN data.

Extracts new physics features and evaluates with Gen11 classifier structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from features import extract_features_batch, extract_evolved_features
from gen12_physics import extract_physics_features, Gen12_Physics
from data_loader import load_single_split

# Holdout splits for validation
CV_SPLITS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
HOLDOUT_SPLITS = [14, 17, 20]


def extract_combined_features(light_curves: pd.DataFrame,
                              metadata: pd.DataFrame = None,
                              verbose: bool = True) -> pd.DataFrame:
    """Extract both evolved features and new physics features."""
    object_ids = light_curves['object_id'].unique()

    all_features = []
    iterator = tqdm(object_ids, desc="Extracting features") if verbose else object_ids

    for obj_id in iterator:
        obj_lc = light_curves[light_curves['object_id'] == obj_id].copy()

        # Get evolved features (Gen2)
        features = extract_evolved_features(obj_lc)

        # Add new physics features
        physics = extract_physics_features(obj_lc)
        features.update(physics)

        # Add metadata
        if metadata is not None and obj_id in metadata.index:
            obj_meta = metadata.loc[obj_id]
            if 'photo_z' in obj_meta:
                features['photo_z'] = obj_meta['photo_z']
            if 'photo_z_err' in obj_meta:
                features['photo_z_err'] = obj_meta['photo_z_err']

        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features).set_index('object_id')


def evaluate_on_split(split_num: int, use_physics: bool = True) -> dict:
    """Evaluate classifier on a single split."""
    # Load data
    train_lc, test_lc, train_meta, test_meta = load_single_split("../../data", split_num)

    if train_meta is None:
        return {"error": "No training data"}

    # Extract features
    if use_physics:
        X_train = extract_combined_features(
            train_lc,
            metadata=train_meta.set_index('object_id'),
            verbose=False
        )
    else:
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

        clf = Gen12_Physics(threshold=0.43, C=0.05)

        try:
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            fold_scores.append(f1)
        except Exception as e:
            print(f"  Error: {e}")
            fold_scores.append(0.0)

    n_features = X_train.shape[1]
    physics_features = [c for c in X_train.columns if any(
        p in c for p in ['alpha_', 'temp_evolution', 'baseline_', 'consistency', 'signature', 'spectral']
    )]

    return {
        "f1_mean": np.mean(fold_scores),
        "f1_std": np.std(fold_scores),
        "n_features": n_features,
        "n_physics": len(physics_features),
        "n_pos": (y_train == 1).sum()
    }


def main():
    print("=" * 60)
    print("Gen12 Physics Features Evaluation")
    print("=" * 60)

    # Evaluate with physics features
    print("\n=== WITH Physics Features ===")
    holdout_scores_physics = []
    for split_num in HOLDOUT_SPLITS:
        result = evaluate_on_split(split_num, use_physics=True)
        print(f"Split {split_num:02d}: F1={result['f1_mean']:.4f} "
              f"({result['n_features']} features, {result['n_physics']} physics)")
        holdout_scores_physics.append(result['f1_mean'])

    # Evaluate without physics features (Gen11 baseline)
    print("\n=== WITHOUT Physics Features (Gen11 baseline) ===")
    holdout_scores_base = []
    for split_num in HOLDOUT_SPLITS:
        result = evaluate_on_split(split_num, use_physics=False)
        print(f"Split {split_num:02d}: F1={result['f1_mean']:.4f} "
              f"({result['n_features']} features)")
        holdout_scores_base.append(result['f1_mean'])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"With Physics Holdout F1:    {np.mean(holdout_scores_physics):.4f}")
    print(f"Baseline Holdout F1:        {np.mean(holdout_scores_base):.4f}")
    improvement = np.mean(holdout_scores_physics) - np.mean(holdout_scores_base)
    print(f"Improvement:                {improvement:+.4f}")


if __name__ == "__main__":
    main()
