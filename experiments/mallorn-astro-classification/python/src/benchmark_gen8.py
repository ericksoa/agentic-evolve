#!/usr/bin/env python3
"""
Gen 8 Benchmark: TabPFN with Holdout Validation

Following /evolve-ml skill protocol:
- CV splits: 1-17 (for selection)
- Holdout splits: 18-20 (NEVER trained on, for overfitting detection)
- Acceptance criteria: CV-holdout gap < 0.10

Note: Each split has train data with labels - we do 5-fold CV within each split's training data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_batch
from data_loader import load_single_split
from classifier import (
    TabPFNTDEClassifier,
    Gen8HybridClassifier,
    EvolvedTDEClassifier  # Gen 4/7 baseline for comparison
)

# Per evolve-ml skill: reserve 10-15% of splits as holdout
# Note: Splits 18-19 have 0 TDEs, so use 14, 17, 20 which have good TDE counts
CV_SPLITS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]  # 15 splits for CV
HOLDOUT_SPLITS = [14, 17, 20]       # Splits with TDEs: 15, 14, 9 TDEs respectively

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def evaluate_on_split(classifier_class, split_num: int, threshold: float = 0.35, n_folds: int = 5) -> dict:
    """
    Evaluate a classifier on a single split using internal CV.

    Since test data has no labels, we do 5-fold CV within the training data.
    """
    # Load data
    train_lc, _, train_meta, _ = load_single_split(str(DATA_DIR), split_num)

    # Extract features for training data
    X = extract_features_batch(train_lc, train_meta.set_index('object_id'),
                               use_evolved=True, verbose=False)

    # Prepare labels
    y = train_meta.set_index('object_id')['target']

    # Align indices
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    # Stratified 5-fold CV within this split
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_f1s = []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        try:
            # Fit and predict
            clf = classifier_class(threshold=threshold)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            f1 = f1_score(y_val, y_pred, zero_division=0)
            fold_f1s.append(f1)

            fold_details.append({
                'fold': fold_idx,
                'f1': f1,
                'n_true_pos': ((y_val == 1) & (y_pred == 1)).sum(),
                'n_false_pos': ((y_val == 0) & (y_pred == 1)).sum(),
                'n_false_neg': ((y_val == 1) & (y_pred == 0)).sum(),
                'n_tde_val': (y_val == 1).sum()
            })
        except Exception as e:
            print(f"    Fold {fold_idx} error: {e}")
            fold_f1s.append(0.0)

    return {
        'split': split_num,
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s),
        'n_tde_total': (y == 1).sum(),
        'n_samples': len(y),
        'fold_details': fold_details
    }


def run_gen8_benchmark(classifier_class, classifier_name: str, quick: bool = False) -> dict:
    """
    Run Gen 8 benchmark with holdout validation.

    Returns dict with:
    - cv_mean: Mean F1 on CV splits (selection metric)
    - cv_std: Std of F1 on CV splits
    - holdout_mean: Mean F1 on holdout splits (overfitting check)
    - holdout_std: Std of F1 on holdout splits
    - gap: CV - holdout (should be < 0.10)
    - accepted: Whether candidate passes overfitting check
    """
    print(f"\n{'='*60}")
    print(f"Gen 8 Benchmark: {classifier_name}")
    print(f"{'='*60}")

    cv_splits = CV_SPLITS[:5] if quick else CV_SPLITS
    holdout_splits = HOLDOUT_SPLITS[:2] if quick else HOLDOUT_SPLITS

    # Evaluate on CV splits
    print(f"\nCV Splits ({len(cv_splits)} splits):")
    cv_results = []
    for split in cv_splits:
        try:
            result = evaluate_on_split(classifier_class, split)
            cv_results.append(result)
            print(f"  Split {split:2d}: F1={result['f1_mean']:.4f} +/- {result['f1_std']:.4f} "
                  f"(TDEs={result['n_tde_total']}, N={result['n_samples']})")
        except Exception as e:
            print(f"  Split {split:2d}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    cv_f1s = [r['f1_mean'] for r in cv_results]
    cv_mean = np.mean(cv_f1s) if cv_f1s else 0.0
    cv_std = np.std(cv_f1s) if cv_f1s else 0.0

    print(f"\n  CV Mean F1: {cv_mean:.4f} +/- {cv_std:.4f}")

    # Evaluate on holdout splits
    print(f"\nHoldout Splits ({len(holdout_splits)} splits):")
    holdout_results = []
    for split in holdout_splits:
        try:
            result = evaluate_on_split(classifier_class, split)
            holdout_results.append(result)
            print(f"  Split {split:2d}: F1={result['f1_mean']:.4f} +/- {result['f1_std']:.4f} "
                  f"(TDEs={result['n_tde_total']}, N={result['n_samples']})")
        except Exception as e:
            print(f"  Split {split:2d}: ERROR - {e}")

    holdout_f1s = [r['f1_mean'] for r in holdout_results]
    holdout_mean = np.mean(holdout_f1s) if holdout_f1s else 0.0
    holdout_std = np.std(holdout_f1s) if holdout_f1s else 0.0

    print(f"\n  Holdout Mean F1: {holdout_mean:.4f} +/- {holdout_std:.4f}")

    # Overfitting check
    gap = cv_mean - holdout_mean
    accepted = gap < 0.10

    print(f"\n{'='*60}")
    print(f"OVERFITTING CHECK")
    print(f"{'='*60}")
    print(f"  CV Mean:      {cv_mean:.4f}")
    print(f"  Holdout Mean: {holdout_mean:.4f}")
    print(f"  Gap:          {gap:.4f}")
    print(f"  Threshold:    0.10")

    if accepted:
        print(f"\n  [PASS] Gap < 0.10 - Candidate ACCEPTED")
    else:
        print(f"\n  [FAIL] Gap >= 0.10 - OVERFITTING DETECTED!")
        print(f"         Candidate REJECTED despite CV improvement")

    return {
        'classifier': classifier_name,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'holdout_mean': holdout_mean,
        'holdout_std': holdout_std,
        'gap': gap,
        'accepted': accepted,
        'cv_results': cv_results,
        'holdout_results': holdout_results
    }


def main():
    """Run Gen 8 benchmark comparing all candidates."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer splits')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("MALLORN Gen 8: TabPFN Foundation Model")
    print("="*60)
    print("\nFollowing /evolve-ml skill protocol:")
    print(f"  - CV splits: {CV_SPLITS[0]}-{CV_SPLITS[-1]} ({len(CV_SPLITS)} splits)")
    print(f"  - Holdout splits: {HOLDOUT_SPLITS} ({len(HOLDOUT_SPLITS)} splits)")
    print(f"  - Acceptance: CV-Holdout gap < 0.10")

    if args.quick:
        print("\n*** QUICK MODE: Testing on subset of splits ***")

    results = []

    # Test Gen 4/7 baseline (EvolvedTDEClassifier)
    print("\n" + "="*60)
    print("BASELINE: Gen 4/7 (LR + XGBoost)")
    print("="*60)
    try:
        baseline_result = run_gen8_benchmark(EvolvedTDEClassifier, "Gen4/7_LR_XGB", quick=args.quick)
        results.append(baseline_result)
    except Exception as e:
        print(f"Baseline failed: {e}")
        import traceback
        traceback.print_exc()

    # Test TabPFN solo
    print("\n" + "="*60)
    print("CANDIDATE 1: TabPFN Solo")
    print("="*60)
    try:
        tabpfn_result = run_gen8_benchmark(TabPFNTDEClassifier, "Gen8_TabPFN", quick=args.quick)
        results.append(tabpfn_result)
    except Exception as e:
        print(f"TabPFN failed: {e}")
        import traceback
        traceback.print_exc()

    # Test TabPFN + LR hybrid
    print("\n" + "="*60)
    print("CANDIDATE 2: TabPFN + LR Hybrid")
    print("="*60)
    try:
        hybrid_result = run_gen8_benchmark(Gen8HybridClassifier, "Gen8_Hybrid", quick=args.quick)
        results.append(hybrid_result)
    except Exception as e:
        print(f"Hybrid failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Gen 8 Results")
    print("="*60)
    print(f"\n{'Classifier':<20} {'CV F1':>10} {'Holdout F1':>12} {'Gap':>8} {'Status':>10}")
    print("-"*60)

    for r in results:
        status = "ACCEPTED" if r['accepted'] else "REJECTED"
        print(f"{r['classifier']:<20} {r['cv_mean']:>10.4f} {r['holdout_mean']:>12.4f} "
              f"{r['gap']:>8.4f} {status:>10}")

    # Determine winner
    accepted_results = [r for r in results if r['accepted']]
    if accepted_results:
        best = max(accepted_results, key=lambda x: x['holdout_mean'])
        print(f"\n*** WINNER: {best['classifier']} ***")
        print(f"    Holdout F1: {best['holdout_mean']:.4f} (generalizes best)")
        return best
    else:
        print("\n*** NO CANDIDATES ACCEPTED - All overfit ***")
        print("    Reverting to Gen 4 baseline")
        return None


if __name__ == '__main__':
    main()
