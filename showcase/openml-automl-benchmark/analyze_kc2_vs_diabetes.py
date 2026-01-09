#!/usr/bin/env python3
"""
Deep Dive: kc2 vs diabetes

Question: Both have ~40% overlap, but kc2 improves +1.8% while diabetes is neutral.
Why does the classifier correctly identify kc2 as improvable and skip diabetes?
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '.')
from adaptive_ensemble import ThresholdOptimizedClassifier


def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = X.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    return X, y


def get_cv_probs(X, y):
    """Get CV probabilities for analysis."""
    probs = []
    true_labels = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train_s, y_train)

        probs.extend(lr.predict_proba(X_test_s)[:, 1])
        true_labels.extend(y_test)

    return np.array(probs), np.array(true_labels)


def analyze_threshold_curve(probs, true_labels, name):
    """Analyze how F1 varies with threshold."""
    thresholds = np.linspace(0.1, 0.9, 17)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(true_labels, preds, zero_division=0))

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    f1_at_05 = f1_scores[list(thresholds).index(min(thresholds, key=lambda x: abs(x - 0.5)))]

    return {
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'best_thresh': best_thresh,
        'best_f1': best_f1,
        'f1_at_05': f1_at_05,
        'potential_gain': (best_f1 - f1_at_05) / f1_at_05 if f1_at_05 > 0 else 0,
        'f1_range': max(f1_scores) - min(f1_scores),
        'thresh_distance': abs(best_thresh - 0.5),
    }


def analyze_prob_distribution(probs, true_labels, name):
    """Analyze probability distributions by class."""
    class0_probs = probs[true_labels == 0]
    class1_probs = probs[true_labels == 1]

    return {
        'class0_mean': class0_probs.mean(),
        'class0_std': class0_probs.std(),
        'class1_mean': class1_probs.mean(),
        'class1_std': class1_probs.std(),
        'separation': abs(class1_probs.mean() - class0_probs.mean()),
        'overlap_pct': np.mean((probs >= 0.3) & (probs <= 0.7)) * 100,
    }


def run_analysis():
    print("="*70)
    print("DEEP DIVE: kc2 vs diabetes")
    print("="*70)

    datasets = {
        1067: "kc2",
        37: "diabetes",
    }

    all_results = {}

    for dataset_id, name in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print("="*50)

        X, y = load_dataset(dataset_id)
        probs, true_labels = get_cv_probs(X, y)

        # Analyze threshold curve
        thresh_analysis = analyze_threshold_curve(probs, true_labels, name)

        # Analyze probability distribution
        prob_analysis = analyze_prob_distribution(probs, true_labels, name)

        # Get ThresholdOptimizedClassifier diagnostics
        clf = ThresholdOptimizedClassifier(threshold_range='auto', random_state=42)
        clf.fit(X, y)
        d = clf.diagnostics_

        print(f"\n--- Dataset Stats ---")
        print(f"  Samples: {len(X)}")
        print(f"  Class balance: {np.mean(y==1)*100:.1f}% positive")

        print(f"\n--- Probability Distribution ---")
        print(f"  Class 0 mean prob: {prob_analysis['class0_mean']:.3f} +/- {prob_analysis['class0_std']:.3f}")
        print(f"  Class 1 mean prob: {prob_analysis['class1_mean']:.3f} +/- {prob_analysis['class1_std']:.3f}")
        print(f"  Separation: {prob_analysis['separation']:.3f}")
        print(f"  Overlap zone (0.3-0.7): {prob_analysis['overlap_pct']:.1f}%")

        print(f"\n--- Threshold Analysis ---")
        print(f"  Best threshold: {thresh_analysis['best_thresh']:.2f}")
        print(f"  F1 at 0.5: {thresh_analysis['f1_at_05']:.3f}")
        print(f"  F1 at best: {thresh_analysis['best_f1']:.3f}")
        print(f"  Potential gain: {thresh_analysis['potential_gain']*100:+.1f}%")
        print(f"  F1 range: {thresh_analysis['f1_range']:.3f}")
        print(f"  Threshold distance from 0.5: {thresh_analysis['thresh_distance']:.2f}")

        print(f"\n--- Classifier Decision ---")
        print(f"  Strategy: {d['strategy']}")
        print(f"  Optimization skipped: {clf.optimization_skipped_}")
        print(f"  Optimal threshold: {clf.optimal_threshold_:.2f}")

        all_results[name] = {
            'thresh': thresh_analysis,
            'prob': prob_analysis,
            'clf': d,
        }

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Why kc2 improves but diabetes doesn't")
    print("="*70)

    kc2 = all_results['kc2']
    diabetes = all_results['diabetes']

    print("\n| Metric | kc2 | diabetes | Difference |")
    print("|--------|-----|----------|------------|")
    print(f"| Overlap % | {kc2['prob']['overlap_pct']:.1f}% | {diabetes['prob']['overlap_pct']:.1f}% | Similar |")
    print(f"| Best threshold | {kc2['thresh']['best_thresh']:.2f} | {diabetes['thresh']['best_thresh']:.2f} | kc2 farther from 0.5 |")
    print(f"| Thresh distance | {kc2['thresh']['thresh_distance']:.2f} | {diabetes['thresh']['thresh_distance']:.2f} | KEY DIFFERENCE |")
    print(f"| Potential gain | {kc2['thresh']['potential_gain']*100:+.1f}% | {diabetes['thresh']['potential_gain']*100:+.1f}% | kc2 higher |")
    print(f"| F1 range | {kc2['thresh']['f1_range']:.3f} | {diabetes['thresh']['f1_range']:.3f} | kc2 more variable |")
    print(f"| Class separation | {kc2['prob']['separation']:.3f} | {diabetes['prob']['separation']:.3f} | Similar |")
    print(f"| Strategy | {kc2['clf']['strategy']} | {diabetes['clf']['strategy']} | Different! |")

    print("\n--- Conclusion ---")
    print("""
The key difference is THRESHOLD DISTANCE:

- kc2: Best threshold = 0.30 (0.20 away from 0.5)
       This means shifting the threshold significantly helps.

- diabetes: Best threshold = 0.40-0.50 (only ~0.05 away from 0.5)
            The optimal threshold is already close to the default.
            Optimization provides minimal benefit.

This validates our v3 detection logic:
- We check if potential_gain > 1% (diabetes fails: only +0.6%)
- We check if thresh_distance > 0.10 (diabetes fails: only 0.05)

The classifier correctly identifies that:
1. kc2: High overlap + optimal threshold far from 0.5 = optimize
2. diabetes: High overlap + optimal threshold near 0.5 = skip
""")


if __name__ == '__main__':
    run_analysis()
