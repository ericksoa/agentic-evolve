#!/usr/bin/env python3
"""
Deep dive analysis: Why do credit-g and mozilla4 have huge threshold optimization gains?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import openml
import warnings
warnings.filterwarnings('ignore')

# Datasets to analyze
DATASETS = {
    # Big winners
    31: "credit-g",      # +18.5%
    1046: "mozilla4",    # +9.0%
    # Neutral/losers for comparison
    37: "diabetes",      # +0.1%
    1489: "phoneme",     # -0.5%
    1068: "pc1",         # -2.7%
    44: "spambase",      # -0.1%
}


def load_dataset(dataset_id):
    """Load dataset from OpenML."""
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
    return X, y, dataset.name


def analyze_probability_distribution(X, y, name):
    """Analyze how probabilities are distributed."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)

    # Get cross-validated probabilities
    all_probs = []
    all_true = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        model_clone.fit(X_train, y_train)
        probs = model_clone.predict_proba(X_test)[:, 1]

        all_probs.extend(probs)
        all_true.extend(y_test)

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    return all_probs, all_true


def compute_threshold_curve(probs, true_labels):
    """Compute F1/precision/recall at different thresholds."""
    thresholds = np.linspace(0.1, 0.9, 50)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})

    return pd.DataFrame(results)


def analyze_dataset(dataset_id, name):
    """Full analysis of a single dataset."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name} (ID: {dataset_id})")
    print('='*70)

    X, y, _ = load_dataset(dataset_id)

    # Basic stats
    n_samples, n_features = X.shape
    class_counts = np.bincount(y)
    imbalance = class_counts.max() / class_counts.min()
    minority_pct = class_counts.min() / len(y) * 100

    print(f"\nBasic Stats:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: {class_counts[0]} vs {class_counts[1]}")
    print(f"  Imbalance ratio: {imbalance:.2f}x")
    print(f"  Minority class: {minority_pct:.1f}%")

    # Get probability distribution
    probs, true_labels = analyze_probability_distribution(X, y, name)

    # Probability stats
    print(f"\nProbability Distribution:")
    print(f"  Mean prob: {probs.mean():.3f}")
    print(f"  Std prob: {probs.std():.3f}")
    print(f"  Prob for class 0: {probs[true_labels==0].mean():.3f} +/- {probs[true_labels==0].std():.3f}")
    print(f"  Prob for class 1: {probs[true_labels==1].mean():.3f} +/- {probs[true_labels==1].std():.3f}")

    # Class separation
    class0_probs = probs[true_labels == 0]
    class1_probs = probs[true_labels == 1]
    overlap = np.sum((class0_probs > 0.3) & (class0_probs < 0.7)) + np.sum((class1_probs > 0.3) & (class1_probs < 0.7))
    overlap_pct = overlap / len(probs) * 100

    print(f"  Overlap zone (0.3-0.7): {overlap_pct:.1f}% of samples")

    # Threshold sensitivity
    curve = compute_threshold_curve(probs, true_labels)
    best_idx = curve['f1'].idxmax()
    best_t = curve.loc[best_idx, 'threshold']
    best_f1 = curve.loc[best_idx, 'f1']
    f1_at_05 = curve[curve['threshold'].round(2) == 0.5]['f1'].values[0] if len(curve[curve['threshold'].round(2) == 0.5]) > 0 else curve[curve['threshold'] >= 0.49].iloc[0]['f1']

    print(f"\nThreshold Sensitivity:")
    print(f"  F1 at 0.50: {f1_at_05:.3f}")
    print(f"  Best F1: {best_f1:.3f} at threshold {best_t:.2f}")
    print(f"  Gain from optimization: {(best_f1 - f1_at_05) / f1_at_05 * 100:+.1f}%")

    # F1 curve shape
    f1_range = curve['f1'].max() - curve['f1'].min()
    f1_std = curve['f1'].std()

    print(f"  F1 range: {f1_range:.3f}")
    print(f"  F1 std across thresholds: {f1_std:.3f}")

    # Where is the F1 peak?
    peak_position = "low" if best_t < 0.4 else "mid" if best_t < 0.6 else "high"
    print(f"  Peak position: {peak_position} (t={best_t:.2f})")

    # Calibration analysis
    prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=10, strategy='uniform')
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    print(f"\nCalibration:")
    print(f"  Mean calibration error: {calibration_error:.3f}")

    # Is model overconfident or underconfident?
    high_conf_correct = np.mean(true_labels[probs > 0.7] == 1) if np.sum(probs > 0.7) > 0 else 0
    low_conf_correct = np.mean(true_labels[probs < 0.3] == 0) if np.sum(probs < 0.3) > 0 else 0

    print(f"  High confidence (>0.7) accuracy: {high_conf_correct:.1%}")
    print(f"  Low confidence (<0.3) accuracy: {low_conf_correct:.1%}")

    return {
        'name': name,
        'n_samples': n_samples,
        'n_features': n_features,
        'imbalance': imbalance,
        'minority_pct': minority_pct,
        'prob_mean': probs.mean(),
        'prob_std': probs.std(),
        'class_separation': class1_probs.mean() - class0_probs.mean(),
        'overlap_pct': overlap_pct,
        'f1_at_05': f1_at_05,
        'best_f1': best_f1,
        'best_threshold': best_t,
        'gain_pct': (best_f1 - f1_at_05) / f1_at_05 * 100,
        'f1_range': f1_range,
        'calibration_error': calibration_error,
        'probs': probs,
        'true_labels': true_labels,
        'curve': curve,
    }


def main():
    results = []

    for dataset_id, name in DATASETS.items():
        try:
            result = analyze_dataset(dataset_id, name)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: What distinguishes winners from losers?")
    print("="*70)

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['probs', 'true_labels', 'curve']} for r in results])
    df = df.sort_values('gain_pct', ascending=False)

    print("\nRanked by threshold optimization gain:")
    print(df[['name', 'gain_pct', 'best_threshold', 'imbalance', 'class_separation', 'overlap_pct', 'calibration_error']].to_string(index=False))

    # Correlation analysis
    print("\nCorrelation with gain:")
    numeric_cols = ['imbalance', 'minority_pct', 'prob_std', 'class_separation', 'overlap_pct', 'f1_range', 'calibration_error', 'n_samples', 'n_features']
    for col in numeric_cols:
        corr = df['gain_pct'].corr(df[col])
        print(f"  {col}: {corr:+.3f}")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    winners = df[df['gain_pct'] > 5]
    losers = df[df['gain_pct'] < 1]

    if len(winners) > 0 and len(losers) > 0:
        print(f"\nWinners (>{5}% gain) vs Losers (<1% gain):")
        print(f"  Avg class separation: {winners['class_separation'].mean():.3f} vs {losers['class_separation'].mean():.3f}")
        print(f"  Avg overlap zone: {winners['overlap_pct'].mean():.1f}% vs {losers['overlap_pct'].mean():.1f}%")
        print(f"  Avg calibration error: {winners['calibration_error'].mean():.3f} vs {losers['calibration_error'].mean():.3f}")
        print(f"  Avg best threshold: {winners['best_threshold'].mean():.2f} vs {losers['best_threshold'].mean():.2f}")

    return results


if __name__ == '__main__':
    results = main()
