#!/usr/bin/env python3
"""
Extended benchmark to test on 15+ datasets.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Import our classifiers
import sys
sys.path.insert(0, '.')
from adaptive_ensemble import ThresholdOptimizedClassifier, AdaptiveEnsembleClassifier

# Extended dataset list (15+ binary classification datasets from OpenML)
EXTENDED_DATASETS = {
    # Original 13
    37: "diabetes",
    1464: "blood-transfusion",
    1480: "ilpd",
    15: "breast-w",
    1063: "kc1",
    1494: "qsar-biodeg",
    1067: "kc2",
    1068: "pc1",
    1461: "bank-marketing",
    1489: "phoneme",
    31: "credit-g",
    1462: "banknote-auth",
    1046: "mozilla4",
    # Additional datasets to reach 15+
    1510: "wdbc",           # 569 samples, breast cancer
    1487: "ozone-level",    # 2536 samples, weather
    44: "spambase",         # 4601 samples, spam detection
}

def load_dataset(dataset_id):
    """Load dataset from OpenML."""
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    return X, y, dataset.name

def compute_imbalance(y):
    """Compute class imbalance ratio."""
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.min() if counts.min() > 0 else 1.0

def evaluate_classifier(clf, X, y, n_splits=5, n_seeds=2):
    """Evaluate classifier with stratified CV over multiple seeds."""
    f1_scores = []

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + seed)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                clf_clone = clf.__class__(**clf.get_params())
                clf_clone.fit(X_train, y_train)
                y_pred = clf_clone.predict(X_test)
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                print(f"  Warning: {e}")
                f1_scores.append(0)

    return np.mean(f1_scores), np.std(f1_scores)

def run_extended_benchmark():
    """Run benchmark on extended dataset list."""
    results = []

    for dataset_id, name in EXTENDED_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_id}: {name}")
        print('='*60)

        try:
            X, y, actual_name = load_dataset(dataset_id)
            imbalance = compute_imbalance(y)
            print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, imbalance={imbalance:.1f}x")

            # LogisticRegression baseline
            lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
            scaler = StandardScaler()

            # Custom LR evaluation with scaling
            lr_scores = []
            for seed in range(2):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + seed)
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    scaler_clone = StandardScaler()
                    X_train_s = scaler_clone.fit_transform(X_train)
                    X_test_s = scaler_clone.transform(X_test)
                    lr_clone = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
                    lr_clone.fit(X_train_s, y_train)
                    y_pred = lr_clone.predict(X_test_s)
                    lr_scores.append(f1_score(y_test, y_pred, zero_division=0))
            lr_f1 = np.mean(lr_scores)
            print(f"  LogReg F1: {lr_f1:.3f}")

            # ThresholdOptimizedClassifier
            toc = ThresholdOptimizedClassifier(random_state=42)
            toc_f1, _ = evaluate_classifier(toc, X, y)
            print(f"  ThresholdOpt F1: {toc_f1:.3f}")

            # AdaptiveEnsembleClassifier
            ae = AdaptiveEnsembleClassifier(verbose=False, random_state=42)
            ae_f1, _ = evaluate_classifier(ae, X, y)
            print(f"  AdaptiveEns F1: {ae_f1:.3f}")

            # Determine winner
            scores = {'LogReg': lr_f1, 'ThreshOpt': toc_f1, 'Ensemble': ae_f1}
            winner = max(scores, key=scores.get)

            results.append({
                'dataset': name,
                'samples': X.shape[0],
                'imbalance': round(imbalance, 1),
                'LogReg': round(lr_f1, 3),
                'ThreshOpt': round(toc_f1, 3),
                'Ensemble': round(ae_f1, 3),
                'winner': winner,
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*80)
    print("EXTENDED BENCHMARK RESULTS")
    print("="*80)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Statistics
    print(f"\n--- Summary ---")
    print(f"Total datasets: {len(results)}")

    lr_wins = sum(1 for r in results if r['winner'] == 'LogReg')
    toc_wins = sum(1 for r in results if r['winner'] == 'ThreshOpt')
    ae_wins = sum(1 for r in results if r['winner'] == 'Ensemble')

    print(f"LogReg wins: {lr_wins}/{len(results)} ({100*lr_wins/len(results):.0f}%)")
    print(f"ThreshOpt wins: {toc_wins}/{len(results)} ({100*toc_wins/len(results):.0f}%)")
    print(f"Ensemble wins: {ae_wins}/{len(results)} ({100*ae_wins/len(results):.0f}%)")

    # Average improvements
    toc_improvements = [(r['ThreshOpt'] - r['LogReg'])/r['LogReg']*100 for r in results if r['LogReg'] > 0]
    ae_improvements = [(r['Ensemble'] - r['LogReg'])/r['LogReg']*100 for r in results if r['LogReg'] > 0]

    print(f"\nAvg ThreshOpt improvement: {np.mean(toc_improvements):+.2f}%")
    print(f"Avg Ensemble improvement: {np.mean(ae_improvements):+.2f}%")

    # By imbalance level
    low_imb = [r for r in results if r['imbalance'] <= 3]
    high_imb = [r for r in results if r['imbalance'] > 3]

    if low_imb:
        print(f"\nLow imbalance (<=3x): {len(low_imb)} datasets")
        toc_low = np.mean([(r['ThreshOpt'] - r['LogReg'])/r['LogReg']*100 for r in low_imb if r['LogReg'] > 0])
        print(f"  Avg ThreshOpt improvement: {toc_low:+.2f}%")

    if high_imb:
        print(f"\nHigh imbalance (>3x): {len(high_imb)} datasets")
        toc_high = np.mean([(r['ThreshOpt'] - r['LogReg'])/r['LogReg']*100 for r in high_imb if r['LogReg'] > 0])
        print(f"  Avg ThreshOpt improvement: {toc_high:+.2f}%")

    return results

if __name__ == '__main__':
    run_extended_benchmark()
