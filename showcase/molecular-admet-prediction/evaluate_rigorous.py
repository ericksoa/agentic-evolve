#!/usr/bin/env python3
"""
Rigorous evaluation script for hERG toxicity prediction models.

Addresses common ML evaluation pitfalls:
1. Multi-seed repeatability (10 seeds)
2. Calibration metrics (ECE, reliability diagram data)
3. Operational metrics (sensitivity@specificity, precision@top-k)
4. PR-AUC for imbalanced data
5. Ablation-ready output

Usage:
    python evaluate_rigorous.py evolved_ensemble.py --seeds 10
    python evaluate_rigorous.py evolved_ensemble.py --ablation
"""

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, precision_score, recall_score,
    f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')


def load_herg_data(seed=42):
    """Load hERG dataset from TDC with specified seed."""
    from tdc.single_pred import Tox
    data = Tox(name='hERG')
    split = data.get_split(method='scaffold', seed=seed)
    return {
        'train': (split['train']['Drug'].tolist(), split['train']['Y'].tolist()),
        'valid': (split['valid']['Drug'].tolist(), split['valid']['Y'].tolist()),
        'test': (split['test']['Drug'].tolist(), split['test']['Y'].tolist())
    }


def load_solution(solution_path: str):
    """Dynamically load a solution module."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'HERGPredictor'):
        return module.HERGPredictor
    raise ValueError(f"Solution must define a HERGPredictor class")


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)
    return ece / len(y_true)


def sensitivity_at_specificity(y_true, y_prob, target_spec=0.90):
    """Calculate sensitivity at a given specificity threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find threshold where specificity >= target
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    best_idx = idx[np.argmax(tpr[idx])]
    return tpr[best_idx], thresholds[best_idx]


def precision_at_top_k(y_true, y_prob, k_percent=10):
    """Calculate precision in top k% highest-risk predictions."""
    n = len(y_true)
    k = max(1, int(n * k_percent / 100))
    top_k_idx = np.argsort(y_prob)[-k:]
    return y_true[top_k_idx].mean()


def get_calibration_curve_data(y_true, y_prob, n_bins=10):
    """Get data for reliability diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}


def evaluate_single_seed(predictor_class, seed=42):
    """Run full evaluation for a single seed."""
    data = load_herg_data(seed)
    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    X_test, y_test = data['test']

    # Train on train+valid, test on test
    X_trainval = X_train + X_valid
    y_trainval = y_train + y_valid

    predictor = predictor_class()
    predictor.fit(X_trainval, y_trainval)
    y_prob = predictor.predict_proba(X_test)

    y_test = np.array(y_test)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # Core metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Calibration
    ece = calculate_ece(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    # Operational metrics
    sens_at_90spec, thresh_90spec = sensitivity_at_specificity(y_test, y_prob, 0.90)
    prec_top10 = precision_at_top_k(y_test, y_prob, 10)

    # Standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        'seed': seed,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ece': ece,
        'brier': brier,
        'sens_at_90spec': sens_at_90spec,
        'prec_top10': prec_top10,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'calibration_curve': get_calibration_curve_data(y_test, y_prob)
    }


def run_multi_seed(predictor_class, n_seeds=10):
    """Run evaluation across multiple seeds."""
    results = []
    for seed in range(42, 42 + n_seeds):
        result = evaluate_single_seed(predictor_class, seed)
        results.append(result)
        print(f"  Seed {seed}: ROC-AUC={result['roc_auc']:.4f}, PR-AUC={result['pr_auc']:.4f}")

    # Aggregate statistics
    metrics = ['roc_auc', 'pr_auc', 'ece', 'brier', 'sens_at_90spec', 'prec_top10', 'accuracy', 'f1']
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in results]
        summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return {'seeds': results, 'summary': summary}


def run_ablation(solution_path):
    """Run ablation study comparing different configurations."""
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    ablation_results = {}

    # 1. Fingerprint-only baseline
    print("\n[1/4] Fingerprint-only (baseline_fingerprint.py)...")
    try:
        fp_class = load_solution('baseline_fingerprint.py')
        fp_result = evaluate_single_seed(fp_class, seed=42)
        ablation_results['fingerprint_only'] = {
            'roc_auc': fp_result['roc_auc'],
            'pr_auc': fp_result['pr_auc']
        }
        print(f"      ROC-AUC: {fp_result['roc_auc']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
        ablation_results['fingerprint_only'] = {'error': str(e)}

    # 2. Descriptor-only baseline
    print("\n[2/4] Descriptor-only (baseline_descriptors.py)...")
    try:
        desc_class = load_solution('baseline_descriptors.py')
        desc_result = evaluate_single_seed(desc_class, seed=42)
        ablation_results['descriptor_only'] = {
            'roc_auc': desc_result['roc_auc'],
            'pr_auc': desc_result['pr_auc']
        }
        print(f"      ROC-AUC: {desc_result['roc_auc']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
        ablation_results['descriptor_only'] = {'error': str(e)}

    # 3. Evolved ensemble
    print(f"\n[3/4] Evolved ensemble ({solution_path})...")
    try:
        evolved_class = load_solution(solution_path)
        evolved_result = evaluate_single_seed(evolved_class, seed=42)
        ablation_results['evolved_ensemble'] = {
            'roc_auc': evolved_result['roc_auc'],
            'pr_auc': evolved_result['pr_auc']
        }
        print(f"      ROC-AUC: {evolved_result['roc_auc']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
        ablation_results['evolved_ensemble'] = {'error': str(e)}

    # 4. Single model from ensemble (RF only) for comparison
    print("\n[4/4] Single RF (no ensemble)...")
    try:
        # Create a simple RF-only version
        from sklearn.ensemble import RandomForestClassifier
        from rdkit import Chem
        from rdkit.Chem import AllChem

        class SingleRFPredictor:
            def __init__(self):
                self.model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced', n_jobs=1)

            def fit(self, X_smiles, y):
                X = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)) if Chem.MolFromSmiles(s) else np.zeros(2048) for s in X_smiles])
                self.model.fit(X, y)
                return self

            def predict_proba(self, X_smiles):
                X = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)) if Chem.MolFromSmiles(s) else np.zeros(2048) for s in X_smiles])
                return self.model.predict_proba(X)[:, 1]

        rf_result = evaluate_single_seed(SingleRFPredictor, seed=42)
        ablation_results['single_rf'] = {
            'roc_auc': rf_result['roc_auc'],
            'pr_auc': rf_result['pr_auc']
        }
        print(f"      ROC-AUC: {rf_result['roc_auc']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
        ablation_results['single_rf'] = {'error': str(e)}

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description='Rigorous hERG model evaluation')
    parser.add_argument('solution', type=str, help='Path to solution Python file')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds for repeatability')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    args = parser.parse_args()

    print("="*60)
    print("RIGOROUS hERG PREDICTION EVALUATION")
    print("="*60)

    predictor_class = load_solution(args.solution)

    # Multi-seed evaluation
    print(f"\n[Multi-seed evaluation: {args.seeds} seeds]")
    results = run_multi_seed(predictor_class, args.seeds)

    # Ablation if requested
    if args.ablation:
        results['ablation'] = run_ablation(args.solution)

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        s = results['summary']
        print("\n" + "="*60)
        print("SUMMARY (mean ± std across seeds)")
        print("="*60)
        print(f"\nCore Metrics:")
        print(f"  ROC-AUC:          {s['roc_auc']['mean']:.4f} ± {s['roc_auc']['std']:.4f}")
        print(f"  PR-AUC:           {s['pr_auc']['mean']:.4f} ± {s['pr_auc']['std']:.4f}")

        print(f"\nCalibration:")
        print(f"  ECE:              {s['ece']['mean']:.4f} ± {s['ece']['std']:.4f}")
        print(f"  Brier Score:      {s['brier']['mean']:.4f} ± {s['brier']['std']:.4f}")

        print(f"\nOperational Metrics:")
        print(f"  Sens@90%Spec:     {s['sens_at_90spec']['mean']:.4f} ± {s['sens_at_90spec']['std']:.4f}")
        print(f"  Prec@Top10%:      {s['prec_top10']['mean']:.4f} ± {s['prec_top10']['std']:.4f}")

        print(f"\nStandard Metrics:")
        print(f"  Accuracy:         {s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f}")
        print(f"  F1:               {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}")

        if 'ablation' in results:
            print("\n" + "="*60)
            print("ABLATION RESULTS")
            print("="*60)
            for name, res in results['ablation'].items():
                if 'error' not in res:
                    print(f"  {name:20s}: ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
