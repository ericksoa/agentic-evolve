#!/usr/bin/env python3
"""
OpenML AutoML Benchmark Runner.

Runs baselines and evolve-sdk evolution on OpenML datasets.

Usage:
    # Run pilot test (5 small datasets)
    python run_benchmark.py --pilot

    # Run single dataset
    python run_benchmark.py --dataset-id 31

    # Run evolve-sdk on a dataset
    python run_benchmark.py --dataset-id 31 --evolve

    # Run full CC18 suite (72 datasets)
    python run_benchmark.py --suite cc18
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from openml_loader import (
    load_dataset, get_cv_splits, preprocess_features,
    get_pilot_dataset_ids, get_cc18_dataset_ids, DatasetInfo
)
from baseline_runner import run_all_baselines, get_best_baseline

warnings.filterwarnings('ignore')


def get_results_dir() -> Path:
    """Get results directory, creating if needed."""
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_dataset_baselines(dataset_id: int, n_folds: int = 10) -> dict:
    """
    Run all baseline models on a single dataset.

    Returns dict with dataset info and baseline results.
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset {dataset_id}...")

    try:
        X, y, info = load_dataset(dataset_id)
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return {'dataset_id': dataset_id, 'error': str(e)}

    print(f"Dataset: {info.name}")
    print(f"  Samples: {info.n_samples}")
    print(f"  Features: {info.n_features}")
    print(f"  Classes: {info.n_classes}")
    print(f"  Imbalance: {info.imbalance_ratio:.2f}")

    # Get CV splits
    splits = get_cv_splits(X, y, n_splits=n_folds)

    # Preprocess all data
    X_processed = preprocess_all(X)
    y_values = y.values

    # Run baselines
    print(f"\nRunning baselines...")
    results_df = run_all_baselines(X_processed, y_values, splits, info.n_classes)

    # Get best baseline
    best_model = results_df.iloc[0]['model']
    best_f1 = results_df.iloc[0]['f1_mean']

    print(f"\nBaseline Results:")
    print(results_df[['model', 'f1', 'gap']].to_string(index=False))
    print(f"\nBest: {best_model} with F1={best_f1:.4f}")

    return {
        'dataset_id': dataset_id,
        'dataset_name': info.name,
        'n_samples': info.n_samples,
        'n_features': info.n_features,
        'n_classes': info.n_classes,
        'imbalance_ratio': info.imbalance_ratio,
        'best_baseline': best_model,
        'best_f1': best_f1,
        'all_baselines': results_df.to_dict('records'),
        'timestamp': datetime.now().isoformat()
    }


def preprocess_all(X: pd.DataFrame) -> np.ndarray:
    """Preprocess entire dataset consistently."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    processed = []

    if len(numeric_cols) > 0:
        X_num = X[numeric_cols].values
        imputer = SimpleImputer(strategy='median')
        X_num = imputer.fit_transform(X_num)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        processed.append(X_num)

    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string first to handle categorical dtype properly
        vals = X[col].astype(str).fillna('__MISSING__')
        encoded = le.fit_transform(vals).reshape(-1, 1)
        processed.append(encoded)

    if processed:
        return np.hstack(processed)
    return np.array([]).reshape(len(X), 0)


def generate_evolve_config(dataset_id: int, info: DatasetInfo) -> Path:
    """
    Generate evolve_config.json for a specific dataset.

    Returns path to generated config file.
    """
    template_path = Path(__file__).parent.parent.parent / 'evolve_config_template.json'
    with open(template_path) as f:
        template = f.read()

    # Replace placeholders
    config_str = template.replace('{dataset_id}', str(dataset_id))
    config_str = config_str.replace('{dataset_name}', info.name)
    config_str = config_str.replace('{n_samples}', str(info.n_samples))
    config_str = config_str.replace('{n_features}', str(info.n_features))
    config_str = config_str.replace('{n_classes}', str(info.n_classes))
    config_str = config_str.replace('{imbalance_ratio}', f'{info.imbalance_ratio:.2f}')

    # Write config
    config_path = Path(__file__).parent.parent.parent / f'evolve_config_{dataset_id}.json'
    with open(config_path, 'w') as f:
        f.write(config_str)

    return config_path


def run_evolution(dataset_id: int, max_generations: int = 10) -> dict:
    """
    Run evolve-sdk evolution on a dataset.

    This generates a config and runs the evolve-sdk CLI.
    """
    print(f"\n{'='*60}")
    print(f"Running evolution on dataset {dataset_id}...")

    # Load dataset info
    X, y, info = load_dataset(dataset_id)

    # Generate config
    config_path = generate_evolve_config(dataset_id, info)
    print(f"Generated config: {config_path}")

    # Run evolve-sdk
    import subprocess

    cmd = [
        sys.executable, '-m', 'evolve_sdk',
        '--config', str(config_path),
        '--max-generations', str(max_generations),
        '--population-size', '6',
        '--plateau', '3',
        '--no-parallel'  # Sequential for resource control
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        return {
            'dataset_id': dataset_id,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'dataset_id': dataset_id,
            'success': False,
            'error': 'Timeout after 1 hour'
        }
    except Exception as e:
        return {
            'dataset_id': dataset_id,
            'success': False,
            'error': str(e)
        }


def run_pilot(evolve: bool = False):
    """Run benchmark on 5 pilot datasets."""
    dataset_ids = get_pilot_dataset_ids()
    print(f"Running pilot benchmark on {len(dataset_ids)} datasets...")

    results = []
    for did in dataset_ids:
        if evolve:
            result = run_evolution(did)
        else:
            result = run_dataset_baselines(did)
        results.append(result)

    # Save results
    results_dir = get_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'pilot_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")

    # Summary
    print(f"\nPilot Summary:")
    for r in results:
        if 'error' in r:
            print(f"  {r['dataset_id']}: ERROR - {r['error']}")
        else:
            print(f"  {r['dataset_name']}: Best={r['best_baseline']} F1={r['best_f1']:.4f}")

    return results


def run_full_suite(evolve: bool = False):
    """Run benchmark on full CC18 suite (72 datasets)."""
    dataset_ids = get_cc18_dataset_ids()
    print(f"Running full benchmark on {len(dataset_ids)} datasets...")
    print("WARNING: This will take many hours!")

    results = []
    for i, did in enumerate(dataset_ids):
        print(f"\n[{i+1}/{len(dataset_ids)}] Dataset {did}")
        if evolve:
            result = run_evolution(did)
        else:
            result = run_dataset_baselines(did)
        results.append(result)

        # Save intermediate results
        results_dir = get_results_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'cc18_results_partial_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final save
    results_file = results_dir / f'cc18_results_final_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Final results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='OpenML AutoML Benchmark Runner'
    )
    parser.add_argument('--pilot', action='store_true',
                        help='Run on 5 pilot datasets')
    parser.add_argument('--suite', choices=['cc18'],
                        help='Run on benchmark suite')
    parser.add_argument('--dataset-id', type=int,
                        help='Run on single dataset')
    parser.add_argument('--evolve', action='store_true',
                        help='Run evolve-sdk evolution instead of baselines')
    parser.add_argument('--max-generations', type=int, default=10,
                        help='Max generations for evolution')

    args = parser.parse_args()

    if args.pilot:
        run_pilot(evolve=args.evolve)
    elif args.suite == 'cc18':
        run_full_suite(evolve=args.evolve)
    elif args.dataset_id:
        if args.evolve:
            run_evolution(args.dataset_id, args.max_generations)
        else:
            result = run_dataset_baselines(args.dataset_id)
            print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
