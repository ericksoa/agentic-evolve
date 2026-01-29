#!/usr/bin/env python3
"""
Evaluate evolved mutation classifiers for TDE classification.
"""

import sys
import json
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'python/src')

from benchmark import fitness, load_data, prepare_labels, extract_features_batch
from sklearn.model_selection import train_test_split


def load_mutation_module(mutation_path):
    """Load a mutation module dynamically."""
    spec = importlib.util.spec_from_file_location("mutation", mutation_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_mutation(mutation_path, X_train, y_train, X_val, y_val, verbose=True):
    """Evaluate a single mutation."""
    try:
        # Load the mutation module
        module = load_mutation_module(mutation_path)

        # Get the TDEClassifier from the module
        if hasattr(module, 'TDEClassifier'):
            classifier_class = module.TDEClassifier
        elif hasattr(module, 'Gen2a_Candidate'):
            classifier_class = module.Gen2a_Candidate
        elif hasattr(module, 'Gen2X_Candidate'):
            classifier_class = module.Gen2X_Candidate
        elif hasattr(module, 'Gen1a_Candidate'):
            classifier_class = module.Gen1a_Candidate
        elif hasattr(module, 'Gen1_Candidate'):
            classifier_class = module.Gen1_Candidate
        else:
            available = [name for name in dir(module) if not name.startswith('_')]
            raise ValueError(f"No TDEClassifier found in {mutation_path}. Available: {available}")

        # Create classifier instance
        classifier = classifier_class()

        # Evaluate using the fitness function
        score = fitness(
            classifier=classifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose
        )

        return {
            "valid": True,
            "fitness": score,
            "error": None,
            "notes": f"Successfully evaluated {classifier.__class__.__name__}"
        }

    except Exception as e:
        if verbose:
            print(f"Error evaluating {mutation_path}: {e}")
        return {
            "valid": False,
            "fitness": 0.0,
            "error": str(e),
            "notes": f"Failed to evaluate: {e}"
        }


def main():
    """Main evaluation function."""

    print("Loading data...")

    # Load MALLORN data
    try:
        train_lc, test_lc, train_meta = load_data('data', split_num=1)

        if train_meta is None:
            print("Error: No metadata found")
            return

        print(f"Loaded {len(train_lc)} training observations, {train_meta['object_id'].nunique()} objects")

        # Prepare labels
        y = prepare_labels(train_meta)
        print(f"Class distribution: {(y==1).sum()} TDE, {(y==0).sum()} non-TDE")

        # Extract features
        print("Extracting features...")
        X = extract_features_batch(
            train_lc,
            metadata=train_meta.set_index('object_id'),
            use_evolved=False,
            verbose=False
        )

        # Align X and y
        common_ids = X.index.intersection(y.index)
        X = X.loc[common_ids]
        y = y.loc[common_ids]

        print(f"Features: {X.shape[1]}, Objects: {X.shape[0]}")

        # Train/validation split (ensure we have both classes in both splits)
        # With very few TDE samples (12), need careful splitting
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y))

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train: {len(y_train)} samples ({(y_train==1).sum()} TDE), Val: {len(y_val)} samples ({(y_val==1).sum()} TDE)")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data is available in ../data directory")
        return

    # Mutations to evaluate
    mutations = [
        ".evolve-sdk/improve_tde_classification_f1/mutations/gen2a.py",
        ".evolve-sdk/improve_tde_classification_f1/mutations/gen2x.py"
    ]

    results = []

    print("\n" + "="*60)
    print("Evaluating mutations...")
    print("="*60)

    for mutation_path in mutations:
        print(f"\nEvaluating {mutation_path}...")

        result = evaluate_mutation(
            mutation_path, X_train, y_train, X_val, y_val, verbose=True
        )

        result["file"] = mutation_path
        results.append(result)

        print(f"Result: Valid={result['valid']}, Fitness={result['fitness']:.4f}")
        if result['error']:
            print(f"Error: {result['error']}")

    # Find best result
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        best = max(valid_results, key=lambda x: x['fitness'])
        ranking = sorted([r['file'] for r in valid_results],
                        key=lambda f: next(r['fitness'] for r in results if r['file'] == f),
                        reverse=True)
    else:
        best = {"file": "None", "fitness": 0.0}
        ranking = []

    # Prepare final output
    output = {
        "evaluations": results,
        "best": best,
        "ranking": ranking
    }

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(output, indent=2))

    return output


if __name__ == '__main__':
    main()