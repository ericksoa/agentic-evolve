#!/usr/bin/env python3
"""
Gen13: Proper per-split submission generator.

Trains a model on each split's training data and predicts on that split's test data.
Combines all predictions into a single submission file.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

from features import extract_features_batch
from data_loader import load_single_split


# Gen11 champion configuration
TOP_FEATURES = ['g_skew', 'r_scatter', 'r_skew', 'i_skew', 'i_kurtosis', 'r_kurtosis']
THRESHOLD = 0.43
C = 0.05


def train_predict_split(split_num: int, model_type: str = 'logreg',
                        data_dir: str = '../../data', verbose: bool = True):
    """Train on a split and predict on its test set."""
    # Load data
    train_lc, test_lc, train_meta, test_meta = load_single_split(data_dir, split_num)

    if train_meta is None or len(train_lc) == 0:
        return None

    # Extract features
    X_train = extract_features_batch(
        train_lc,
        metadata=train_meta.set_index('object_id'),
        use_evolved=True,
        verbose=False
    )

    X_test = extract_features_batch(
        test_lc,
        metadata=test_meta.set_index('object_id') if test_meta is not None else None,
        use_evolved=True,
        verbose=False
    )

    # Get labels
    y_train = train_meta.set_index('object_id')['target']

    # Align training data
    common = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common]
    y_train = y_train.loc[common]

    # Store feature names and find top feature indices
    feature_names = list(X_train.columns)
    top_indices = [i for i, col in enumerate(feature_names) if col in TOP_FEATURES]

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Polynomial features on top predictors
    if len(top_indices) >= 2:
        X_train_top = X_train_scaled[:, top_indices]
        X_test_top = X_test_scaled[:, top_indices]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_top)
        X_test_poly = poly.transform(X_test_top)

        n_orig = len(top_indices)
        X_train_final = np.hstack([X_train_scaled, X_train_poly[:, n_orig:]])
        X_test_final = np.hstack([X_test_scaled, X_test_poly[:, n_orig:]])
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train model based on type
    if model_type == 'logreg':
        model = LogisticRegression(
            class_weight='balanced',
            C=C,
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    elif model_type == 'gbt':
        # Compute sample weights for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1
        sample_weight = np.where(y_train == 1, scale_pos, 1.0)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train_final, y_train, sample_weight=sample_weight)
        proba = model.predict_proba(X_test_final)[:, 1]
        predictions = (proba >= THRESHOLD).astype(int)

        return pd.DataFrame({
            'object_id': X_test.index,
            'prediction': predictions,
            'proba': proba
        })
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train_final, y_train)

    # Predict
    proba = model.predict_proba(X_test_final)[:, 1]
    predictions = (proba >= THRESHOLD).astype(int)

    return pd.DataFrame({
        'object_id': X_test.index,
        'prediction': predictions,
        'proba': proba
    })


def generate_submission(model_type: str = 'logreg', data_dir: str = '../../data',
                        output_file: str = 'submission.csv'):
    """Generate full submission by training on all splits."""
    print(f"Generating submission with model_type={model_type}")
    print("=" * 60)

    all_predictions = []

    for split_num in tqdm(range(1, 21), desc="Processing splits"):
        result = train_predict_split(split_num, model_type, data_dir, verbose=False)
        if result is not None:
            all_predictions.append(result)

    # Combine all predictions
    combined = pd.concat(all_predictions, ignore_index=True)

    # Create submission
    submission = combined[['object_id', 'prediction']].copy()

    # Save
    submission.to_csv(output_file, index=False)

    # Summary
    n_tde = (submission['prediction'] == 1).sum()
    print(f"\nSubmission saved to: {output_file}")
    print(f"Total objects: {len(submission)}")
    print(f"Predicted TDE: {n_tde} ({100 * n_tde / len(submission):.1f}%)")

    return submission, combined


def evaluate_holdout(model_type: str = 'logreg', data_dir: str = '../../data'):
    """Evaluate on holdout splits (14, 17, 20) using per-split CV."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, make_scorer

    holdout_splits = [14, 17, 20]
    cv_splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

    print(f"Evaluating {model_type} on holdout splits")
    print("=" * 60)

    results = []

    for split_num in holdout_splits:
        # Load this split's data
        train_lc, test_lc, train_meta, test_meta = load_single_split(data_dir, split_num)

        # Extract features
        X = extract_features_batch(
            train_lc,
            metadata=train_meta.set_index('object_id'),
            use_evolved=True,
            verbose=False
        )
        y = train_meta.set_index('object_id')['target']

        # Align
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]

        # Preprocess (same as train_predict_split)
        feature_names = list(X.columns)
        top_indices = [i for i, col in enumerate(feature_names) if col in TOP_FEATURES]

        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)

        if len(top_indices) >= 2:
            X_top = X_scaled[:, top_indices]
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_top)
            n_orig = len(top_indices)
            X_final = np.hstack([X_scaled, X_poly[:, n_orig:]])
        else:
            X_final = X_scaled

        # Create model
        if model_type == 'logreg':
            model = LogisticRegression(
                class_weight='balanced', C=C, max_iter=1000, random_state=42
            )
        elif model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
                max_iter=500, random_state=42, early_stopping=True
            )
        elif model_type == 'gbt':
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            )

        # 3-fold CV
        f1_scorer = make_scorer(f1_score, zero_division=0)
        scores = cross_val_score(model, X_final, y, cv=3, scoring=f1_scorer)

        results.append({
            'split': split_num,
            'f1_mean': scores.mean(),
            'f1_std': scores.std(),
            'n_pos': y.sum()
        })
        print(f"Split {split_num:02d}: F1={scores.mean():.4f} (+/- {scores.std():.4f}), TDEs={y.sum()}")

    # Summary
    mean_f1 = np.mean([r['f1_mean'] for r in results])
    print(f"\nMean Holdout F1: {mean_f1:.4f}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['logreg', 'mlp', 'gbt'], default='logreg')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on holdout')
    parser.add_argument('--submit', action='store_true', help='Generate submission')
    parser.add_argument('--output', default='submission.csv')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_holdout(args.model)

    if args.submit:
        generate_submission(args.model, output_file=args.output)
