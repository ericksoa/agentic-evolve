"""
Training utilities for the meta-learning detector.

Provides functions to:
1. Evaluate threshold optimization benefit on a dataset
2. Collect training data from multiple datasets
3. Train and evaluate the meta-detector
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .extractor import MetaFeatureExtractor


def evaluate_threshold_benefit(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
    significance_threshold: float = 0.01,
) -> Dict:
    """
    Evaluate whether threshold optimization helps on a dataset.

    Runs CV with both default threshold (0.5) and optimized threshold,
    then compares F1 scores.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Random state for reproducibility.
    significance_threshold : float, default=0.01
        Minimum relative gain to be considered significant.

    Returns
    -------
    result : dict
        - 'f1_default': F1 with default threshold (0.5)
        - 'f1_optimized': F1 with optimized threshold
        - 'optimal_threshold': Best threshold found
        - 'gain': Relative improvement (f1_opt - f1_def) / f1_def
        - 'will_help': True if gain > significance_threshold
    """
    # Check for binary classification
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        return {
            'f1_default': 0.0,
            'f1_optimized': 0.0,
            'optimal_threshold': 0.5,
            'gain': 0.0,
            'will_help': False,
            'error': 'Not binary classification',
        }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    f1_default_scores = []
    f1_optimized_scores = []
    optimal_thresholds = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
        )
        model.fit(X_train_scaled, y_train)

        # Get probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]

        # F1 with default threshold
        preds_default = (probs >= 0.5).astype(int)
        pred_labels_default = np.where(preds_default == 1, unique_labels[1], unique_labels[0])
        f1_default = f1_score(y_test, pred_labels_default, zero_division=0)
        f1_default_scores.append(f1_default)

        # Find optimal threshold
        best_thresh = 0.5
        best_f1 = f1_default
        for thresh in np.linspace(0.1, 0.7, 25):
            preds = (probs >= thresh).astype(int)
            pred_labels = np.where(preds == 1, unique_labels[1], unique_labels[0])
            f1 = f1_score(y_test, pred_labels, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        f1_optimized_scores.append(best_f1)
        optimal_thresholds.append(best_thresh)

    # Aggregate results
    f1_default = np.mean(f1_default_scores)
    f1_optimized = np.mean(f1_optimized_scores)
    optimal_threshold = np.mean(optimal_thresholds)

    # Compute gain
    gain = (f1_optimized - f1_default) / f1_default if f1_default > 0 else 0.0

    return {
        'f1_default': f1_default,
        'f1_optimized': f1_optimized,
        'optimal_threshold': optimal_threshold,
        'gain': gain,
        'gain_pct': gain * 100,
        'will_help': gain > significance_threshold,
        'f1_default_scores': f1_default_scores,
        'f1_optimized_scores': f1_optimized_scores,
        'optimal_thresholds': optimal_thresholds,
    }


def collect_training_data(
    datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    cv: int = 5,
    significance_threshold: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Collect training data from multiple datasets.

    Parameters
    ----------
    datasets : list of (X, y, name) tuples
        List of datasets to evaluate.
    cv : int, default=5
        CV folds for evaluation.
    significance_threshold : float, default=0.01
        Minimum gain to be considered significant.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    training_data : dict
        - 'X': Feature matrix of shape (n_datasets, n_features)
        - 'y': Binary labels (1 = helped, 0 = didn't help)
        - 'feature_names': List of feature names
        - 'dataset_names': List of dataset names
        - 'details': List of detailed results per dataset
    """
    extractor = MetaFeatureExtractor()

    feature_matrix = []
    labels = []
    dataset_names = []
    details = []

    for i, (X, y, name) in enumerate(datasets):
        if verbose:
            print(f"[{i+1}/{len(datasets)}] Processing {name}...", end=' ')

        try:
            # Extract meta-features
            features = extractor.extract(X, y)
            feature_array = extractor.to_array(features)

            # Evaluate threshold benefit
            benefit = evaluate_threshold_benefit(
                X, y, cv=cv, significance_threshold=significance_threshold
            )

            feature_matrix.append(feature_array)
            labels.append(1 if benefit['will_help'] else 0)
            dataset_names.append(name)
            details.append({
                'name': name,
                'features': features,
                'benefit': benefit,
            })

            if verbose:
                status = "HELPED" if benefit['will_help'] else "no help"
                print(f"gain={benefit['gain_pct']:.1f}% ({status})")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    return {
        'X': np.array(feature_matrix),
        'y': np.array(labels),
        'feature_names': extractor.get_feature_names(),
        'dataset_names': dataset_names,
        'details': details,
    }


def save_training_data(training_data: Dict, path: str) -> None:
    """Save training data to CSV and pickle files."""
    import pickle
    import os

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Save feature matrix as CSV
    df = pd.DataFrame(
        training_data['X'],
        columns=training_data['feature_names'],
    )
    df['will_help'] = training_data['y']
    df['dataset_name'] = training_data['dataset_names']
    df.to_csv(path.replace('.pkl', '.csv'), index=False)

    # Save full data as pickle
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


def load_training_data(path: str) -> Dict:
    """Load training data from pickle file."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_and_evaluate(
    training_data: Dict,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Train meta-detector and evaluate on held-out data.

    Parameters
    ----------
    training_data : dict
        Training data from collect_training_data().
    test_size : float, default=0.2
        Fraction to hold out for testing.
    random_state : int, default=42
        Random state.
    verbose : bool, default=True
        Print evaluation results.

    Returns
    -------
    results : dict
        - 'detector': Trained MetaLearningDetector
        - 'accuracy': Test accuracy
        - 'precision': Test precision
        - 'recall': Test recall
        - 'predictions': Test predictions
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    from .detector import MetaLearningDetector

    X = training_data['X']
    y = training_data['y']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"Training on {len(X_train)} datasets, testing on {len(X_test)}")
        print(f"Train positive rate: {y_train.mean():.1%}")
        print(f"Test positive rate: {y_test.mean():.1%}")

    # Train
    detector = MetaLearningDetector()
    detector.fit({
        'X': X_train,
        'y': y_train,
        'feature_names': training_data['feature_names'],
    }, verbose=verbose)

    # Evaluate
    predictions = []
    probabilities = []
    for i in range(len(X_test)):
        prob = detector.predict_proba(X_test[i])
        probabilities.append(prob)
        predictions.append(1 if prob >= 0.5 else 0)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    if verbose:
        print(f"\n=== Evaluation Results ===")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Precision: {precision:.1%} (when we say 'optimize', how often does it help?)")
        print(f"Recall: {recall:.1%} (of datasets that benefit, how many do we find?)")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    return {
        'detector': detector,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'y_test': y_test,
    }
