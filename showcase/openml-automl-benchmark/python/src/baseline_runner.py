"""
Baseline model runner for OpenML benchmarks.

Runs default sklearn models to establish baseline scores before evolution.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


def get_baseline_models() -> Dict[str, Any]:
    """
    Get dictionary of baseline models with default hyperparameters.

    Returns:
        Dict mapping model name to (model_class, default_params)
    """
    models = {
        'logistic_regression': (
            LogisticRegression,
            {'max_iter': 1000, 'random_state': 42}
        ),
        'random_forest': (
            RandomForestClassifier,
            {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        ),
        'gradient_boosting': (
            GradientBoostingClassifier,
            {'n_estimators': 100, 'random_state': 42}
        ),
        'decision_tree': (
            DecisionTreeClassifier,
            {'random_state': 42}
        ),
        'knn': (
            KNeighborsClassifier,
            {'n_neighbors': 5}
        ),
        'naive_bayes': (
            GaussianNB,
            {}
        ),
    }

    if XGB_AVAILABLE:
        models['xgboost'] = (
            xgb.XGBClassifier,
            {
                'n_estimators': 100,
                'random_state': 42,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
        )

    if LGB_AVAILABLE:
        models['lightgbm'] = (
            lgb.LGBMClassifier,
            {
                'n_estimators': 100,
                'random_state': 42,
                'verbose': -1
            }
        )

    return models


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int
) -> Dict[str, float]:
    """
    Evaluate a fitted model on test data.

    Args:
        model: Fitted sklearn-compatible model
        X_train, y_train: Training data (for gap calculation)
        X_test, y_test: Test data
        n_classes: Number of classes

    Returns:
        Dict with accuracy, f1, precision, recall, train_acc, gap
    """
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate metrics
    average = 'binary' if n_classes == 2 else 'macro'

    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_f1 = f1_score(y_test, y_pred_test, average=average, zero_division=0)
        test_precision = precision_score(y_test, y_pred_test, average=average, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, average=average, zero_division=0)

    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'train_accuracy': train_acc,
        'train_test_gap': train_acc - test_acc
    }


def run_baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str = 'xgboost',
    n_classes: int = 2
) -> Dict[str, Any]:
    """
    Run cross-validation for a single baseline model.

    Args:
        X: Features (preprocessed numpy array)
        y: Target (numpy array)
        splits: List of (train_idx, test_idx) tuples
        model_name: Name of baseline model
        n_classes: Number of classes

    Returns:
        Dict with mean, std, and per-fold results
    """
    models = get_baseline_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    model_class, params = models[model_name]

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model
        model = model_class(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, n_classes)
        metrics['fold'] = fold_idx
        fold_results.append(metrics)

    # Aggregate
    df = pd.DataFrame(fold_results)

    return {
        'model': model_name,
        'accuracy_mean': df['accuracy'].mean(),
        'accuracy_std': df['accuracy'].std(),
        'f1_mean': df['f1'].mean(),
        'f1_std': df['f1'].std(),
        'gap_mean': df['train_test_gap'].mean(),
        'gap_std': df['train_test_gap'].std(),
        'per_fold': fold_results
    }


def run_all_baselines(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_classes: int = 2,
    models_to_run: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run all baseline models and return comparison DataFrame.

    Args:
        X: Features (preprocessed numpy array)
        y: Target (numpy array)
        splits: CV splits
        n_classes: Number of classes
        models_to_run: Optional list of model names to run

    Returns:
        DataFrame with baseline results
    """
    models = get_baseline_models()
    if models_to_run is None:
        models_to_run = list(models.keys())

    results = []
    for model_name in models_to_run:
        try:
            print(f"  Running {model_name}...")
            result = run_baseline_cv(X, y, splits, model_name, n_classes)
            results.append({
                'model': model_name,
                'accuracy': f"{result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}",
                'f1': f"{result['f1_mean']:.4f} +/- {result['f1_std']:.4f}",
                'gap': f"{result['gap_mean']:.4f} +/- {result['gap_std']:.4f}",
                'accuracy_mean': result['accuracy_mean'],
                'f1_mean': result['f1_mean'],
                'gap_mean': result['gap_mean']
            })
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results.append({
                'model': model_name,
                'accuracy': 'FAILED',
                'f1': 'FAILED',
                'gap': 'N/A',
                'accuracy_mean': 0.0,
                'f1_mean': 0.0,
                'gap_mean': 0.0
            })

    df = pd.DataFrame(results)
    df = df.sort_values('f1_mean', ascending=False)

    return df


def get_best_baseline(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_classes: int = 2,
    metric: str = 'f1'
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Find the best baseline model for a dataset.

    Args:
        X, y: Data
        splits: CV splits
        n_classes: Number of classes
        metric: Metric to optimize ('f1' or 'accuracy')

    Returns:
        (best_model_name, best_score, full_results)
    """
    df = run_all_baselines(X, y, splits, n_classes)

    metric_col = f'{metric}_mean'
    best_row = df.loc[df[metric_col].idxmax()]

    return best_row['model'], best_row[metric_col], df.to_dict('records')


if __name__ == "__main__":
    # Test baseline runner
    from openml_loader import load_dataset, get_cv_splits, preprocess_features

    print("Testing baseline runner on credit-g dataset...")
    X, y, info = load_dataset(31)  # credit-g

    # Get splits
    splits = get_cv_splits(X, y, n_splits=5)

    # Preprocess
    X_full = X.copy()
    y_full = y.values

    # For simplicity, preprocess using first split
    train_idx, test_idx = splits[0]
    X_processed, _ = preprocess_features(X_full.iloc[train_idx], X_full.iloc[test_idx])

    # Need to preprocess all data consistently
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Simple full-data preprocessing
    X_num = X_full.select_dtypes(include=[np.number]).values
    imputer = SimpleImputer(strategy='median')
    X_num = imputer.fit_transform(X_num)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    print(f"\nDataset: {info.name}")
    print(f"Samples: {info.n_samples}, Features: {X_num.shape[1]}, Classes: {info.n_classes}")
    print(f"Imbalance ratio: {info.imbalance_ratio:.2f}")
    print("\nRunning baselines...")

    df = run_all_baselines(X_num, y_full, splits, info.n_classes)
    print("\n" + df[['model', 'accuracy', 'f1', 'gap']].to_string(index=False))
