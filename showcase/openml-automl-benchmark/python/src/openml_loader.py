"""
OpenML dataset loading utilities.

Handles fetching datasets from OpenML API and preprocessing for evaluation.
"""

import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Metadata about an OpenML dataset."""
    dataset_id: int
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: Dict[str, int]
    imbalance_ratio: float
    has_missing: bool
    has_categorical: bool


def get_cc18_dataset_ids() -> List[int]:
    """
    Get the dataset IDs for OpenML-CC18 benchmark suite.

    OpenML-CC18 contains 72 classification datasets selected for benchmarking.
    Suite ID: 99
    """
    suite = openml.study.get_suite(99)  # OpenML-CC18
    return list(suite.data)


def get_pilot_dataset_ids() -> List[int]:
    """
    Get 5 representative pilot datasets for initial testing.

    Selected for diversity:
    - credit-g (31): 1000 samples, binary, imbalanced
    - diabetes (37): 768 samples, binary, classic
    - vehicle (54): 846 samples, 4-class
    - segment (36): 2310 samples, 7-class
    - kc1 (1067): 2109 samples, binary, software defect
    """
    return [31, 37, 54, 36, 1067]


def load_dataset(dataset_id: int) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    """
    Load a dataset from OpenML and preprocess it.

    Args:
        dataset_id: OpenML dataset ID

    Returns:
        X: Feature DataFrame
        y: Target Series
        info: Dataset metadata
    """
    # Fetch from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    # Handle target encoding
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name='target')

    # Compute metadata
    class_counts = y.value_counts().to_dict()
    n_classes = len(class_counts)
    majority = max(class_counts.values())
    minority = min(class_counts.values())
    imbalance_ratio = majority / minority if minority > 0 else float('inf')

    info = DatasetInfo(
        dataset_id=dataset_id,
        name=dataset.name,
        n_samples=len(X),
        n_features=X.shape[1],
        n_classes=n_classes,
        class_distribution={str(k): v for k, v in class_counts.items()},
        imbalance_ratio=imbalance_ratio,
        has_missing=X.isnull().any().any(),
        has_categorical=any(categorical_indicator) if categorical_indicator else False
    )

    return X, y, info


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = 'default'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features for ML models.

    Args:
        X_train: Training features
        X_test: Test features
        strategy: Preprocessing strategy ('default', 'minimal', 'aggressive')

    Returns:
        X_train_processed, X_test_processed as numpy arrays
    """
    # Separate numeric and categorical
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    processed_train = []
    processed_test = []

    # Handle numeric features
    if len(numeric_cols) > 0:
        X_train_num = X_train[numeric_cols].values
        X_test_num = X_test[numeric_cols].values

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_train_num = imputer.fit_transform(X_train_num)
        X_test_num = imputer.transform(X_test_num)

        # Scale
        if strategy in ['default', 'aggressive']:
            scaler = StandardScaler()
            X_train_num = scaler.fit_transform(X_train_num)
            X_test_num = scaler.transform(X_test_num)

        processed_train.append(X_train_num)
        processed_test.append(X_test_num)

    # Handle categorical features
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            # Simple label encoding
            le = LabelEncoder()
            train_vals = X_train[col].fillna('__MISSING__').astype(str)
            test_vals = X_test[col].fillna('__MISSING__').astype(str)

            # Fit on train, handle unseen values in test
            le.fit(list(set(train_vals) | set(test_vals)))

            train_encoded = le.transform(train_vals).reshape(-1, 1)
            test_encoded = le.transform(test_vals).reshape(-1, 1)

            processed_train.append(train_encoded)
            processed_test.append(test_encoded)

    X_train_final = np.hstack(processed_train) if processed_train else np.array([]).reshape(len(X_train), 0)
    X_test_final = np.hstack(processed_test) if processed_test else np.array([]).reshape(len(X_test), 0)

    return X_train_final, X_test_final


def get_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get stratified cross-validation splits.

    Args:
        X: Features
        y: Target
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        List of (train_idx, test_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))


def get_openml_task_splits(task_id: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get the official OpenML task splits for reproducibility.

    Args:
        task_id: OpenML task ID

    Returns:
        List of (train_idx, test_idx) tuples
    """
    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()

    splits = []
    for fold_idx in range(n_folds):
        train_idx, test_idx = task.get_train_test_split_indices(
            repeat=0,  # First repeat
            fold=fold_idx
        )
        splits.append((train_idx, test_idx))

    return splits


def load_dataset_with_task(task_id: int) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Load dataset and official splits from an OpenML task.

    Args:
        task_id: OpenML task ID

    Returns:
        X, y, info, splits
    """
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id

    X, y, info = load_dataset(dataset_id)
    splits = get_openml_task_splits(task_id)

    return X, y, info, splits


if __name__ == "__main__":
    # Test loading
    print("Loading pilot datasets...")
    for did in get_pilot_dataset_ids():
        try:
            X, y, info = load_dataset(did)
            print(f"  {info.name} (ID={did}): {info.n_samples} samples, "
                  f"{info.n_features} features, {info.n_classes} classes, "
                  f"imbalance={info.imbalance_ratio:.2f}")
        except Exception as e:
            print(f"  Failed to load {did}: {e}")
