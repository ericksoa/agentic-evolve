"""
Baseline Random Forest classifier.

This is a starter solution for evolve-sdk to mutate.
Random Forest provides good out-of-box performance on many datasets.
"""

from sklearn.ensemble import RandomForestClassifier


def get_classifier():
    """
    Return a configured Random Forest classifier.

    Key parameters to evolve:
    - n_estimators: Number of trees [50, 100, 200, 500]
    - max_depth: Tree depth [None, 5, 10, 20]
    - min_samples_split: Min samples to split [2, 5, 10]
    - min_samples_leaf: Min samples in leaf [1, 2, 4]
    - class_weight: 'balanced' for imbalanced datasets
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,  # Evolve: try [None, 5, 10, 20]
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all cores
    )


# For direct testing
if __name__ == '__main__':
    clf = get_classifier()
    print(f"Classifier: {clf}")
    print(f"Parameters: {clf.get_params()}")
