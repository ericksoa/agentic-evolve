"""
Baseline Logistic Regression classifier.

This is a starter solution for evolve-sdk to mutate.
Logistic Regression is often the best choice for small datasets.
"""

from sklearn.linear_model import LogisticRegression


def get_classifier():
    """
    Return a configured Logistic Regression classifier.

    Key parameters to evolve:
    - C: Inverse regularization strength (lower = more regularization)
    - class_weight: 'balanced' for imbalanced datasets
    - max_iter: Increase if convergence issues
    - solver: 'lbfgs' (default), 'liblinear' for small datasets
    """
    return LogisticRegression(
        C=1.0,  # Evolve: try [0.01, 0.1, 1.0, 10.0, 100.0]
        class_weight='balanced',  # Good for imbalanced datasets
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )


# For direct testing
if __name__ == '__main__':
    clf = get_classifier()
    print(f"Classifier: {clf}")
    print(f"Parameters: {clf.get_params()}")
