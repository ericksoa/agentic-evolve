"""
Baseline XGBoost classifier.

This is a starter solution for evolve-sdk to mutate.
XGBoost is often the best performer on tabular data.
"""

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier


def get_classifier():
    """
    Return a configured XGBoost classifier.

    Key parameters to evolve:
    - n_estimators: Number of boosting rounds [50, 100, 200, 500]
    - max_depth: Tree depth [3, 5, 7, 10]
    - learning_rate: Step size [0.01, 0.05, 0.1, 0.2]
    - subsample: Row sampling [0.6, 0.8, 1.0]
    - colsample_bytree: Feature sampling [0.6, 0.8, 1.0]
    - scale_pos_weight: For imbalanced classes
    - reg_alpha: L1 regularization [0, 0.1, 1.0]
    - reg_lambda: L2 regularization [1, 2, 5]
    """
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,  # Evolve: try [3, 5, 7, 10]
            learning_rate=0.1,  # Evolve: try [0.01, 0.05, 0.1, 0.2]
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Evolve based on imbalance ratio
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    else:
        # Fallback to sklearn GradientBoosting
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )


# For direct testing
if __name__ == '__main__':
    clf = get_classifier()
    print(f"Classifier: {clf}")
    print(f"Parameters: {clf.get_params()}")
