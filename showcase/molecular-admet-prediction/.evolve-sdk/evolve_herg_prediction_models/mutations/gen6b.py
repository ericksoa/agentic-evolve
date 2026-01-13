"""
Gen6-B: XGBoost with Morgan Fingerprints

Mutation from: gen0_a (Random Forest with Morgan Fingerprints)
Mutation type: Model swap
Change: Replace Random Forest with XGBoost classifier

Hypothesis: The parent (gen0_a) uses Random Forest which averages predictions from
independent trees. XGBoost is a gradient boosted ensemble that builds trees
sequentially, where each tree corrects the errors of the previous ones. This
sequential error correction often leads to better performance on classification
tasks. XGBoost also has built-in L1/L2 regularization which can help prevent
overfitting on the 2048-dimensional fingerprint space. Key differences:

1. Boosting vs Bagging: Sequential trees focused on hard examples vs independent trees
2. Built-in regularization: reg_alpha and reg_lambda control overfitting
3. Learning rate: Allows fine-tuning the contribution of each tree
4. Scale_pos_weight: Alternative to class_weight for handling imbalance
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


class HERGPredictor:
    """XGBoost with Morgan fingerprints for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._class_ratio = 1.0  # Will be computed during fit

        if HAS_XGBOOST:
            # MUTATION: Replaced RandomForestClassifier with XGBClassifier
            self.model = xgb.XGBClassifier(
                n_estimators=200,            # Same as parent
                max_depth=8,                 # Lower than RF (15) - boosting needs shallower trees
                learning_rate=0.1,           # Standard learning rate
                min_child_weight=2,          # Similar to min_samples_leaf
                subsample=0.8,               # Row sampling for regularization
                colsample_bytree=0.8,        # Column sampling for regularization
                reg_alpha=0.1,               # L1 regularization
                reg_lambda=1.0,              # L2 regularization
                scale_pos_weight=1.0,        # Will be updated in fit()
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            # Fallback to sklearn GradientBoosting if xgboost not available
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=random_state
            )

        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan fingerprints for molecules (same as parent)."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(2048)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                features = np.array(fp)
            all_features.append(features)

        self._feature_names = [f'Morgan2_{i}' for i in range(2048)]
        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings."""
        X = self._calculate_features(X_smiles)
        y = np.array(y)

        # MUTATION: Compute class imbalance ratio for scale_pos_weight
        if HAS_XGBOOST:
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0:
                self._class_ratio = n_neg / n_pos
                self.model.set_params(scale_pos_weight=self._class_ratio)

        self.model.fit(X, y)

        # Get feature importances
        if HAS_XGBOOST:
            self._feature_importances = self.model.feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            self._feature_importances = self.model.feature_importances_

        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return top 30 most important features."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
