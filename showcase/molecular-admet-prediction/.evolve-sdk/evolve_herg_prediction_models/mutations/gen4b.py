"""
Gen4-B: Random Forest with Threshold Optimization via F1 Score

Mutation from: gen0_a (Random Forest with Morgan Fingerprints)
Mutation type: Threshold optimization
Change: Added automatic threshold optimization during training using F1 score
        maximization on out-of-bag predictions.

Hypothesis: The default 0.5 threshold is often suboptimal for imbalanced datasets
like hERG prediction. By finding the threshold that maximizes F1 score on
out-of-bag predictions during training, we can improve the balance between
precision and recall without overfitting. The OOB approach provides unbiased
estimates since each sample is predicted by trees that didn't see it during training.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


class HERGPredictor:
    """Random Forest with optimized decision threshold for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,  # MUTATION: Enable OOB for threshold tuning
            random_state=random_state,
            n_jobs=-1
        )
        self._feature_names = None
        self._feature_importances = None
        # MUTATION: Optimal threshold found during training
        self._optimal_threshold = 0.5

    def _calculate_features(self, smiles_list):
        """Calculate Morgan fingerprints for molecules."""
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

    def _find_optimal_threshold(self, y_true, y_proba):
        """Find threshold that maximizes F1 score.

        MUTATION: Uses grid search over thresholds to find optimal decision boundary.
        """
        best_f1 = 0.0
        best_threshold = 0.5

        # Search over range of thresholds
        for threshold in np.arange(0.3, 0.7, 0.02):
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate F1 score manually to avoid sklearn import
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def fit(self, X_smiles, y):
        """Train the model and optimize decision threshold."""
        X = self._calculate_features(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_

        # MUTATION: Optimize threshold using OOB predictions
        if hasattr(self.model, 'oob_decision_function_'):
            oob_proba = self.model.oob_decision_function_[:, 1]
            self._optimal_threshold = self._find_optimal_threshold(y, oob_proba)

        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=None):
        """Binary prediction using optimized threshold.

        MUTATION: Uses optimized threshold by default instead of 0.5.
        """
        if threshold is None:
            threshold = self._optimal_threshold
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
