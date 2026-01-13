"""
Gen2-B: Gradient Boosting with Morgan Fingerprints

Mutation from: gen0_a (Random Forest with Morgan Fingerprints)
Mutation type: Model swap
Change: Replace RandomForestClassifier with HistGradientBoostingClassifier

Hypothesis: Gradient Boosting builds trees sequentially to correct previous
errors, which often leads to better generalization than Random Forest's
parallel ensemble approach. HistGradientBoostingClassifier is optimized for
large datasets and should train faster while potentially achieving higher
AUC-ROC on the hERG prediction task.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


class HERGPredictor:
    """Gradient Boosting with Morgan fingerprints for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            min_samples_leaf=20,
            l2_regularization=0.1,
            max_bins=255,
            random_state=random_state
        )
        self._feature_names = None
        self._feature_importances = None

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

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings."""
        X = self._calculate_features(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return top 30 most important features.

        Note: HistGradientBoostingClassifier doesn't have feature_importances_
        by default, so we return None (this is acceptable per the interface).
        """
        return None
