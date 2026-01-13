"""
Gen0-A: Random Forest Baseline with Morgan Fingerprints

Approach: Simple Random Forest classifier with Morgan circular fingerprints.
This provides a strong baseline using a well-understood algorithm.
Optimized for the larger 16K molecule ChEMBL dataset with more estimators.

Key features:
- Morgan fingerprints (radius=2, 2048 bits) - captures local chemical environments
- Balanced class weights for imbalanced data
- Tuned hyperparameters for larger dataset
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


class HERGPredictor:
    """Random Forest with Morgan fingerprints for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
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
