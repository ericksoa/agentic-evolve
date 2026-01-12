"""
Simple Random Forest with Morgan Fingerprints - Fast and Interpretable

This solution uses Morgan fingerprints (circular fingerprints) with Random Forest.
Morgan fingerprints are efficient to compute and capture local molecular environments.
Random Forest provides good baseline performance with interpretability.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class HERGPredictor:
    """
    hERG toxicity predictor using Morgan fingerprints and Random Forest.

    Optimized for fast inference with good baseline performance.
    Uses circular fingerprints to capture molecular substructures.
    """

    def __init__(self, n_estimators=50, max_depth=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=1  # Single thread for consistent timing
        )
        self.scaler = StandardScaler()
        self._feature_importances = None

    def _calculate_morgan_fingerprints(self, smiles_list, radius=2, n_bits=1024):
        """Calculate Morgan fingerprints from SMILES."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vector
                fp = np.zeros(n_bits)
            else:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fp = np.array(morgan_fp)
            fingerprints.append(fp)

        return np.array(fingerprints)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings and labels."""
        X = self._calculate_morgan_fingerprints(X_smiles)
        y = np.array(y)

        # Standard scaling helps with Random Forest feature importance
        X = self.scaler.fit_transform(X.astype(float))

        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_morgan_fingerprints(X_smiles)
        X = self.scaler.transform(X.astype(float))
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance summary."""
        if self._feature_importances is None:
            return None

        # For Morgan fingerprints, just return top bit importances
        top_indices = np.argsort(self._feature_importances)[-20:][::-1]

        return {
            'features': [f'Morgan_bit_{i}' for i in top_indices],
            'importances': self._feature_importances[top_indices].tolist()
        }


if __name__ == '__main__':
    # Quick test
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
    ]

    train_smiles = test_smiles * 10
    train_labels = [0, 1, 0] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Morgan Fingerprint + Random Forest Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")