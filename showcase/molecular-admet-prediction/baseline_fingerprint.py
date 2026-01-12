"""
Baseline hERG Predictor using Morgan Fingerprints + Random Forest.

This is a classic cheminformatics approach that serves as our starting baseline.
Morgan fingerprints (circular fingerprints) encode local chemical environments
around each atom, capturing substructural features relevant to molecular activity.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem


class HERGPredictor:
    """
    hERG toxicity predictor using Morgan fingerprints and Random Forest.

    Morgan fingerprints are circular fingerprints that encode the chemical
    environment around each atom up to a specified radius. They are effective
    for capturing substructural patterns associated with hERG binding.
    """

    def __init__(self, radius=2, n_bits=2048, n_estimators=100, random_state=42):
        self.radius = radius
        self.n_bits = n_bits
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
            class_weight='balanced'
        )
        self._feature_importances = None

    def _smiles_to_fingerprint(self, smiles_list):
        """Convert SMILES strings to Morgan fingerprints."""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vector
                fp = np.zeros(self.n_bits)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                fp = np.array(fp)
            fingerprints.append(fp)

        return np.array(fingerprints)

    def fit(self, X_smiles, y):
        """
        Train the model on SMILES strings and labels.

        Args:
            X_smiles: List of SMILES strings
            y: List/array of binary labels (1 = hERG blocker, 0 = non-blocker)
        """
        X = self._smiles_to_fingerprint(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """
        Predict probability of hERG blocking for each molecule.

        Args:
            X_smiles: List of SMILES strings

        Returns:
            Array of probabilities (probability of being a hERG blocker)
        """
        X = self._smiles_to_fingerprint(X_smiles)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """
        Get feature importance from the Random Forest model.

        Returns most important bit positions and their importance scores.
        """
        if self._feature_importances is None:
            return None

        # Get top 20 most important bits
        top_idx = np.argsort(self._feature_importances)[-20:][::-1]
        return {
            'top_bits': top_idx.tolist(),
            'importances': self._feature_importances[top_idx].tolist()
        }


if __name__ == '__main__':
    # Quick test
    predictor = HERGPredictor()

    # Example SMILES (aspirin, caffeine, ibuprofen)
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
    ]

    # Dummy training
    train_smiles = test_smiles * 10
    train_labels = [0, 1, 0] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")

    print("\nFeature importance:", predictor.get_feature_importance())
