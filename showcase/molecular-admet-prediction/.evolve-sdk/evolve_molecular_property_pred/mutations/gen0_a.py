"""
Generation 0 Variant A: Simple Random Forest with Morgan Fingerprints
Approach: Classic cheminformatics baseline using circular fingerprints
Strategy: Focus on simplicity and interpretability as foundation for evolution
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Simple Random Forest predictor using Morgan fingerprints.

    This is a clean, interpretable baseline that serves as one starting point
    for evolutionary optimization. Uses standard cheminformatics features
    with a proven algorithm.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048

        # Simple RF with basic tuning
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

    def _smiles_to_fingerprint(self, smiles_list):
        """Convert SMILES strings to Morgan fingerprints."""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vector
                fp = np.zeros(self.fp_bits)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)
            fingerprints.append(fp)
        return np.array(fingerprints)

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._smiles_to_fingerprint(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._smiles_to_fingerprint(X_smiles)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)


if __name__ == '__main__':
    # Quick test to ensure it works
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    ]

    # Dummy training
    predictor.fit(test_smiles * 5, [0, 1] * 5)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 0A test: {proba}")