"""
Generation 4 Variant B: Random Forest with Morgan Fingerprints + Molecular Descriptors
Approach: Feature addition mutation - enhance Morgan fingerprints with molecular descriptors
Strategy: Add physicochemical descriptors to capture different aspects of molecular properties
Mutation: Extended feature set by adding LogP, MW, TPSA, HBD, HBA descriptors
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Random Forest predictor using Morgan fingerprints + molecular descriptors.

    Mutation from gen0_a: Added 5 key molecular descriptors (LogP, MW, TPSA, HBD, HBA)
    to supplement Morgan fingerprints. These descriptors capture physicochemical
    properties that may be important for hERG binding prediction.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048

        # Same RF parameters as parent
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

    def _smiles_to_features(self, smiles_list):
        """Convert SMILES to Morgan fingerprints + molecular descriptors."""
        fingerprints = []
        descriptors = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vectors
                fp = np.zeros(self.fp_bits)
                desc = np.zeros(5)  # 5 descriptors
            else:
                # Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)

                # Molecular descriptors
                try:
                    desc = np.array([
                        Descriptors.MolLogP(mol),      # LogP
                        Descriptors.MolWt(mol),        # Molecular weight
                        Descriptors.TPSA(mol),         # Topological polar surface area
                        Descriptors.NumHDonors(mol),   # Hydrogen bond donors
                        Descriptors.NumHAcceptors(mol) # Hydrogen bond acceptors
                    ])
                except:
                    desc = np.zeros(5)

            fingerprints.append(fp)
            descriptors.append(desc)

        # Combine fingerprints and descriptors
        X_fp = np.array(fingerprints)
        X_desc = np.array(descriptors)
        X_combined = np.concatenate([X_fp, X_desc], axis=1)

        return X_combined

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._smiles_to_features(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._smiles_to_features(X_smiles)
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

    print(f"Generation 4B test: {proba}")