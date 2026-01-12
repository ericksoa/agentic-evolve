"""
Generation 2 Variant B: Enhanced RF with Morgan Fingerprints + Molecular Descriptors
Mutation: Feature addition - added physicochemical descriptors alongside fingerprints
Parent: Generation 0 Variant A
Strategy: Enrich feature space with interpretable molecular properties
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Enhanced Random Forest predictor using Morgan fingerprints + molecular descriptors.

    Mutation from Gen0A: Added physicochemical descriptors (MW, LogP, TPSA, etc.)
    to complement the Morgan fingerprints for a richer feature representation.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048

        # Same RF as parent - only changing feature engineering
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
                desc = np.zeros(8)  # 8 descriptors
            else:
                # Morgan fingerprint (unchanged from parent)
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)

                # NEW: Add molecular descriptors relevant to hERG blocking
                desc = np.array([
                    Descriptors.MolWt(mol),           # Molecular weight
                    Descriptors.MolLogP(mol),         # LogP (lipophilicity)
                    Descriptors.TPSA(mol),           # Topological polar surface area
                    Descriptors.NumHDonors(mol),     # H-bond donors
                    Descriptors.NumHAcceptors(mol),  # H-bond acceptors
                    Descriptors.NumRotatableBonds(mol), # Rotatable bonds
                    Descriptors.NumAromaticRings(mol),  # Aromatic rings
                    Descriptors.BertzCT(mol)         # Complexity
                ])

            fingerprints.append(fp)
            descriptors.append(desc)

        # Combine fingerprints and descriptors
        fingerprints = np.array(fingerprints)
        descriptors = np.array(descriptors)

        # Simple normalization for descriptors (prevent one feature dominating)
        desc_means = np.mean(descriptors, axis=0)
        desc_stds = np.std(descriptors, axis=0) + 1e-8  # Avoid division by zero
        descriptors = (descriptors - desc_means) / desc_stds

        # Concatenate features
        features = np.hstack([fingerprints, descriptors])
        return features

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

    print(f"Generation 2B test: {proba}")