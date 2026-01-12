"""
Generation 3 Variant B: Enhanced Feature Set with Molecular Descriptors
Approach: Add molecular descriptors to complement Morgan fingerprints
Strategy: Feature addition - combine fingerprints with physicochemical descriptors
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

    This mutation adds molecular descriptors (MW, LogP, TPSA, etc.) to the feature set
    alongside Morgan fingerprints to provide richer chemical information.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048

        # Same RF hyperparameters as parent
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

    def _calculate_descriptors(self, mol):
        """Calculate key molecular descriptors."""
        if mol is None:
            return np.zeros(8)  # Return zeros for invalid molecules

        try:
            descriptors = [
                Descriptors.MolWt(mol),          # Molecular weight
                Descriptors.MolLogP(mol),        # LogP (lipophilicity)
                Descriptors.TPSA(mol),           # Topological polar surface area
                Descriptors.NumHDonors(mol),     # Hydrogen bond donors
                Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
                Descriptors.NumRotatableBonds(mol), # Rotatable bonds
                Descriptors.NumAromaticRings(mol),  # Aromatic rings
                Descriptors.FractionCSP3(mol)    # Fraction of sp3 carbons
            ]
            return np.array(descriptors)
        except:
            return np.zeros(8)

    def _smiles_to_features(self, smiles_list):
        """Convert SMILES strings to combined fingerprints + descriptors."""
        fingerprints = []
        descriptors_list = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Morgan fingerprints
            if mol is None:
                fp = np.zeros(self.fp_bits)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)
            fingerprints.append(fp)

            # Molecular descriptors
            desc = self._calculate_descriptors(mol)
            descriptors_list.append(desc)

        # Combine fingerprints and descriptors
        fingerprints = np.array(fingerprints)
        descriptors = np.array(descriptors_list)

        # Concatenate features
        combined_features = np.hstack([fingerprints, descriptors])
        return combined_features

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

    print(f"Generation 3B test: {proba}")