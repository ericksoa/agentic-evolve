"""
Generation 4 Variant A: Enhanced Random Forest with Optimized Hyperparameters
Approach: Hyperparameter tuning - increased ensemble size and adjusted tree parameters
Strategy: Mutation from Gen1A with deeper trees and more estimators for improved performance
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Enhanced Random Forest predictor with optimized hyperparameters.

    Mutation from Gen1A: Increased n_estimators from 100 to 200,
    increased max_depth from 10 to 15, and adjusted min_samples parameters
    for better model capacity and reduced overfitting.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048

        # Optimized RF parameters - more estimators and deeper trees
        self.model = RandomForestClassifier(
            n_estimators=200,       # Increased from 100
            max_depth=15,          # Increased from 10
            min_samples_split=8,   # Increased from 5 for regularization
            min_samples_leaf=3,    # Increased from 2 for regularization
            max_features='sqrt',   # Added for better feature selection
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

    def _calculate_molecular_descriptors(self, mol):
        """Calculate key molecular descriptors for ADMET prediction."""
        if mol is None:
            return np.zeros(8)  # Return zeros for invalid molecules

        try:
            descriptors = [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                rdMolDescriptors.CalcNumAromaticRings(mol), # Aromatic rings
                rdMolDescriptors.CalcNumAliphaticRings(mol) # Aliphatic rings
            ]
            return np.array(descriptors)
        except:
            return np.zeros(8)

    def _smiles_to_features(self, smiles_list):
        """Convert SMILES strings to combined fingerprints + descriptors."""
        fingerprints = []
        descriptors = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Morgan fingerprint (same as parent)
            if mol is None:
                fp = np.zeros(self.fp_bits)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)
            fingerprints.append(fp)

            # Molecular descriptors (from Gen1A)
            desc = self._calculate_molecular_descriptors(mol)
            descriptors.append(desc)

        # Combine fingerprints and descriptors
        fingerprints = np.array(fingerprints)
        descriptors = np.array(descriptors)

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

    print(f"Generation 4A test: {proba}")