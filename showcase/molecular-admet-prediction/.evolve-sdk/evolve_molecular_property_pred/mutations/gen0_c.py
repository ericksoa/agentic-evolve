"""
Generation 0 Variant C: SVM with Hybrid Features (Fingerprints + Descriptors)
Approach: Combine structural patterns with physicochemical properties
Strategy: Different algorithm (SVM) with complementary feature fusion
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    SVM predictor with hybrid feature representation.

    Combines Morgan fingerprints (structural patterns) with molecular
    descriptors (physicochemical properties) to capture both local
    substructural features and global molecular properties.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 1024  # Smaller fingerprint for efficiency

        # SVM with RBF kernel and probability calibration
        base_svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=random_state
        )

        # Calibrated for probability estimation
        self.model = CalibratedClassifierCV(base_svm, method='isotonic', cv=3)

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _smiles_to_fingerprint(self, smiles_list):
        """Convert SMILES to Morgan fingerprints."""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = np.zeros(self.fp_bits)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp = np.array(fp)
            fingerprints.append(fp)
        return np.array(fingerprints)

    def _calculate_key_descriptors(self, smiles_list):
        """Calculate essential molecular descriptors."""
        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                desc = [np.nan] * 15
            else:
                desc = [
                    # Core hERG-relevant properties
                    Descriptors.MolLogP(mol),
                    Descriptors.MolWt(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumAromaticRings(mol),

                    # Hydrogen bonding
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),

                    # Nitrogen content (critical for hERG)
                    Descriptors.fr_NH0(mol),  # Tertiary N
                    Descriptors.fr_NH1(mol),  # Secondary N
                    Descriptors.fr_NH2(mol),  # Primary N

                    # Size and complexity
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol),

                    # Electronic properties
                    rdMolDescriptors.BertzCT(mol),
                    Descriptors.BalabanJ(mol),
                    Descriptors.LabuteASA(mol),
                ]
            descriptors.append(desc)
        return np.array(descriptors)

    def _combine_features(self, smiles_list):
        """Combine fingerprints and descriptors into hybrid features."""
        fp_features = self._smiles_to_fingerprint(smiles_list)
        desc_features = self._calculate_key_descriptors(smiles_list)

        # Impute missing descriptor values
        desc_features = self.imputer.fit_transform(desc_features) if hasattr(self.imputer, 'statistics_') else self.imputer.transform(desc_features)

        # Concatenate features
        combined = np.hstack([fp_features, desc_features])
        return combined

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._combine_features(X_smiles)
        y = np.array(y)

        # Scale all features (important for SVM)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._combine_features(X_smiles)
        X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_info(self):
        """Get information about feature composition."""
        return {
            'fingerprint_bits': self.fp_bits,
            'descriptor_count': 15,
            'total_features': self.fp_bits + 15,
            'kernel': 'RBF',
            'calibration': 'Isotonic'
        }


if __name__ == '__main__':
    # Quick test to ensure it works
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine (known hERG blocker)
    ]

    # Dummy training
    predictor.fit(test_smiles * 4, [0, 1, 1] * 4)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 0C test: {proba}")
    print(f"Feature info: {predictor.get_feature_info()}")