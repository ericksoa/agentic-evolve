"""
Generation 3 Crossover Hybrid: Enhanced Random Forest with Multi-Fingerprint Features
Approach: Combine Gen1A's winning RF+descriptors with Gen0D's multi-fingerprint diversity
Strategy: Feature fusion - take proven algorithm from best performer + feature diversity from complex parent

Parents Combined:
- Gen1A (0.73 fitness): Random Forest algorithm + core molecular descriptors
- Gen0D (0.0 fitness): Multi-fingerprint extraction (Morgan + MACCS) + preprocessing
- Gen0A (0.0 fitness): Clean foundational structure
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Hybrid Random Forest predictor combining multi-fingerprint features with molecular descriptors.

    Crossover strategy:
    - Algorithm: Random Forest from Gen1A (proven high performance)
    - Features: Morgan + MACCS fingerprints from Gen0D + key descriptors from Gen1A
    - Preprocessing: Scaling from Gen0D for descriptor portion
    - Structure: Clean foundation from Gen0A
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Morgan fingerprint settings (from Gen1A - proven)
        self.morgan_radius = 2
        self.morgan_bits = 2048

        # Random Forest model from Gen1A (high-performing configuration)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Scaler for descriptor features (from Gen0D)
        self.scaler = StandardScaler()

    def _morgan_fingerprint(self, mol):
        """Generate Morgan fingerprint (from Gen1A/Gen0A)."""
        if mol is None:
            return np.zeros(self.morgan_bits)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
            return np.array(fp)
        except:
            return np.zeros(self.morgan_bits)

    def _maccs_fingerprint(self, mol):
        """Generate MACCS structural keys (from Gen0D)."""
        if mol is None:
            return np.zeros(167)
        try:
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)

    def _calculate_molecular_descriptors(self, mol):
        """Calculate key molecular descriptors for ADMET prediction (from Gen1A)."""
        if mol is None:
            return np.zeros(8)

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

    def _extract_hybrid_features(self, smiles_list):
        """Extract hybrid features combining multiple fingerprints + descriptors."""
        morgan_fps = []
        maccs_fps = []
        descriptors = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Multi-fingerprint approach from Gen0D
            morgan_fp = self._morgan_fingerprint(mol)
            maccs_fp = self._maccs_fingerprint(mol)

            # Key descriptors from Gen1A
            desc = self._calculate_molecular_descriptors(mol)

            morgan_fps.append(morgan_fp)
            maccs_fps.append(maccs_fp)
            descriptors.append(desc)

        morgan_fps = np.array(morgan_fps)
        maccs_fps = np.array(maccs_fps)
        descriptors = np.array(descriptors)

        # Scale only the descriptor features (from Gen0D preprocessing approach)
        if hasattr(self, '_fitted'):
            descriptors_scaled = self.scaler.transform(descriptors)
        else:
            descriptors_scaled = self.scaler.fit_transform(descriptors)
            self._fitted = True

        # Combine all features: fingerprints (binary) + scaled descriptors (continuous)
        combined_features = np.hstack([morgan_fps, maccs_fps, descriptors_scaled])
        return combined_features

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._extract_hybrid_features(X_smiles)
        y = np.array(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._extract_hybrid_features(X_smiles)
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_info(self):
        """Get information about the hybrid feature set."""
        return {
            'morgan_bits': self.morgan_bits,
            'maccs_keys': 167,
            'descriptors': 8,
            'total_features': self.morgan_bits + 167 + 8,
            'algorithm': 'Random Forest',
            'preprocessing': 'StandardScaler on descriptors only',
            'heritage': 'Gen1A algorithm + Gen0D multi-fingerprints + Gen0A structure'
        }


if __name__ == '__main__':
    # Test the hybrid predictor
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine
    ]

    # Training with sufficient data
    train_smiles = test_smiles * 10
    train_labels = [0, 1, 1] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 3X hybrid test: {proba}")
    print(f"Feature info: {predictor.get_feature_info()}")