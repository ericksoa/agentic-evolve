"""
Generation 2 Crossover X: Enhanced Random Forest with Multi-Fingerprint Features
Approach: Hybrid combining best aspects of all three parents
Strategy: RF algorithm (Gen1A) + comprehensive features (Gen0D) + clean architecture (Gen0A)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Crossover hybrid: Random Forest with multi-fingerprint features and comprehensive descriptors.

    Inherited from parents:
    - Gen1A: Random Forest algorithm and core molecular descriptors
    - Gen0D: Multi-fingerprint approach (Morgan + MACCS) and advanced descriptors
    - Gen0A: Clean architecture and proven hyperparameters

    Combines the proven Random Forest approach with richer feature representation.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.morgan_radius = 2
        self.morgan_bits = 2048

        # Random Forest from Gen1A (best performer) with same hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Preprocessing from Gen0D for handling advanced descriptors
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _morgan_fingerprint(self, mol):
        """Generate Morgan fingerprint (from Gen1A/Gen0D)."""
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
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)

    def _core_descriptors(self, mol):
        """Calculate core molecular descriptors (from Gen1A)."""
        if mol is None:
            return [np.nan] * 8

        try:
            return [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                rdMolDescriptors.CalcNumAromaticRings(mol), # Aromatic rings
                rdMolDescriptors.CalcNumAliphaticRings(mol) # Aliphatic rings
            ]
        except:
            return [np.nan] * 8

    def _herg_specific_descriptors(self, mol):
        """Calculate hERG-specific descriptors (from Gen0D)."""
        if mol is None:
            return [np.nan] * 12

        try:
            return [
                # Electronic and topological properties
                Descriptors.BalabanJ(mol),
                rdMolDescriptors.BertzCT(mol),
                Descriptors.Ipc(mol),                      # Information content
                rdMolDescriptors.Kappa1(mol),              # Molecular connectivity
                rdMolDescriptors.Kappa2(mol),

                # Ring features important for hERG
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeterocycles(mol),

                # hERG pharmacophore features
                Descriptors.fr_NH0(mol),                   # Tertiary amines
                Descriptors.fr_NH1(mol),                   # Secondary amines
                Descriptors.fr_NH2(mol),                   # Primary amines
                Descriptors.fr_piperdine(mol),             # Piperidine rings
            ]
        except:
            return [np.nan] * 12

    def _extract_features(self, smiles_list):
        """Extract hybrid multi-scale features combining all parent approaches."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Multi-fingerprint approach from Gen0D
            morgan_fp = self._morgan_fingerprint(mol)
            maccs_fp = self._maccs_fingerprint(mol)

            # Molecular descriptors: core (Gen1A) + hERG-specific (Gen0D)
            core_desc = self._core_descriptors(mol)
            herg_desc = self._herg_specific_descriptors(mol)

            # Combine all features
            combined = np.concatenate([
                morgan_fp,      # 2048 bits
                maccs_fp,       # 167 bits
                core_desc,      # 8 descriptors
                herg_desc       # 12 descriptors
            ])

            all_features.append(combined)

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._extract_features(X_smiles)
        y = np.array(y)

        # Handle missing values in descriptor portions (from Gen0D approach)
        X = self.imputer.fit_transform(X)

        # Scale only the descriptor features, not the binary fingerprints
        # Fingerprints are in columns 0:2215, descriptors in 2215:2235
        fingerprints = X[:, :2215]  # Morgan (2048) + MACCS (167) = 2215
        descriptors = X[:, 2215:]   # Core (8) + hERG (12) = 20

        # Scale only descriptors
        descriptors_scaled = self.scaler.fit_transform(descriptors)

        # Recombine
        X = np.hstack([fingerprints, descriptors_scaled])

        # Use Random Forest from successful Gen1A
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._extract_features(X_smiles)
        X = self.imputer.transform(X)

        # Apply same scaling split as in training
        fingerprints = X[:, :2215]
        descriptors = X[:, 2215:]
        descriptors_scaled = self.scaler.transform(descriptors)
        X = np.hstack([fingerprints, descriptors_scaled])

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_info(self):
        """Get information about the feature composition."""
        return {
            'morgan_fingerprint': 2048,
            'maccs_keys': 167,
            'core_descriptors': 8,
            'herg_specific_descriptors': 12,
            'total_features': 2235,
            'algorithm': 'Random Forest',
            'preprocessing': 'Imputation + Selective Scaling'
        }


if __name__ == '__main__':
    # Test the hybrid
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',         # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',     # Caffeine
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12', # Chloroquine
    ]

    # Dummy training
    train_smiles = test_smiles * 10
    train_labels = [0, 1, 1] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 2X crossover test: {proba}")
    print(f"Feature composition: {predictor.get_feature_info()}")