"""
Generation 4 Crossover: Enhanced Random Forest with Multi-Scale Features
Approach: Hybrid combining best aspects of gen1a (high fitness) with gen0_d innovations
Strategy: RF architecture + multi-fingerprint fusion + curated descriptor set + robust preprocessing
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Hybrid Random Forest predictor with multi-scale feature representation.

    Combines:
    - gen1a: Proven RF architecture + focused molecular descriptors (fitness 0.73)
    - gen0_d: Multi-fingerprint approach (Morgan+MACCS) + hERG-specific descriptors
    - Enhanced preprocessing for robustness
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Fingerprint parameters (optimized balance from parents)
        self.morgan_radius = 2
        self.morgan_bits = 1024  # Reduced from gen0_d for efficiency

        # RF architecture from best parent (gen1a) with minor enhancements
        self.model = RandomForestClassifier(
            n_estimators=150,  # Increased from gen1a for better ensemble
            max_depth=12,      # Slightly increased for more complex features
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Preprocessing from gen0_d for robustness
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _morgan_fingerprint(self, mol):
        """Generate Morgan fingerprint (from both parents)."""
        if mol is None:
            return np.zeros(self.morgan_bits)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
            return np.array(fp)
        except:
            return np.zeros(self.morgan_bits)

    def _maccs_fingerprint(self, mol):
        """Generate MACCS keys (from gen0_d)."""
        if mol is None:
            return np.zeros(167)
        try:
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)

    def _enhanced_descriptors(self, mol):
        """
        Curated descriptor set combining best from both parents.
        - gen1a: Core ADMET descriptors (8 features)
        - gen0_d: hERG-specific pharmacophore features
        """
        if mol is None:
            return [np.nan] * 15

        try:
            descriptors = [
                # Core descriptors from gen1a (proven effective)
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                rdMolDescriptors.CalcNumAromaticRings(mol), # Aromatic rings
                rdMolDescriptors.CalcNumAliphaticRings(mol), # Aliphatic rings

                # hERG-specific features from gen0_d (pharmacophore relevance)
                Descriptors.fr_NH0(mol),        # Tertiary amines (hERG binding)
                Descriptors.fr_NH1(mol),        # Secondary amines
                Descriptors.fr_NH2(mol),        # Primary amines
                Descriptors.fr_piperdine(mol),  # Piperidine (key hERG pharmacophore)
                Descriptors.fr_piperzine(mol),  # Piperazine (key hERG pharmacophore)

                # Additional structural complexity features
                rdMolDescriptors.BertzCT(mol),   # Molecular complexity
                Descriptors.NumHeterocycles(mol) # Heterocycle count
            ]
            return descriptors
        except:
            return [np.nan] * 15

    def _extract_hybrid_features(self, smiles_list):
        """
        Extract hybrid feature set combining strengths from all parents.
        Features: Morgan + MACCS + Enhanced descriptors
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Multi-fingerprint approach (gen0_d innovation)
            morgan_fp = self._morgan_fingerprint(mol)
            maccs_fp = self._maccs_fingerprint(mol)

            # Enhanced descriptors (gen1a + gen0_d best features)
            descriptors = self._enhanced_descriptors(mol)

            # Combine all features
            combined = np.concatenate([
                morgan_fp,      # 1024 features
                maccs_fp,       # 167 features
                descriptors     # 15 features
            ])

            all_features.append(combined)

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train with robust preprocessing and proven RF architecture."""
        X = self._extract_hybrid_features(X_smiles)
        y = np.array(y)

        # Robust preprocessing from gen0_d
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        # Train with proven RF approach from gen1a
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._extract_hybrid_features(X_smiles)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_info(self):
        """Get information about the hybrid feature representation."""
        return {
            'morgan_bits': self.morgan_bits,
            'maccs_keys': 167,
            'descriptors': 15,
            'total_features': self.morgan_bits + 167 + 15,
            'parents_combined': ['gen1a_rf_architecture', 'gen0_d_multi_fingerprints', 'gen1a_core_descriptors', 'gen0_d_herg_pharmacophores'],
            'preprocessing': 'imputation + standardization'
        }


if __name__ == '__main__':
    # Test the hybrid predictor
    predictor = HERGPredictor()

    # Test molecules including hERG-relevant structures
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin (baseline)
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine (baseline)
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine (hERG blocker)
        'CN(C)CCN1c2ccccc2Sc2ccc(C(F)(F)F)cc21',  # Trifluoperazine-like (hERG blocker)
    ]

    # Training with sufficient data
    train_smiles = test_smiles * 15
    train_labels = [0, 0, 1, 1] * 15

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 4X hybrid test: {proba}")
    print(f"Feature info: {predictor.get_feature_info()}")
    print("Hybrid successfully combines RF architecture + multi-fingerprints + enhanced descriptors")