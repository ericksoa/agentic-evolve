"""
Generation 1 Crossover Hybrid: Multi-Scale Random Forest
Approach: Combine comprehensive feature extraction with interpretable ensemble model
Strategy: Rich features from neural network approach + robust Random Forest algorithm

CROSSOVER LINEAGE:
- Feature extraction strategy from gen0_d.py (multi-scale fingerprints + descriptors)
- Algorithm choice from gen0_a.py (Random Forest for interpretability)
- hERG-specific descriptors from gen0_c.py (targeted molecular properties)
- Robust preprocessing pipeline from gen0_c.py and gen0_d.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Crossover hybrid combining multi-scale feature extraction with Random Forest.

    HYBRID DESIGN:
    - Uses comprehensive feature set (Morgan + MACCS + Avalon + descriptors) from gen0_d
    - Employs Random Forest algorithm from gen0_a for interpretability and robustness
    - Incorporates hERG-specific descriptor selection from gen0_c
    - Applies robust preprocessing pipeline for feature standardization

    This combination aims to capture the feature richness of deep learning approaches
    while maintaining the interpretability and reliability of tree-based methods.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Feature parameters (from gen0_d but optimized)
        self.morgan_bits = 1024  # Balanced size from gen0_c
        self.morgan_radius = 2

        # Enhanced Random Forest (from gen0_a but with more trees for rich features)
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees to handle richer feature space
            max_depth=15,      # Deeper for complex features
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Better for high-dimensional features
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1  # Parallel processing
        )

        # Preprocessing (from gen0_c and gen0_d)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _morgan_fingerprint(self, mol):
        """Generate Morgan fingerprint (from gen0_d)."""
        if mol is None:
            return np.zeros(self.morgan_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
        return np.array(fp)

    def _maccs_fingerprint(self, mol):
        """Generate MACCS structural keys (from gen0_d)."""
        if mol is None:
            return np.zeros(167)
        try:
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)

    def _avalon_fingerprint(self, mol):
        """Generate Avalon fingerprint (from gen0_d)."""
        if mol is None:
            return np.zeros(512)
        try:
            fp = pyAvalonTools.GetAvalonFP(mol, 512)
            return np.array(fp)
        except:
            return np.zeros(512)

    def _herg_focused_descriptors(self, mol):
        """
        Calculate hERG-focused molecular descriptors.
        Combines the comprehensive approach from gen0_d with
        the hERG-specific focus from gen0_c.
        """
        if mol is None:
            return [np.nan] * 18

        try:
            return [
                # Core physicochemical (from gen0_d + gen0_c)
                Descriptors.MolLogP(mol),
                Descriptors.MolWt(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),

                # hERG-critical nitrogen features (from gen0_c)
                Descriptors.fr_NH0(mol),  # Tertiary amines (critical for hERG)
                Descriptors.fr_NH1(mol),  # Secondary amines
                Descriptors.fr_NH2(mol),  # Primary amines
                Descriptors.fr_piperdine(mol),  # Piperidine (hERG pharmacophore)
                Descriptors.fr_piperzine(mol),  # Piperazine

                # Aromatic and structural features (combined from all parents)
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumRotatableBonds(mol),

                # Advanced topological (from gen0_d)
                rdMolDescriptors.BertzCT(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.LabuteASA(mol),
            ]
        except:
            return [np.nan] * 18

    def _extract_hybrid_features(self, smiles_list):
        """
        Extract comprehensive multi-scale features.
        Combines the feature extraction strategy from gen0_d with
        the focused approach from gen0_c.
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Multi-scale fingerprints (from gen0_d)
            morgan_fp = self._morgan_fingerprint(mol)
            maccs_fp = self._maccs_fingerprint(mol)
            avalon_fp = self._avalon_fingerprint(mol)

            # hERG-focused descriptors (hybrid from gen0_c + gen0_d)
            descriptors = self._herg_focused_descriptors(mol)

            # Concatenate all features
            combined = np.concatenate([
                morgan_fp,      # 1024 bits
                maccs_fp,       # 167 keys
                avalon_fp,      # 512 bits
                descriptors     # 18 descriptors
            ])

            all_features.append(combined)

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """
        Train on SMILES strings and binary labels.
        Uses preprocessing pipeline from gen0_c and gen0_d.
        """
        X = self._extract_hybrid_features(X_smiles)
        y = np.array(y)

        # Handle missing values in descriptor portion (from gen0_d)
        X = self.imputer.fit_transform(X)

        # Scale features (from gen0_c)
        X = self.scaler.fit_transform(X)

        # Train Random Forest (from gen0_a but enhanced)
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

    def get_feature_importance(self):
        """
        Get feature importance from Random Forest.
        This interpretability is inherited from gen0_a approach.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def get_hybrid_info(self):
        """Get information about the hybrid architecture."""
        total_features = self.morgan_bits + 167 + 512 + 18
        return {
            'algorithm': 'Random Forest (from gen0_a)',
            'features': 'Multi-scale hybrid (from gen0_d + gen0_c)',
            'preprocessing': 'Scaling + Imputation (from gen0_c + gen0_d)',
            'total_features': total_features,
            'feature_breakdown': {
                'morgan_fingerprint': self.morgan_bits,
                'maccs_keys': 167,
                'avalon_fingerprint': 512,
                'herg_descriptors': 18
            },
            'n_estimators': 200,
            'interpretability': 'High (feature importance available)'
        }


if __name__ == '__main__':
    # Test the hybrid predictor
    predictor = HERGPredictor()

    # Test molecules (from gen0_d)
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine (known hERG blocker)
    ]

    # Training data (sufficient for Random Forest)
    train_smiles = test_smiles * 10
    train_labels = [0, 1, 1] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 1 Crossover test: {proba}")
    print(f"Hybrid info: {predictor.get_hybrid_info()}")

    # Show feature importance (unique to Random Forest approach)
    importance = predictor.get_feature_importance()
    if importance is not None:
        print(f"Top 5 most important features: {np.argsort(importance)[-5:]}")