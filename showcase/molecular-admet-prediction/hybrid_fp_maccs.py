"""
Solution: Hybrid Morgan + MACCS Fingerprints Predictor
Approach: Combine two complementary fingerprint types for richer structural encoding

Key features:
- Morgan fingerprints (ECFP4): Capture local circular substructures
- MACCS keys (167 bits): Predefined pharmacophore patterns
- Physicochemical descriptors: hERG-relevant features
- Random Forest classifier with balanced weighting
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """
    hERG predictor using dual fingerprint approach.

    Feature representation:
    - Morgan FP (2048 bits): Local structural patterns
    - MACCS keys (167 bits): Global pharmacophore features
    - Physicochemical (10 features): hERG-specific descriptors
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_bits = 2048
        self.fp_radius = 2
        self.scaler = StandardScaler()

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=1,
            random_state=random_state,
            class_weight='balanced'
        )
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan + MACCS + physicochemical features."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vector
                fp = np.zeros(self.fp_bits + 167 + 10)
            else:
                # Morgan fingerprints
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
                morgan_array = np.array(morgan)

                # MACCS keys (167 bits of pharmacophore patterns)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.array(maccs)

                # Physicochemical descriptors
                physchem = self._calculate_physchem(mol)

                fp = np.concatenate([morgan_array, maccs_array, physchem])

            features.append(fp)

        return np.array(features)

    def _calculate_physchem(self, mol):
        """Calculate key physicochemical descriptors for hERG."""
        return np.array([
            Descriptors.MolLogP(mol),               # Lipophilicity
            Descriptors.MolWt(mol),                 # Molecular weight
            rdMolDescriptors.CalcTPSA(mol),         # Polar surface area
            Descriptors.NumAromaticRings(mol),      # Aromatic content
            Lipinski.NumHDonors(mol),               # H-bond donors
            Lipinski.NumHAcceptors(mol),            # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),     # Flexibility
            self._count_basic_nitrogens(mol),       # Basic N (hERG key)
            rdMolDescriptors.CalcFractionCSP3(mol), # 3D character
            Descriptors.HeavyAtomCount(mol),        # Size
        ], dtype=float)

    def _count_basic_nitrogens(self, mol):
        """Count basic nitrogens - key pharmacophore for hERG binding."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetHybridization().name == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def fit(self, X_smiles, y):
        """Train on SMILES strings and labels."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale only physicochemical features (last 10)
        n_fp = self.fp_bits + 167
        X_fp = X[:, :n_fp]
        X_phys = self.scaler.fit_transform(X[:, n_fp:])
        X = np.concatenate([X_fp, X_phys], axis=1)

        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_

        return self

    def predict_proba(self, X_smiles):
        """Predict hERG blocking probability."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_fp = self.fp_bits + 167
        X_fp = X[:, :n_fp]
        X_phys = self.scaler.transform(X[:, n_fp:])
        X = np.concatenate([X_fp, X_phys], axis=1)

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance breakdown."""
        if self._feature_importances is None:
            return None

        morgan_imp = np.sum(self._feature_importances[:self.fp_bits])
        maccs_imp = np.sum(self._feature_importances[self.fp_bits:self.fp_bits+167])
        phys_imp = self._feature_importances[-10:]

        names = ['MorganFP', 'MACCSKeys', 'MolLogP', 'MolWt', 'TPSA',
                 'NumAromaticRings', 'NumHDonors', 'NumHAcceptors',
                 'NumRotatableBonds', 'BasicNitrogens', 'FractionCSP3', 'HeavyAtomCount']
        importances = [morgan_imp, maccs_imp] + phys_imp.tolist()

        return {'features': names, 'importances': importances}


if __name__ == '__main__':
    predictor = HERGPredictor()
    test_smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    train_smiles = test_smiles * 20
    train_labels = [0, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Hybrid FP+MACCS Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
