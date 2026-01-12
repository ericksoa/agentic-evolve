"""
Solution: Lightweight Hybrid FP + Descriptors Predictor
Approach: Morgan fingerprints + only the TOP 10 most predictive descriptors

Key insight: Focus on descriptors with known hERG relevance instead of
using all 200+ RDKit descriptors. Less overfitting, faster training.

Top hERG-relevant features:
1. MolLogP - lipophilicity (primary driver)
2. BasicNitrogenCount - positive charge center
3. NumAromaticRings - pi-stacking capability
4. TPSA - permeability correlate
5. MolWt - size constraint for channel binding
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    Lightweight hybrid predictor: Morgan FP + top-10 descriptors.

    More focused feature set to reduce noise and overfitting.
    Uses GBM for better handling of mixed feature types.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_bits = 2048
        self.fp_radius = 2
        self.scaler = StandardScaler()

        # GBM handles mixed features (binary FP + continuous desc) well
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan FP + top-10 descriptors."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = np.zeros(self.fp_bits + 10)
            else:
                # Morgan fingerprints
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
                morgan_array = np.array(morgan)

                # Top-10 hERG-relevant descriptors (curated from literature)
                desc = np.array([
                    Descriptors.MolLogP(mol),                # 1. Lipophilicity (#1 driver)
                    self._count_basic_nitrogens(mol),        # 2. Basic N (pharmacophore)
                    Descriptors.NumAromaticRings(mol),       # 3. Aromatic content
                    rdMolDescriptors.CalcTPSA(mol),          # 4. Polar surface area
                    Descriptors.MolWt(mol),                  # 5. Molecular size
                    Descriptors.NumRotatableBonds(mol),      # 6. Flexibility
                    Lipinski.NumHAcceptors(mol),             # 7. H-bond acceptors
                    rdMolDescriptors.CalcFractionCSP3(mol),  # 8. 3D character
                    self._count_total_nitrogens(mol),        # 9. Total N atoms
                    Descriptors.HeavyAtomCount(mol),         # 10. Heavy atom count
                ], dtype=float)

                fp = np.concatenate([morgan_array, desc])

            features.append(fp)

        return np.array(features)

    def _count_basic_nitrogens(self, mol):
        """Count basic (sp3) nitrogens - key hERG pharmacophore."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                hyb = atom.GetHybridization().name
                if hyb == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def _count_total_nitrogens(self, mol):
        """Count all nitrogen atoms."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

    def fit(self, X_smiles, y):
        """Train on SMILES and labels."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale only descriptor features (last 10)
        X_fp = X[:, :self.fp_bits]
        X_desc = self.scaler.fit_transform(X[:, self.fp_bits:])
        X = np.concatenate([X_fp, X_desc], axis=1)

        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_

        return self

    def predict_proba(self, X_smiles):
        """Predict hERG blocking probability."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_fp = X[:, :self.fp_bits]
        X_desc = self.scaler.transform(X[:, self.fp_bits:])
        X = np.concatenate([X_fp, X_desc], axis=1)

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary labels."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance with descriptor names."""
        if self._feature_importances is None:
            return None

        fp_importance = np.sum(self._feature_importances[:self.fp_bits])
        desc_importance = self._feature_importances[self.fp_bits:]

        desc_names = [
            'MolLogP', 'BasicNitrogens', 'NumAromaticRings', 'TPSA',
            'MolWt', 'NumRotatableBonds', 'NumHAcceptors',
            'FractionCSP3', 'TotalNitrogens', 'HeavyAtomCount'
        ]

        return {
            'features': ['MorganFingerprints'] + desc_names,
            'importances': [fp_importance] + desc_importance.tolist()
        }


if __name__ == '__main__':
    predictor = HERGPredictor()
    test_smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    train_smiles = test_smiles * 20
    train_labels = [0, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Lightweight Hybrid FP+Desc Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
