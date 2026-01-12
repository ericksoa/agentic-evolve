"""
Evolved Solution: Hybrid Ensemble Predictor for hERG Toxicity

This solution combines the strengths of multiple approaches:
1. Morgan fingerprints (structural features)
2. Molecular descriptors (interpretable physicochemical features)
3. Ensemble of 3 classifiers with optimized weights

Key innovations:
- Concatenated feature representation (fingerprints + descriptors)
- Voting ensemble with probability averaging
- hERG-specific features (basic nitrogens, aromatic content, lipophilicity)
- Robust to invalid SMILES with graceful fallbacks
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    Ensemble hERG toxicity predictor combining fingerprints and descriptors.

    Architecture:
    - Feature extraction: Morgan FP (2048) + Molecular descriptors (25)
    - Ensemble: RF + GBM + ExtraTrees with weighted averaging
    - Optimized for hERG-specific molecular features
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_radius = 2
        self.fp_bits = 2048
        self.scaler = StandardScaler()

        # Ensemble of 3 diverse classifiers
        self.models = [
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=random_state,
                class_weight='balanced'
            ),
            GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=random_state
            ),
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                n_jobs=1,
                random_state=random_state,
                class_weight='balanced'
            )
        ]

        # Model weights (learned from CV performance)
        self.weights = [0.35, 0.35, 0.30]
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate combined fingerprint + descriptor features."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zero vector
                fp = np.zeros(self.fp_bits + 25)
            else:
                # Morgan fingerprints
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
                fp_array = np.array(morgan)

                # Molecular descriptors
                desc = self._calculate_descriptors(mol)
                fp = np.concatenate([fp_array, desc])

            features.append(fp)

        return np.array(features)

    def _calculate_descriptors(self, mol):
        """Calculate hERG-relevant molecular descriptors."""
        desc = [
            # Lipophilicity - key for hERG binding
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),

            # Size
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomCount(mol),

            # Flexibility
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),

            # Polar surface
            rdMolDescriptors.CalcTPSA(mol),

            # Aromatic features - critical for hERG
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),

            # Ring systems
            Lipinski.RingCount(mol),
            Descriptors.NumAliphaticRings(mol),

            # H-bonding
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),

            # Nitrogen features - basic N key for hERG
            self._count_basic_nitrogens(mol),
            self._count_nitrogens(mol),

            # Heteroatoms
            Descriptors.NumHeteroatoms(mol),

            # Topological
            self._safe_balaban_j(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),

            # Charge
            Descriptors.MaxAbsPartialCharge(mol) if self._safe_charge(mol) else 0,
            Descriptors.MinAbsPartialCharge(mol) if self._safe_charge(mol) else 0,

            # Shape
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),

            # Surface area
            Descriptors.LabuteASA(mol),
        ]
        return np.array(desc, dtype=float)

    def _count_basic_nitrogens(self, mol):
        """Count basic nitrogens (key pharmacophore for hERG)."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                # Basic if sp3 and not in amide/nitro
                if atom.GetHybridization().name == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def _count_nitrogens(self, mol):
        """Count total nitrogens."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

    def _safe_balaban_j(self, mol):
        """Safely calculate Balaban J index."""
        try:
            j = Descriptors.BalabanJ(mol)
            return j if j != 0 and not np.isinf(j) else 0
        except:
            return 0

    def _safe_charge(self, mol):
        """Check if partial charges can be computed."""
        try:
            Descriptors.MaxAbsPartialCharge(mol)
            return True
        except:
            return False

    def fit(self, X_smiles, y):
        """Train the ensemble on SMILES strings and labels."""
        X = self._calculate_features(X_smiles)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale only descriptor part (last 25 features)
        X_fp = X[:, :self.fp_bits]
        X_desc = X[:, self.fp_bits:]
        X_desc = self.scaler.fit_transform(X_desc)
        X = np.concatenate([X_fp, X_desc], axis=1)

        y = np.array(y)

        # Train all models
        for model in self.models:
            model.fit(X, y)

        # Store averaged feature importances
        self._feature_importances = np.mean([m.feature_importances_ for m in self.models], axis=0)

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using weighted ensemble."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale descriptors
        X_fp = X[:, :self.fp_bits]
        X_desc = X[:, self.fp_bits:]
        X_desc = self.scaler.transform(X_desc)
        X = np.concatenate([X_fp, X_desc], axis=1)

        # Weighted probability averaging
        probas = []
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)[:, 1]
            probas.append(proba * weight)

        return np.sum(probas, axis=0)

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get top descriptor importances."""
        if self._feature_importances is None:
            return None

        # Focus on descriptor importances (more interpretable)
        desc_importances = self._feature_importances[self.fp_bits:]
        desc_names = [
            'MolLogP', 'MolMR', 'MolWt', 'HeavyAtomCount', 'NumRotatableBonds',
            'FractionCSP3', 'TPSA', 'NumAromaticRings', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'RingCount', 'NumAliphaticRings',
            'NumHDonors', 'NumHAcceptors', 'BasicNitrogens', 'TotalNitrogens',
            'NumHeteroatoms', 'BalabanJ', 'Chi0', 'Chi1',
            'MaxAbsPartialCharge', 'MinAbsPartialCharge',
            'Kappa1', 'Kappa2', 'LabuteASA'
        ]

        importance_dict = dict(zip(desc_names, desc_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance],
            'importances': [x[1] for x in sorted_importance]
        }


if __name__ == '__main__':
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
    ]

    train_smiles = test_smiles * 20
    train_labels = [0, 1, 0] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Evolved Ensemble Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")

    fi = predictor.get_feature_importance()
    print("\nTop features for hERG prediction:")
    for name, imp in zip(fi['features'][:10], fi['importances'][:10]):
        print(f"  {name}: {imp:.4f}")
