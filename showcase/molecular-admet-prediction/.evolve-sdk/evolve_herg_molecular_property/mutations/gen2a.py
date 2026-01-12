"""
XGBoost with Morgan Fingerprints + Physicochemical Descriptors

Mutation from gen0_b: Adds Morgan fingerprints (2048 bits) concatenated with
the hERG-specific descriptors. The config identifies this hybrid approach as
the most effective strategy.

Hypothesis: Morgan fingerprints capture structural patterns that descriptors miss,
while descriptors provide interpretable physicochemical properties. Combined
features should outperform either alone.
"""

import numpy as np
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    hERG toxicity predictor using XGBoost with hybrid features:
    - Morgan fingerprints (2048 bits) for structural patterns
    - hERG-specific physicochemical descriptors (25 features)

    Total: 2073 features
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=200,  # Increased for higher-dimensional features
                max_depth=5,       # Reduced to prevent overfitting
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.6,  # Reduced for many features
                reg_alpha=0.1,     # L1 regularization
                reg_lambda=0.2,    # L2 regularization
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=random_state
            )

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_hybrid_features(self, smiles_list):
        """Calculate Morgan fingerprints + hERG-specific descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use zeros
                features = np.zeros(2048 + 25)
            else:
                # 1. Morgan fingerprints (2048 bits)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                morgan_bits = np.array(morgan_fp)

                # 2. hERG-specific descriptors (25 features)
                mol_h = Chem.AddHs(mol)

                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # Basic nitrogen count (key hERG predictor)
                basic_nitrogens = 0
                for atom in mol_h.GetAtoms():
                    if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0:
                        basic_nitrogens += 1

                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                descriptors = [
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # Lipophilic efficiency
                    mw,
                    heavy_atoms,
                    tpsa,
                    Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    aromatic_atoms,
                    aromatic_rings,
                    aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens,
                    basic_nitrogens * logP,  # Lipophilic basicity
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                features = np.concatenate([morgan_bits, descriptors])

            all_features.append(features)

        # Feature names
        morgan_names = [f'Morgan_{i}' for i in range(2048)]
        desc_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]
        self._feature_names = morgan_names + desc_names

        return np.array(all_features)

    def _impute_and_scale(self, X, fit=False):
        """Handle missing values and scale only descriptor features."""
        X = np.array(X, dtype=float)

        # Impute NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        # Only scale descriptor features (last 25), not fingerprint bits
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])

        return X

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings and labels."""
        X = self._calculate_hybrid_features(X_smiles)
        X = self._impute_and_scale(X, fit=True)
        y = np.array(y)

        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = np.array([class_weights[int(label)] for label in y])

        if HAS_XGBOOST:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_hybrid_features(X_smiles)
        X = self._impute_and_scale(X, fit=False)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance with descriptor names."""
        if self._feature_importances is None or self._feature_names is None:
            return None

        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }


if __name__ == '__main__':
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'
    ]

    train_smiles = test_smiles * 15
    train_labels = [0, 1, 0, 1] * 15

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("XGBoost + Morgan + Descriptors Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
