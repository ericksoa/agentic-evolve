"""
Gen-3D-Descriptors: Enhanced gen12c with 3D Conformer Features

Architecture:
- Base: gen12c (4-model ensemble, 0.890 ROC-AUC)
- Enhancement: +20 3D conformer descriptors
- Total features: 704 (2D) + 20 (3D) = 724 features

This version avoids slow transformer embeddings by using only
fast-to-compute 3D descriptors from pre-generated conformers.

Hypothesis: 3D shape and pharmacophore distance features capture
spatial information about hERG binding that 2D fingerprints miss.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys

# Import 3D feature extraction
from conformer_gen import get_all_3d_features, get_3d_feature_names


class HERGPredictor:
    """
    Enhanced 4-model ensemble with 3D conformer descriptors.

    Combines gen12c's proven 2D approach with 3D spatial features:
    - Shape descriptors (PMI, asphericity, spherocity)
    - Pharmacophore distances (N-aromatic, aromatic-aromatic)
    - Molecular extent and 3D size metrics
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Gen12c-style ensemble
        self.rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.65,
                colsample_bytree=0.5,
                reg_alpha=0.4,
                reg_lambda=0.5,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03,
                random_state=random_state
            )

        self.et = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # Ensemble weights (gen12c style)
        self.weights = [0.28, 0.28, 0.28, 0.16]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

        # 3D feature caching for faster inference
        self._3d_cache = {}

    def _calculate_features(self, smiles_list):
        """Calculate 2D + 3D features."""
        all_features = []
        n_2d = 512 + 167 + 25  # Morgan + MACCS + 2D descriptors
        n_3d = len(get_3d_feature_names())

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(n_2d + n_3d)
            else:
                # ========== 2D Features (from gen12c) ==========
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                descriptors_2d = [
                    logP, Descriptors.MolMR(mol), logP * mw / 100, mw, heavy_atoms,
                    tpsa, Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                    aromatic_atoms, aromatic_rings, aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens, basic_nitrogens * logP, Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol), Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),
                    Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                # ========== 3D Features ==========
                if smi in self._3d_cache:
                    features_3d = self._3d_cache[smi]
                else:
                    features_3d_dict = get_all_3d_features(smi, random_seed=self.random_state)
                    features_3d = [features_3d_dict[k] for k in get_3d_feature_names()]
                    self._3d_cache[smi] = features_3d

                features = np.concatenate([morgan_bits, maccs_bits, descriptors_2d, features_3d])

            all_features.append(features)

        # Build feature names
        self._feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'] +
            get_3d_feature_names()
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        # Scale all continuous features (last 25 2D + 20 3D = 45 features)
        n_continuous = 45
        if fit:
            X[:, -n_continuous:] = self.scaler.fit_transform(X[:, -n_continuous:])
        else:
            X[:, -n_continuous:] = self.scaler.transform(X[:, -n_continuous:])
        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Feature importances (tree models only)
        tree_weight_sum = self.weights[0] + self.weights[1] + self.weights[2]
        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        ) / tree_weight_sum

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]

        return (self.weights[0] * proba_rf +
                self.weights[1] * proba_xgb +
                self.weights[2] * proba_et +
                self.weights[3] * proba_svm)

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }


if __name__ == '__main__':
    print("Testing Gen-3D-Descriptors model...")

    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)NCC(O)C1=CC=C(O)C(O)=C1',
    ]
    test_labels = [0, 0, 1]

    print("\nFitting on test data...")
    predictor.fit(test_smiles * 10, test_labels * 10)

    print("Predicting...")
    proba = predictor.predict_proba(test_smiles)
    print(f"Probabilities: {proba}")

    print(f"\nTotal features: {len(predictor._feature_names)}")
    print("Top 10 feature importances:")
    importance = predictor.get_feature_importance()
    for f, i in zip(importance['features'][:10], importance['importances'][:10]):
        print(f"  {f}: {i:.4f}")
