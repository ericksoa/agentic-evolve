"""
Gen-Pharma-3D: gen12c + Selective 3D Pharmacophore Features

Architecture:
- Base: gen12c (4-model ensemble, 0.890 ROC-AUC)
- Enhancement: 9 key pharmacophore-related 3D features (not shape descriptors)
- Total features: 704 (2D) + 9 (3D pharmacophore) = 713 features

Hypothesis: Shape descriptors (PMI, asphericity) may add noise. The key 3D
features for hERG are pharmacophore distances which capture the spatial
relationship between basic nitrogens and aromatic rings - the hallmark of
hERG blockers.

Selected 3D features:
- N-aromatic distances (min/max/mean) - critical for hERG binding
- Aromatic-aromatic distances (min/max/mean) - ring stacking geometry
- Molecular extent - overall size
- num_basic_N, num_aromatic_rings - pharmacophore counts

Parent: gen12c.py (0.890 ROC-AUC)
"""

import numpy as np
from pathlib import Path
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


# Key pharmacophore features for hERG (indices in the 20-feature 3D vector)
# Full 3D features order: PMI1,PMI2,PMI3,NPR1,NPR2,Asphericity,Eccentricity,
# InertialShapeFactor,RadiusOfGyration,SpherocityIndex,PBF,
# num_basic_N,num_aromatic_rings,min_N_aromatic_dist,max_N_aromatic_dist,
# mean_N_aromatic_dist,min_aromatic_dist,max_aromatic_dist,mean_aromatic_dist,
# molecular_extent
PHARMA_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19]  # Last 9 features


class HERGPredictor:
    """
    Enhanced 4-model ensemble with selective 3D pharmacophore features.

    Uses only the pharmacophore-related 3D features (distances between
    key functional groups) rather than all shape descriptors.
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

        # 3D pharmacophore features (selective)
        self._3d_pharma_names = [
            'num_basic_N', 'num_aromatic_rings',
            'min_N_aromatic_dist', 'max_N_aromatic_dist', 'mean_N_aromatic_dist',
            'min_aromatic_dist', 'max_aromatic_dist', 'mean_aromatic_dist',
            'molecular_extent'
        ]

        # 3D feature cache
        self._3d_cache = {}
        self._load_cached_3d_features()

    def _load_cached_3d_features(self):
        """Load pre-computed 3D features and extract pharmacophore subset."""
        cache_dir = Path(__file__).parent / "data" / "embeddings"

        for split in ['train', 'valid', 'test']:
            cache_file = cache_dir / f"{split}_embeddings.npz"
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                embeddings = data['embeddings']
                smiles = data['smiles']

                for smi, emb in zip(smiles, embeddings):
                    # Extract just pharmacophore features (last 9 of the 20 3D features)
                    full_3d = emb[-20:]
                    self._3d_cache[smi] = full_3d[PHARMA_INDICES]

    def _get_3d_features(self, smiles):
        """Get 3D pharmacophore features from cache or compute if missing."""
        if smiles in self._3d_cache:
            return self._3d_cache[smiles]

        # Fallback: compute on the fly
        try:
            from conformer_gen import get_all_3d_features, get_3d_feature_names
            features_dict = get_all_3d_features(smiles, random_seed=self.random_state)
            all_names = get_3d_feature_names()
            features = [features_dict[all_names[i]] for i in PHARMA_INDICES]
            return np.array(features)
        except Exception:
            return np.zeros(9)

    def _calculate_features(self, smiles_list):
        """Calculate 2D + selective 3D pharmacophore features."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + 25 + 9)
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

                # ========== 3D Pharmacophore Features (selective) ==========
                features_3d = self._get_3d_features(smi)

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
            self._3d_pharma_names
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        # Scale continuous features (last 25 2D + 9 3D = 34 features)
        n_continuous = 34
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
    print("Testing Gen-Pharma-3D model...")

    predictor = HERGPredictor()
    print(f"Loaded {len(predictor._3d_cache)} cached 3D features")
    print(f"Using {len(predictor._3d_pharma_names)} pharmacophore features")

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
