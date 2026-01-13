"""
Gen-Pretrained-3D: Hybrid 2D+3D Ensemble with Pre-trained Embeddings

Architecture:
- Branch A: Current champion gen12c (4-model ensemble, 0.890 ROC-AUC)
- Branch B: ChemBERTa embeddings + 3D descriptors + MLP head
- Meta-Ensemble: Weighted combination of A and B

Hypothesis: Pre-trained transformer embeddings capture learned molecular
representations that complement hand-crafted fingerprints. Combined with
3D pharmacophore features, this should improve hERG prediction accuracy.

Parent: gen12c.py (0.890 ROC-AUC)
Target: 0.92+ ROC-AUC
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
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
    Hybrid 2D+3D ensemble for hERG toxicity prediction.

    Combines:
    1. Traditional fingerprints + 2D descriptors (proven gen12c approach)
    2. ChemBERTa transformer embeddings (learned representations)
    3. 3D conformer descriptors (spatial pharmacophore features)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # ========== Branch A: Gen12c-style ensemble ==========
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

        # Gen12c weights
        self.branch_a_weights = [0.28, 0.28, 0.28, 0.16]

        # ========== Branch B: MLP on embeddings ==========
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state
        )

        # ========== Meta-ensemble weights ==========
        # Start with equal weighting, can be evolved
        self.meta_weights = [0.5, 0.5]  # [branch_a, branch_b]

        # Scalers
        self.scaler_2d = RobustScaler()
        self.scaler_embeddings = StandardScaler()

        # Feature info
        self._feature_names = None
        self._feature_importances = None

        # Embedding cache
        self._embedding_cache = {}
        self._embeddings_precomputed = False
        self._use_cached_embeddings = True

    def _calculate_2d_features(self, smiles_list):
        """Calculate traditional 2D features (fingerprints + descriptors)."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + 25)
            else:
                # Morgan fingerprints (compact 512-bit)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                # MACCS keys
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # 25 molecular descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                descriptors = [
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

                features = np.concatenate([morgan_bits, maccs_bits, descriptors])
            all_features.append(features)

        return np.array(all_features)

    def _calculate_3d_features(self, smiles_list):
        """Calculate 3D conformer features."""
        all_features = []
        feature_names = get_3d_feature_names()

        for smi in smiles_list:
            features_dict = get_all_3d_features(smi, random_seed=self.random_state)
            features = [features_dict[k] for k in feature_names]
            all_features.append(features)

        return np.array(all_features)

    def _get_embeddings(self, smiles_list):
        """Get ChemBERTa embeddings (with caching)."""
        # Try to use pre-computed embeddings if available
        if self._use_cached_embeddings and self._embeddings_precomputed:
            embeddings = []
            for smi in smiles_list:
                if smi in self._embedding_cache:
                    embeddings.append(self._embedding_cache[smi])
                else:
                    # Compute on the fly for unknown molecules
                    from embeddings_3d import get_smiles_embedding
                    emb = get_smiles_embedding(smi)
                    self._embedding_cache[smi] = emb
                    embeddings.append(emb)
            return np.array(embeddings)

        # Compute fresh embeddings
        from embeddings_3d import get_batch_embeddings
        return get_batch_embeddings(smiles_list, show_progress=False)

    def _preprocess_2d(self, X, fit=False):
        """Preprocess 2D features."""
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -25:] = self.scaler_2d.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler_2d.transform(X[:, -25:])
        return X

    def _preprocess_embeddings(self, X, fit=False):
        """Preprocess embedding features."""
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            return self.scaler_embeddings.fit_transform(X)
        return self.scaler_embeddings.transform(X)

    def load_cached_embeddings(self, train_smiles, valid_smiles=None, test_smiles=None):
        """
        Load pre-computed embeddings from cache.

        Call this before fit() to use cached embeddings instead of
        computing them on the fly (much faster).
        """
        try:
            from precompute_embeddings import load_cached_embeddings

            # Load train embeddings
            train_emb, _, train_smi = load_cached_embeddings("train")
            for smi, emb in zip(train_smi, train_emb):
                self._embedding_cache[smi] = emb[:384]  # Only transformer part

            # Load valid/test if available
            for split in ["valid", "test"]:
                try:
                    emb, _, smi = load_cached_embeddings(split)
                    for s, e in zip(smi, emb):
                        self._embedding_cache[s] = e[:384]
                except FileNotFoundError:
                    pass

            self._embeddings_precomputed = True
            print(f"Loaded {len(self._embedding_cache)} cached embeddings")

        except Exception as e:
            print(f"Could not load cached embeddings: {e}")
            self._embeddings_precomputed = False

    def fit(self, X_smiles, y):
        """Train the hybrid ensemble."""
        y = np.array(y)

        # ========== Branch A: 2D features ==========
        print("Computing 2D features...")
        X_2d = self._calculate_2d_features(X_smiles)
        X_2d = self._preprocess_2d(X_2d, fit=True)

        print("Training Branch A (2D ensemble)...")
        self.rf.fit(X_2d, y)
        self.et.fit(X_2d, y)
        self.svm.fit(X_2d, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X_2d, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X_2d, y)

        # ========== Branch B: Embeddings + 3D ==========
        print("Computing transformer embeddings...")
        X_emb = self._get_embeddings(X_smiles)

        print("Computing 3D features...")
        X_3d = self._calculate_3d_features(X_smiles)

        # Combine embeddings and 3D features
        X_combined = np.concatenate([X_emb, X_3d], axis=1)
        X_combined = self._preprocess_embeddings(X_combined, fit=True)

        print("Training Branch B (MLP on embeddings+3D)...")
        self.mlp.fit(X_combined, y)

        # Store feature importances (from tree models only)
        tree_weight_sum = sum(self.branch_a_weights[:3])
        self._feature_importances = (
            self.branch_a_weights[0] * self.rf.feature_importances_ +
            self.branch_a_weights[1] * self.xgb.feature_importances_ +
            self.branch_a_weights[2] * self.et.feature_importances_
        ) / tree_weight_sum

        return self

    def predict_proba(self, X_smiles):
        """Predict hERG blocking probability."""
        # Branch A: 2D ensemble
        X_2d = self._calculate_2d_features(X_smiles)
        X_2d = self._preprocess_2d(X_2d, fit=False)

        proba_rf = self.rf.predict_proba(X_2d)[:, 1]
        proba_xgb = self.xgb.predict_proba(X_2d)[:, 1]
        proba_et = self.et.predict_proba(X_2d)[:, 1]
        proba_svm = self.svm.predict_proba(X_2d)[:, 1]

        proba_branch_a = (
            self.branch_a_weights[0] * proba_rf +
            self.branch_a_weights[1] * proba_xgb +
            self.branch_a_weights[2] * proba_et +
            self.branch_a_weights[3] * proba_svm
        )

        # Branch B: Embeddings + 3D
        X_emb = self._get_embeddings(X_smiles)
        X_3d = self._calculate_3d_features(X_smiles)
        X_combined = np.concatenate([X_emb, X_3d], axis=1)
        X_combined = self._preprocess_embeddings(X_combined, fit=False)

        proba_branch_b = self.mlp.predict_proba(X_combined)[:, 1]

        # Meta-ensemble
        return (
            self.meta_weights[0] * proba_branch_a +
            self.meta_weights[1] * proba_branch_b
        )

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary labels."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return feature importance (from 2D branch)."""
        if self._feature_importances is None:
            return None

        feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex']
        )

        importance_dict = dict(zip(feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }


if __name__ == '__main__':
    # Quick test
    print("Testing Gen-Pretrained-3D hybrid model...")

    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)NCC(O)C1=CC=C(O)C(O)=C1',  # Isoproterenol (hERG blocker)
    ]
    test_labels = [0, 0, 1]

    # Quick fit test (not meaningful with 3 samples)
    print("\nFitting on test data...")
    predictor.fit(test_smiles * 10, test_labels * 10)

    print("\nPredicting...")
    proba = predictor.predict_proba(test_smiles)
    print(f"Probabilities: {proba}")

    print("\nModel ready for evolution!")
