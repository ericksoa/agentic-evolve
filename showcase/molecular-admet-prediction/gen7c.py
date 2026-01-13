"""
Gen7c: Topological Descriptors Enhanced Ensemble

Parent: gen6b (0.881 ROC-AUC)

Mutation: Add topological/graph-based molecular descriptors
- Kappa shape indices (molecular shape/branching)
- LabuteASA (accessible surface area)
- Hall-Kier alpha, Chi connectivity indices
- Balaban J index (topological complexity)

Hypothesis: hERG binding depends on molecular shape and topology.
These 3D-independent descriptors may capture structural features
that influence ion channel interactions.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys, GraphDescriptors


class HERGPredictor:
    """
    Topologically-enhanced triple ensemble for hERG toxicity.
    Adds shape and graph-based descriptors.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # RF - extreme regularization (same as gen6b)
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

        # XGBoost - extreme regularization (same as gen6b)
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
                n_estimators=80,
                max_depth=3,
                learning_rate=0.03,
                random_state=random_state
            )

        # ExtraTrees - extreme regularization (same as gen6b)
        self.et = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # RF-dominant weights (same as gen6b)
        self.weights = [0.42, 0.33, 0.25]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate dual fingerprints + enhanced molecular descriptors with topology."""
        all_features = []
        n_descriptors = 35  # Extended from 25 to 35

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(1024 + 167 + n_descriptors)
            else:
                # Fingerprints (same as gen6b)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
                morgan_bits = np.array(morgan_fp)

                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                basic_nitrogens = 0
                for atom in mol_h.GetAtoms():
                    if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0:
                        basic_nitrogens += 1

                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                # Original 25 descriptors
                base_descriptors = [
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

                # NEW: Topological descriptors (10 additional)
                try:
                    kappa1 = GraphDescriptors.Kappa1(mol)
                    kappa2 = GraphDescriptors.Kappa2(mol)
                    kappa3 = GraphDescriptors.Kappa3(mol)
                except:
                    kappa1 = kappa2 = kappa3 = 0.0

                try:
                    balaban_j = GraphDescriptors.BalabanJ(mol)
                except:
                    balaban_j = 0.0

                try:
                    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
                except:
                    labute_asa = 0.0

                try:
                    hall_kier_alpha = Descriptors.HallKierAlpha(mol)
                except:
                    hall_kier_alpha = 0.0

                chi2 = Descriptors.Chi2n(mol)
                chi3 = Descriptors.Chi3n(mol)
                chi4 = Descriptors.Chi4n(mol)

                # Ipc (information content)
                try:
                    ipc = Descriptors.Ipc(mol)
                except:
                    ipc = 0.0

                topo_descriptors = [
                    kappa1, kappa2, kappa3,  # Shape indices
                    balaban_j,               # Topological index
                    labute_asa,              # Surface area
                    hall_kier_alpha,         # Hall-Kier alpha
                    chi2, chi3, chi4,        # Higher-order connectivity
                    ipc,                     # Information content
                ]

                descriptors = base_descriptors + topo_descriptors
                features = np.concatenate([morgan_bits, maccs_bits, descriptors])
            all_features.append(features)

        morgan_names = [f'Morgan3_{i}' for i in range(1024)]
        maccs_names = [f'MACCS_{i}' for i in range(167)]
        desc_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex',
            # New topological descriptors
            'Kappa1', 'Kappa2', 'Kappa3', 'BalabanJ', 'LabuteASA',
            'HallKierAlpha', 'Chi2n', 'Chi3n', 'Chi4n', 'Ipc'
        ]
        self._feature_names = morgan_names + maccs_names + desc_names

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        n_desc = 35  # Extended descriptor count
        if fit:
            X[:, -n_desc:] = self.scaler.fit_transform(X[:, -n_desc:])
        else:
            X[:, -n_desc:] = self.scaler.transform(X[:, -n_desc:])
        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        self.rf.fit(X, y)
        self.et.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        )
        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]

        return self.weights[0] * proba_rf + self.weights[1] * proba_xgb + self.weights[2] * proba_et

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
    predictor = HERGPredictor()
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'
    ]
    train_smiles = test_smiles * 15
    train_labels = [0, 1, 0, 1] * 15
    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)
    print("Gen7c: Topological Descriptors Enhanced predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
