"""
Gen29: MapLight-Inspired Approach

Parent: gen12c (0.865 TDC AUROC)

Inspired by MapLight+GNN (#1 on TDC at 0.880):
- CatBoost gradient boosting (their primary model)
- Avalon fingerprints (in addition to Morgan/MACCS)
- Extended physicochemical descriptors (~100 vs our 25)
- Keep our existing ensemble for blending

Hypothesis: MapLight's success may come from:
1. CatBoost's handling of categorical features
2. Avalon fingerprints capturing different patterns
3. More comprehensive descriptor set
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys
from rdkit.Avalon import pyAvalonTools
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_extended_descriptors(mol):
    """Calculate ~100 physicochemical descriptors (inspired by MapLight's 200)."""
    if mol is None:
        return np.zeros(100)

    mol_h = Chem.AddHs(mol)

    # Basic properties
    logP = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    heavy_atoms = Descriptors.HeavyAtomCount(mol)

    # Counts
    basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                         if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    descriptors = [
        # Basic (10)
        logP, mw, tpsa, heavy_atoms, Descriptors.MolMR(mol),
        Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.RingCount(mol),

        # Aromaticity (10)
        aromatic_atoms, Descriptors.NumAromaticRings(mol),
        Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
        aromatic_atoms / max(1, heavy_atoms), Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAliphaticCarbocycles(mol), Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumSaturatedRings(mol), Descriptors.NumSaturatedCarbocycles(mol),

        # Nitrogen/hERG-specific (10)
        basic_nitrogens, basic_nitrogens * logP, Descriptors.NumHeteroatoms(mol),
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]),  # Total N
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]),  # Total O
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]), # Total S
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 9]),  # Total F
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 17]), # Total Cl
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 35]), # Total Br
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]]),  # Halogens

        # Complexity (10)
        Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
        Descriptors.Chi0v(mol), Descriptors.Chi1v(mol), Descriptors.Chi2v(mol),
        Descriptors.Chi3v(mol), Descriptors.Chi4v(mol),
        Descriptors.Kappa1(mol), Descriptors.Kappa2(mol),

        # Surface area (10)
        Descriptors.LabuteASA(mol), Descriptors.PEOE_VSA1(mol), Descriptors.PEOE_VSA2(mol),
        Descriptors.PEOE_VSA3(mol), Descriptors.PEOE_VSA4(mol), Descriptors.PEOE_VSA5(mol),
        Descriptors.PEOE_VSA6(mol), Descriptors.PEOE_VSA7(mol), Descriptors.PEOE_VSA8(mol),
        Descriptors.PEOE_VSA9(mol),

        # More VSA (10)
        Descriptors.SMR_VSA1(mol), Descriptors.SMR_VSA2(mol), Descriptors.SMR_VSA3(mol),
        Descriptors.SMR_VSA4(mol), Descriptors.SMR_VSA5(mol), Descriptors.SMR_VSA6(mol),
        Descriptors.SMR_VSA7(mol), Descriptors.SMR_VSA8(mol), Descriptors.SMR_VSA9(mol),
        Descriptors.SMR_VSA10(mol),

        # SLogP VSA (10)
        Descriptors.SlogP_VSA1(mol), Descriptors.SlogP_VSA2(mol), Descriptors.SlogP_VSA3(mol),
        Descriptors.SlogP_VSA4(mol), Descriptors.SlogP_VSA5(mol), Descriptors.SlogP_VSA6(mol),
        Descriptors.SlogP_VSA7(mol), Descriptors.SlogP_VSA8(mol), Descriptors.SlogP_VSA9(mol),
        Descriptors.SlogP_VSA10(mol),

        # Estate (10)
        Descriptors.EState_VSA1(mol), Descriptors.EState_VSA2(mol), Descriptors.EState_VSA3(mol),
        Descriptors.EState_VSA4(mol), Descriptors.EState_VSA5(mol), Descriptors.EState_VSA6(mol),
        Descriptors.EState_VSA7(mol), Descriptors.EState_VSA8(mol), Descriptors.EState_VSA9(mol),
        Descriptors.EState_VSA10(mol),

        # Additional (20)
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol), rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol), Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumValenceElectrons(mol), Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol), Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol), Descriptors.HallKierAlpha(mol),
        mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),  # Rigidity index
        logP * mw / 100,  # Lipophilic efficiency
        tpsa / mw * 100,  # TPSA density
        logP / max(1, heavy_atoms),  # logP per atom
        Descriptors.TPSA(mol) / max(1, heavy_atoms),  # TPSA per atom
        Descriptors.MolWt(mol) / max(1, Lipinski.RingCount(mol) + 1),  # MW per ring
        logP - 0.026 * mw,  # Druglikeness proxy
        basic_nitrogens / max(1, heavy_atoms) * 100,  # Basic N density
        aromatic_atoms / max(1, Lipinski.RingCount(mol) + 1),  # Aromatics per ring
        Descriptors.Kappa3(mol),
    ]

    return np.array(descriptors, dtype=np.float32)


class HERGPredictor:
    """MapLight-inspired ensemble with CatBoost and extended features."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Original ensemble
        self.rf = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_split=10, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced', random_state=random_state, n_jobs=1
        )

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03, subsample=0.65,
                colsample_bytree=0.5, reg_alpha=0.4, reg_lambda=0.5,
                random_state=random_state, eval_metric='logloss', n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03, random_state=random_state
            )

        self.et = ExtraTreesClassifier(
            n_estimators=80, max_depth=5, min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=random_state, n_jobs=1
        )

        self.svm = SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
            probability=True, random_state=random_state
        )

        # New: CatBoost (MapLight's primary model)
        if HAS_CATBOOST:
            self.catboost = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=random_state,
                auto_class_weights='Balanced', verbose=False
            )
        else:
            self.catboost = None

        # Ensemble weights: include CatBoost if available
        if HAS_CATBOOST:
            # Give CatBoost higher weight (it's MapLight's primary)
            self.weights = [0.20, 0.20, 0.20, 0.10, 0.30]  # RF, XGB, ET, SVM, CatBoost
        else:
            self.weights = [0.28, 0.28, 0.28, 0.16]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan + Avalon + MACCS + extended descriptors."""
        all_features = []
        n_desc = 100

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 512 + 167 + n_desc)
            else:
                # Morgan fingerprints (512 bits)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                # Avalon fingerprints (512 bits) - MapLight uses this
                avalon_fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
                avalon_bits = np.array(avalon_fp)

                # MACCS keys (167 bits)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # Extended descriptors (~100)
                descriptors = get_extended_descriptors(mol)

                features = np.concatenate([morgan_bits, avalon_bits, maccs_bits, descriptors])

            all_features.append(features)

        self._feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'Avalon_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            [f'Desc_{i}' for i in range(n_desc)]
        )

        return np.array(all_features, dtype=np.float32)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_desc = 100
        if fit:
            X[:, -n_desc:] = self.scaler.fit_transform(X[:, -n_desc:])
        else:
            X[:, -n_desc:] = self.scaler.transform(X[:, -n_desc:])
        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Fit all models
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        if self.catboost is not None:
            self.catboost.fit(X, y)

        # Compute feature importances
        tree_weights = self.weights[:3]
        self._feature_importances = (
            tree_weights[0] * self.rf.feature_importances_ +
            tree_weights[1] * self.xgb.feature_importances_ +
            tree_weights[2] * self.et.feature_importances_
        ) / sum(tree_weights)

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]

        if self.catboost is not None:
            proba_cb = self.catboost.predict_proba(X)[:, 1]
            return (self.weights[0] * proba_rf +
                    self.weights[1] * proba_xgb +
                    self.weights[2] * proba_et +
                    self.weights[3] * proba_svm +
                    self.weights[4] * proba_cb)
        else:
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
