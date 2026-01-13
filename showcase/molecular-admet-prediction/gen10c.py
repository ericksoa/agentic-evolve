"""
Gen10c: Compact FP + 4-Model Ensemble

Crossover: gen9b (compact FP) x gen9a (4-model)
- 512-bit Morgan fingerprints
- RF + XGB + ET + LightGBM ensemble

Hypothesis: Combining compact fingerprints with additional
ensemble diversity may yield further improvements.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state

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
            self.xgb = None

        self.et = ExtraTreesClassifier(
            n_estimators=80, max_depth=5, min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=random_state, n_jobs=1
        )

        if HAS_LIGHTGBM:
            self.lgb = lgb.LGBMClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.03, subsample=0.65,
                colsample_bytree=0.5, reg_alpha=0.4, reg_lambda=0.5,
                random_state=random_state, verbose=-1, n_jobs=1
            )
        else:
            self.lgb = None

        self.models = [self.rf, self.xgb, self.et, self.lgb]
        self.available_models = [m for m in self.models if m is not None]
        n_models = len(self.available_models)
        self.weights = [1.0 / n_models] * n_models

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        all_features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + 25)
            else:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms() if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
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
        self._feature_names = [f'Morgan3_{i}' for i in range(512)] + [f'MACCS_{i}' for i in range(167)] + [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]
        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])
        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = np.array([class_weights[int(label)] for label in y])
        if self.xgb is not None:
            self.xgb.fit(X, y, sample_weight=sample_weights)
        if self.lgb is not None:
            self.lgb.fit(X, y, sample_weight=sample_weights)
        importances = []
        for model in self.available_models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        if importances:
            self._feature_importances = np.mean(importances, axis=0)
        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)
        probas = [self.rf.predict_proba(X)[:, 1]]
        if self.xgb is not None:
            probas.append(self.xgb.predict_proba(X)[:, 1])
        probas.append(self.et.predict_proba(X)[:, 1])
        if self.lgb is not None:
            probas.append(self.lgb.predict_proba(X)[:, 1])
        return sum(w * p for w, p in zip(self.weights, probas))

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {'features': [x[0] for x in sorted_importance[:30]], 'importances': [x[1] for x in sorted_importance[:30]]}
