"""
Gen9a: Four-Model Ensemble with LightGBM

Parent: gen7b (0.882 ROC-AUC)

Mutation: Add LightGBM as 4th model
- RF + XGB + ET + LightGBM
- Equal weights [0.25, 0.25, 0.25, 0.25]

Hypothesis: Adding a 4th diverse model may provide
additional variance reduction through ensemble diversity.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
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
    """
    Four-model ensemble for hERG toxicity.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # RF
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

        # XGBoost
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
            self.xgb = None

        # ExtraTrees
        self.et = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # LightGBM
        if HAS_LIGHTGBM:
            self.lgb = lgb.LGBMClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.65,
                colsample_bytree=0.5,
                reg_alpha=0.4,
                reg_lambda=0.5,
                random_state=random_state,
                verbose=-1,
                n_jobs=1
            )
        else:
            self.lgb = None

        # Determine weights based on available models
        self.models = [self.rf, self.xgb, self.et, self.lgb]
        self.available_models = [m for m in self.models if m is not None]
        n_models = len(self.available_models)
        self.weights = [1.0 / n_models] * n_models

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate extended dual fingerprints + hERG descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(1024 + 167 + 25)
            else:
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

        morgan_names = [f'Morgan3_{i}' for i in range(1024)]
        maccs_names = [f'MACCS_{i}' for i in range(167)]
        desc_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]
        self._feature_names = morgan_names + maccs_names + desc_names

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

        if self.xgb is not None:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)

        if self.lgb is not None:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.lgb.fit(X, y, sample_weight=sample_weights)

        # Compute feature importances from available models
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

        probas = []
        probas.append(self.rf.predict_proba(X)[:, 1])
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
    print("Gen9a: Four-Model Ensemble predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
