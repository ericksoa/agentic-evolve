"""
Gen17c: 6-Model Diverse Ensemble

Parent: gen12c (0.890 ROC-AUC)

Mutation: Add model diversity through parameter variation
- 6 models: RF1 + XGB1 + ET + SVM + RF2 + XGB2
- RF2: max_depth=5, n_estimators=100, max_features='log2'
- XGB2: max_depth=4, learning_rate=0.05
- Weights: [0.18, 0.18, 0.18, 0.18, 0.15, 0.13]

Hypothesis: More model diversity through parameter variation
reduces variance and improves generalization.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """6-model ensemble with diverse parameter configurations for hERG toxicity."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Original RF (from gen12c)
        self.rf1 = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Second RF with different hyperparameters
        self.rf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='log2',
            class_weight='balanced',
            random_state=random_state + 1,
            n_jobs=1
        )

        if HAS_XGBOOST:
            # Original XGBoost (from gen12c)
            self.xgb1 = xgb.XGBClassifier(
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
            # Second XGBoost with different hyperparameters
            self.xgb2 = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=0.3,
                reg_lambda=0.4,
                random_state=random_state + 1,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb1 = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03,
                random_state=random_state
            )
            self.xgb2 = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=random_state + 1
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

        # SVM with RBF kernel (from gen12c)
        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # 6-model weights: RF1, XGB1, ET, SVM, RF2, XGB2
        self.weights = [0.18, 0.18, 0.18, 0.18, 0.15, 0.13]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate compact fingerprints + hERG descriptors."""
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

        self._feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex']
        )

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

        # Fit all tree-based models
        self.rf1.fit(X, y)
        self.rf2.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)

        # Fit XGBoost models with sample weights
        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb1.fit(X, y, sample_weight=sample_weights)
            self.xgb2.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb1.fit(X, y)
            self.xgb2.fit(X, y)

        # Feature importances from tree models (SVM doesn't have feature_importances_)
        tree_weights = [self.weights[0], self.weights[1], self.weights[2], self.weights[4], self.weights[5]]
        total_tree_weight = sum(tree_weights)
        self._feature_importances = (
            self.weights[0] * self.rf1.feature_importances_ +
            self.weights[1] * self.xgb1.feature_importances_ +
            self.weights[2] * self.et.feature_importances_ +
            self.weights[4] * self.rf2.feature_importances_ +
            self.weights[5] * self.xgb2.feature_importances_
        ) / total_tree_weight

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf1 = self.rf1.predict_proba(X)[:, 1]
        proba_xgb1 = self.xgb1.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]
        proba_rf2 = self.rf2.predict_proba(X)[:, 1]
        proba_xgb2 = self.xgb2.predict_proba(X)[:, 1]

        return (self.weights[0] * proba_rf1 +
                self.weights[1] * proba_xgb1 +
                self.weights[2] * proba_et +
                self.weights[3] * proba_svm +
                self.weights[4] * proba_rf2 +
                self.weights[5] * proba_xgb2)

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
