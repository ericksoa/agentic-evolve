"""
Gen4x: Advanced Crossover Hybrid with Adaptive Ensemble

Parents:
- gen0_champion (0.89 ROC-AUC): 4-model ensemble with SVM diversity, C=1.0
- gen1a (0.8899 ROC-AUC): Optimized weights emphasizing XGBoost
- gen2x (0.8896 ROC-AUC): Blended weights and softer SVM margins

Crossover Strategy:
1. From gen0_champion: Keep SVM C=1.0 (best fitness had this)
2. From gen1a: XGBoost emphasis (performs well on tabular molecular data)
3. From gen2x: Balanced tree ensemble approach
4. NEW: Add AdaBoost as 5th model for additional ensemble diversity
5. NEW: Slightly increased n_estimators for all models (100 vs 80)
6. NEW: Refined weight distribution [0.22, 0.30, 0.22, 0.14, 0.12]
   - XGBoost highest (from gen1a insight)
   - RF and ET equal (diversity matters more than redundancy)
   - SVM and AdaBoost provide decision boundary diversity

Hypothesis: Adding a 5th model (AdaBoost) with boosting on weak learners
provides orthogonal diversity to the existing ensemble. Combined with
slightly more estimators and refined weights from successful parents,
this should improve generalization.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
    """5-model ensemble with AdaBoost diversity for hERG toxicity prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # From gen0_champion: RF with slightly increased estimators
        self.rf = RandomForestClassifier(
            n_estimators=100,  # Increased from 80
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # From gen0_champion: XGBoost with regularization
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=100,  # Increased from 80
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
                n_estimators=100, max_depth=3, learning_rate=0.03,
                random_state=random_state
            )

        # From gen0_champion: ExtraTrees with increased estimators
        self.et = ExtraTreesClassifier(
            n_estimators=100,  # Increased from 80
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # From gen0_champion: SVM with C=1.0 (best parent used this)
        self.svm = SVC(
            C=1.0,  # Keep from gen0_champion (best fitness)
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # NEW: AdaBoost for additional ensemble diversity
        # Uses weak decision stumps boosted iteratively
        self.ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=80,
            learning_rate=0.05,
            random_state=random_state,
            algorithm='SAMME'
        )

        # CROSSOVER: 5-model weights combining insights from all parents
        # gen0_champion: [0.28, 0.28, 0.28, 0.16] - equal tree weights
        # gen1a:        [0.25, 0.35, 0.22, 0.18] - XGB emphasis
        # gen2x:        [0.26, 0.33, 0.25, 0.16] - blended
        # Hybrid:       [0.22, 0.30, 0.22, 0.14, 0.12] - 5 models
        # - XGBoost highest (gen1a insight: good on tabular data)
        # - RF and ET equal (ensemble diversity)
        # - SVM provides kernel-based boundary
        # - AdaBoost adds boosting diversity
        self.weights = [0.22, 0.30, 0.22, 0.14, 0.12]

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

        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)
        self.ada.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Tree models have feature_importances_
        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_ +
            self.weights[4] * self.ada.feature_importances_
        ) / (self.weights[0] + self.weights[1] + self.weights[2] + self.weights[4])

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]
        proba_ada = self.ada.predict_proba(X)[:, 1]

        return (self.weights[0] * proba_rf +
                self.weights[1] * proba_xgb +
                self.weights[2] * proba_et +
                self.weights[3] * proba_svm +
                self.weights[4] * proba_ada)

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
