"""
Gen1x: Enhanced Crossover Hybrid

Parent: gen0_champion (0.890 ROC-AUC)

Crossover Strategy:
- From ensemble architecture: Keep RF+XGB+ET+SVM base with optimized weights
- From feature engineering: Enhanced hERG-specific descriptors with pharmacophoric features
- New addition: Stacking meta-learner for adaptive weight learning
- New addition: Additional molecular descriptors (BCUT, EState, Chi connectivity)

Hypothesis: Combining the strong 4-model ensemble with enhanced features
and an optional stacking meta-learner should capture more patterns.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_predict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys
from rdkit.Chem.EState import EState_VSA


class HERGPredictor:
    """5-model ensemble with stacking meta-learner for hERG toxicity."""

    def __init__(self, random_state=42, use_stacking=True):
        self.random_state = random_state
        self.use_stacking = use_stacking

        # Model 1: Random Forest (from parent - optimized)
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Model 2: XGBoost (from parent - slightly tuned)
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.025,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=0.3,
                reg_lambda=0.6,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.025,
                random_state=random_state
            )

        # Model 3: Extra Trees (from parent - optimized)
        self.et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Model 4: SVM with RBF kernel (from parent)
        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # Model 5: New - Logistic Regression on fingerprints (simple but diverse)
        self.lr = LogisticRegression(
            C=0.5,
            max_iter=500,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Base weights (used when stacking is disabled or for feature importance)
        self.weights = [0.25, 0.25, 0.25, 0.15, 0.10]

        # Meta-learner for stacking
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=300,
            random_state=random_state
        )

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._stacking_fitted = False

    def _calculate_features(self, smiles_list):
        """Calculate enhanced fingerprints + expanded hERG descriptors."""
        all_features = []
        num_descriptors = 35  # Expanded from 25

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + num_descriptors)
            else:
                # Morgan fingerprints (radius 3, 512 bits)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                # MACCS keys (167 bits)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                mol_h = Chem.AddHs(mol)

                # Core descriptors (from parent)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                # Expanded descriptors (enhanced for hERG)
                descriptors = [
                    # From parent (hERG-relevant)
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # Lipophilic efficiency proxy
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

                    # New descriptors (crossover additions)
                    Descriptors.Chi2n(mol),  # Chi connectivity index
                    Descriptors.Chi3n(mol),
                    Descriptors.Kappa1(mol),  # Kappa shape indices
                    Descriptors.Kappa2(mol),
                    Descriptors.HallKierAlpha(mol),
                    Descriptors.LabuteASA(mol),  # Surface area
                    rdMolDescriptors.CalcNumSpiroAtoms(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    basic_nitrogens * aromatic_rings,  # Basic nitrogen in aromatic context
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
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex',
             'Chi2n', 'Chi3n', 'Kappa1', 'Kappa2', 'HallKierAlpha',
             'LabuteASA', 'NumSpiroAtoms', 'NumAliphaticRings', 'NumSaturatedRings',
             'BasicAromaticInteraction']
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        num_descriptors = 35
        if fit:
            X[:, -num_descriptors:] = self.scaler.fit_transform(X[:, -num_descriptors:])
        else:
            X[:, -num_descriptors:] = self.scaler.transform(X[:, -num_descriptors:])
        return X

    def _get_base_predictions(self, X):
        """Get predictions from all base models."""
        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]
        proba_lr = self.lr.predict_proba(X)[:, 1]
        return np.column_stack([proba_rf, proba_xgb, proba_et, proba_svm, proba_lr])

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Fit all base models
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)
        self.lr.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Fit stacking meta-learner using cross-validated predictions
        if self.use_stacking and len(y) >= 20:
            try:
                # Get out-of-fold predictions for stacking
                meta_features = np.zeros((len(y), 5))
                for i, model in enumerate([self.rf, self.xgb, self.et, self.svm, self.lr]):
                    meta_features[:, i] = cross_val_predict(
                        model, X, y, cv=3, method='predict_proba'
                    )[:, 1]
                self.meta_learner.fit(meta_features, y)
                self._stacking_fitted = True
            except Exception:
                # Fall back to simple weighting if stacking fails
                self._stacking_fitted = False
        else:
            self._stacking_fitted = False

        # Feature importances (only from tree-based models)
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

        base_preds = self._get_base_predictions(X)

        if self._stacking_fitted:
            # Use meta-learner for final prediction
            return self.meta_learner.predict_proba(base_preds)[:, 1]
        else:
            # Fall back to weighted average
            return np.dot(base_preds, self.weights)

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
