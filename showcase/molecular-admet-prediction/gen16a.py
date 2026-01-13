"""
Gen16a: Stacking Meta-Learner Ensemble

Parent: gen12c (0.890 ROC-AUC)

Mutation: Replace weighted average with stacking meta-learner
- Keep same 4 base models: RF + XGB + ET + SVM
- Use StackingClassifier with LogisticRegression as meta-learner
- Meta-learner learns optimal combination weights via cross-validation
- cv=3 for small dataset stability

Hypothesis: A learned meta-learner should find better combination
weights than manual tuning, adapting to the specific prediction patterns.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
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
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """Stacking ensemble with learned meta-learner for hERG toxicity."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Base estimators (same as gen12c)
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
            self.xgb_model = xgb.XGBClassifier(
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
            self.xgb_model = GradientBoostingClassifier(
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

        # Meta-learner: LogisticRegression learns optimal combination
        self.meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )

        # Build stacking classifier
        self.stacking_clf = StackingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb_model),
                ('et', self.et),
                ('svm', self.svm)
            ],
            final_estimator=self.meta_learner,
            cv=3,  # 3-fold CV for small dataset
            stack_method='predict_proba',
            passthrough=False,  # Only use base model predictions
            n_jobs=1  # Avoid parallel overhead
        )

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._is_fitted = False

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

        # Check for minimal training data - need at least 2*cv samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(counts)

        if min_samples_per_class < 6:
            # Fallback to simple averaging if too few samples for CV
            self._use_fallback = True
            self._fallback_weights = [0.28, 0.28, 0.28, 0.16]

            # Fit base models directly
            self.rf.fit(X, y)
            self.et.fit(X, y)
            self.svm.fit(X, y)

            if HAS_XGBOOST:
                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                sample_weights = np.array([class_weights[int(label)] for label in y])
                self.xgb_model.fit(X, y, sample_weight=sample_weights)
            else:
                self.xgb_model.fit(X, y)
        else:
            # Use stacking classifier
            self._use_fallback = False

            # For XGBoost with sample weights, we need custom handling
            # StackingClassifier doesn't support per-estimator sample weights
            # So we fit stacking without XGBoost sample weights
            self.stacking_clf.fit(X, y)

        self._is_fitted = True

        # Compute feature importances from tree-based models
        if hasattr(self, '_use_fallback') and self._use_fallback:
            rf_imp = self.rf.feature_importances_
            xgb_imp = self.xgb_model.feature_importances_
            et_imp = self.et.feature_importances_
        else:
            # Get fitted estimators from stacking classifier
            fitted_estimators = dict(self.stacking_clf.named_estimators_)
            rf_imp = fitted_estimators['rf'].feature_importances_
            xgb_imp = fitted_estimators['xgb'].feature_importances_
            et_imp = fitted_estimators['et'].feature_importances_

        self._feature_importances = (rf_imp + xgb_imp + et_imp) / 3.0

        return self

    def predict_proba(self, X_smiles):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        if hasattr(self, '_use_fallback') and self._use_fallback:
            # Use weighted average fallback
            proba_rf = self.rf.predict_proba(X)[:, 1]
            proba_xgb = self.xgb_model.predict_proba(X)[:, 1]
            proba_et = self.et.predict_proba(X)[:, 1]
            proba_svm = self.svm.predict_proba(X)[:, 1]

            return (self._fallback_weights[0] * proba_rf +
                    self._fallback_weights[1] * proba_xgb +
                    self._fallback_weights[2] * proba_et +
                    self._fallback_weights[3] * proba_svm)
        else:
            # Use stacking classifier
            return self.stacking_clf.predict_proba(X)[:, 1]

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
