"""
Gen0-I: Stacking Ensemble with Meta-Learner

Approach: Stacking ensemble using out-of-fold predictions as meta-features.
The meta-learner learns optimal combination weights from base model predictions.

Key features:
- Base models: RF, XGB, ExtraTrees
- Meta-learner: Logistic Regression (linear combination)
- 5-fold cross-validation for out-of-fold predictions
- Prevents overfitting through proper stacking protocol
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


class HERGPredictor:
    """Stacking ensemble with meta-learner for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Base models
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=random_state, n_jobs=-1
        )

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                random_state=random_state, eval_metric='logloss', n_jobs=-1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state
            )

        self.et = ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=random_state, n_jobs=-1
        )

        # Meta-learner
        self.meta = LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000, random_state=random_state
        )

        self.scaler = RobustScaler()
        self._feature_names = None
        self._n_fp_bits = 1024
        self._n_descriptors = 20

    def _calculate_features(self, smiles_list):
        """Calculate hybrid features."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                descriptors = [
                    logP, Descriptors.MolMR(mol), mw, heavy_atoms,
                    rdMolDescriptors.CalcTPSA(mol), Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol), aromatic_atoms,
                    Descriptors.NumAromaticRings(mol), aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens, basic_nitrogens * logP,
                    Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol), Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol), Descriptors.Chi1(mol)
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        """Preprocess features."""
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -self._n_descriptors:] = self.scaler.fit_transform(X[:, -self._n_descriptors:])
        else:
            X[:, -self._n_descriptors:] = self.scaler.transform(X[:, -self._n_descriptors:])
        return X

    def fit(self, X_smiles, y):
        """Train with stacking protocol."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Generate out-of-fold predictions for meta-learner training
        n_samples = len(y)
        oof_preds = np.zeros((n_samples, 3))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            # Train base models on fold
            self.rf.fit(X_train, y_train)
            self.xgb.fit(X_train, y_train)
            self.et.fit(X_train, y_train)

            # Get OOF predictions
            oof_preds[val_idx, 0] = self.rf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 1] = self.xgb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, 2] = self.et.predict_proba(X_val)[:, 1]

        # Train meta-learner on OOF predictions
        self.meta.fit(oof_preds, y)

        # Retrain base models on full data for inference
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.et.fit(X, y)

        return self

    def predict_proba(self, X_smiles):
        """Predict using stacked ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        # Get base model predictions
        base_preds = np.column_stack([
            self.rf.predict_proba(X)[:, 1],
            self.xgb.predict_proba(X)[:, 1],
            self.et.predict_proba(X)[:, 1]
        ])

        # Meta-learner prediction
        return self.meta.predict_proba(base_preds)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return meta-learner weights as proxy for model importance."""
        try:
            weights = self.meta.coef_[0]
            return {
                'features': ['RF', 'XGB', 'ExtraTrees'],
                'importances': weights.tolist()
            }
        except Exception:
            return None
