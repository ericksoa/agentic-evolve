"""
Weighted Ensemble: RF + XGBoost + ExtraTrees with hERG Descriptors

Mutation from gen0_b: Creates an ensemble of 3 diverse classifiers using
the proven hERG-specific descriptors. Ensemble diversity typically improves
generalization and reduces variance.

Hypothesis: Three diverse tree-based models with different inductive biases
will produce more robust predictions than a single XGBoost model.
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
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    hERG toxicity predictor using a weighted ensemble of 3 classifiers:
    - Random Forest (handles mixed features well)
    - XGBoost (gradient boosting power)
    - ExtraTrees (additional diversity through random splits)

    All use the same hERG-specific descriptors (25 features).
    Weights: RF=0.35, XGB=0.35, ET=0.30
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # XGBoost
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )

        # ExtraTrees
        self.et = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=3,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Ensemble weights (tuned for hERG)
        self.weights = [0.35, 0.35, 0.30]  # RF, XGB, ET

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_herg_descriptors(self, smiles_list):
        """Calculate hERG-specific molecular descriptors."""
        descriptors = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                desc = [np.nan] * 25
            else:
                mol_h = Chem.AddHs(mol)

                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # Basic nitrogen count
                basic_nitrogens = 0
                for atom in mol_h.GetAtoms():
                    if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0:
                        basic_nitrogens += 1

                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                desc = [
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,
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
                    basic_nitrogens * logP,
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
                ]

            descriptors.append(desc)

        self._feature_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]

        return np.array(descriptors)

    def _impute_and_scale(self, X, fit=False):
        """Handle missing values and scale features."""
        X = np.array(X, dtype=float)

        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if not np.isnan(median_val) else 0

        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def fit(self, X_smiles, y):
        """Train all ensemble models."""
        X = self._calculate_herg_descriptors(X_smiles)
        X = self._impute_and_scale(X, fit=True)
        y = np.array(y)

        # Fit all models
        self.rf.fit(X, y)
        self.et.fit(X, y)

        if HAS_XGBOOST:
            # XGBoost with sample weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Average feature importance across models
        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        )

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using weighted ensemble."""
        X = self._calculate_herg_descriptors(X_smiles)
        X = self._impute_and_scale(X, fit=False)

        # Get probabilities from each model
        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]

        # Weighted average
        proba = (
            self.weights[0] * proba_rf +
            self.weights[1] * proba_xgb +
            self.weights[2] * proba_et
        )

        return proba

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get ensemble-averaged feature importance."""
        if self._feature_importances is None or self._feature_names is None:
            return None

        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance],
            'importances': [x[1] for x in sorted_importance]
        }


if __name__ == '__main__':
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'
    ]

    train_smiles = test_smiles * 15
    train_labels = [0, 1, 0, 1] * 15

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Ensemble (RF + XGB + ET) Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
