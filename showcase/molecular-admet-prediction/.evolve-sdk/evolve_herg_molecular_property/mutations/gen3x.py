"""
Crossover: Triple Ensemble + Dual Fingerprints + Descriptors

Parents:
- gen2x: RF+XGB ensemble with Morgan+MACCS+descriptors (champion at 0.854)
- gen2b: RF+XGB+ET triple ensemble with descriptors (0.854)

Crossover combines:
1. Triple ensemble from gen2b (RF + XGBoost + ExtraTrees)
2. Dual fingerprints from gen2x (Morgan 1024 + MACCS 167)
3. hERG-specific descriptors (25 features)
4. Optimized ensemble weights based on gen2 performance

Total: 1216 features with 3-model ensemble
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
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """
    Ultimate hERG predictor combining best elements from gen2x and gen2b:
    - Dual fingerprints: Morgan (1024) + MACCS (167)
    - hERG descriptors (25 features)
    - Triple ensemble: RF (0.30) + XGBoost (0.45) + ExtraTrees (0.25)

    Total: 1216 features, 3 diverse models
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Random Forest - good for high-dimensional mixed features
        self.rf = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # XGBoost - strong gradient boosting with regularization
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.15,
                min_child_weight=2,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.07,
                random_state=random_state
            )

        # ExtraTrees - adds diversity through random splits
        self.et = ExtraTreesClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_split=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # Optimized weights: XGBoost highest (best single model performance)
        self.weights = [0.30, 0.45, 0.25]  # RF, XGB, ET

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate dual fingerprints + hERG descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(1024 + 167 + 25)
            else:
                # 1. Morgan fingerprints (from gen2x)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                morgan_bits = np.array(morgan_fp)

                # 2. MACCS keys (from gen2x)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # 3. hERG-specific descriptors (shared by both parents)
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

                features = np.concatenate([morgan_bits, maccs_bits, descriptors])

            all_features.append(features)

        morgan_names = [f'Morgan_{i}' for i in range(1024)]
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

        # Only scale descriptor features (last 25)
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])

        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Fit all three models
        self.rf.fit(X, y)
        self.et.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Weighted average of feature importances
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

        # Weighted ensemble prediction
        proba = (
            self.weights[0] * proba_rf +
            self.weights[1] * proba_xgb +
            self.weights[2] * proba_et
        )

        return proba

    def predict(self, X_smiles, threshold=0.5):
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

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
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'
    ]
    train_smiles = test_smiles * 15
    train_labels = [0, 1, 0, 1] * 15
    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)
    print("Triple Ensemble + Dual FP Crossover Test:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
