"""
Gen5c: RF-Dominant Triple Ensemble with Optimized Class Handling

Parent: gen4x (0.858 ROC-AUC), gen4c (0.856 RF-weighted)

Mutation: Increase RF weight since RF handles class imbalance better
- RF gets highest weight (0.45) due to balanced class handling
- XGBoost reduced (0.30) but with better regularization
- ExtraTrees moderate weight (0.25) for diversity
- Deeper RF for better pattern capture

Hypothesis: RF with built-in class_weight='balanced' may be more robust
than XGBoost with sample weights for imbalanced molecular data.
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
    RF-dominant triple ensemble for hERG toxicity prediction:
    - Same features as gen4x: Morgan r=3 + MACCS + descriptors
    - RF gets highest weight for better class imbalance handling
    - Deeper RF trees for more complex pattern capture
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Random Forest - dominant model with deeper trees
        self.rf = RandomForestClassifier(
            n_estimators=200,    # More estimators
            max_depth=13,        # Deeper for complex patterns
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # XGBoost - supporting role with strong regularization
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=175,
                max_depth=4,         # Shallower to prevent overfitting
                learning_rate=0.06,
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=0.15,
                reg_lambda=0.2,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=175,
                max_depth=4,
                learning_rate=0.06,
                random_state=random_state
            )

        # ExtraTrees - diversity component
        self.et = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=11,
            min_samples_split=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        # RF-dominant ensemble weights
        self.weights = [0.45, 0.30, 0.25]  # RF, XGB, ET

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate extended dual fingerprints + hERG descriptors (same as gen4x)."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(1024 + 167 + 25)
            else:
                # Extended Morgan fingerprints (radius=3)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
                morgan_bits = np.array(morgan_fp)

                # MACCS keys
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # hERG descriptors
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

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

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

        return self.weights[0] * proba_rf + self.weights[1] * proba_xgb + self.weights[2] * proba_et

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
    print("Gen5c: RF-Dominant Triple Ensemble predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
