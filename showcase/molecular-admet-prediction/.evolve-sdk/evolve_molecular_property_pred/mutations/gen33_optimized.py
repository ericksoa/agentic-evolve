"""
Gen33: Optimized SMILES Augmentation

Parent: gen32 (0.8690 ± 0.0047 TDC AUROC)

Final optimization:
- More augmentation variants (15 train, 7 test)
- Optimized ensemble hyperparameters
- Careful regularization tuning
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


def enumerate_smiles(smiles, n_variants=15):
    """Generate multiple SMILES representations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    variants = set()
    variants.add(smiles)
    # Also add canonical
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        variants.add(canonical)
    except:
        pass

    for _ in range(n_variants * 5):
        try:
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(random_smiles)
            if len(variants) >= n_variants:
                break
        except:
            continue

    return list(variants)[:n_variants]


class HERGPredictor:
    """Optimized ensemble with maximal SMILES augmentation."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.n_augment_train = 15
        self.n_augment_test = 7

        # Optimized RF
        self.rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=6, min_samples_leaf=3,
            max_features='sqrt', class_weight='balanced',
            random_state=random_state, n_jobs=1, bootstrap=True, oob_score=True
        )

        # Optimized XGBoost
        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.02, subsample=0.75,
                colsample_bytree=0.65, reg_alpha=0.2, reg_lambda=0.3,
                min_child_weight=3,
                random_state=random_state, eval_metric='logloss', n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.02, random_state=random_state
            )

        # Optimized ET
        self.et = ExtraTreesClassifier(
            n_estimators=150, max_depth=7, min_samples_split=6, min_samples_leaf=3,
            class_weight='balanced', random_state=random_state, n_jobs=1
        )

        # Optimized SVM
        self.svm = SVC(
            C=0.8, kernel='rbf', gamma='scale', class_weight='balanced',
            probability=True, random_state=random_state
        )

        # Slightly favor tree models over SVM
        self.weights = [0.30, 0.30, 0.28, 0.12]
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate fingerprints + descriptors."""
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

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])
        return X

    def _predict_single(self, X):
        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]

        return (self.weights[0] * proba_rf +
                self.weights[1] * proba_xgb +
                self.weights[2] * proba_et +
                self.weights[3] * proba_svm)

    def fit(self, X_smiles, y):
        np.random.seed(self.random_state)

        aug_smiles = []
        aug_labels = []
        for smi, label in zip(X_smiles, y):
            variants = enumerate_smiles(smi, self.n_augment_train)
            aug_smiles.extend(variants)
            aug_labels.extend([label] * len(variants))

        X = self._calculate_features(aug_smiles)
        X = self._preprocess(X, fit=True)
        y_aug = np.array(aug_labels)

        self.rf.fit(X, y_aug)
        self.et.fit(X, y_aug)
        self.svm.fit(X, y_aug)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_aug), y=y_aug)
            sample_weights = np.array([class_weights[int(label)] for label in y_aug])
            self.xgb.fit(X, y_aug, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y_aug)

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

        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        ) / (self.weights[0] + self.weights[1] + self.weights[2])

        return self

    def predict_proba(self, X_smiles):
        X_smiles = list(X_smiles)
        all_proba = []

        for smi in X_smiles:
            variants = enumerate_smiles(smi, self.n_augment_test)
            X = self._calculate_features(variants)
            X = self._preprocess(X, fit=False)
            proba = self._predict_single(X)
            all_proba.append(np.mean(proba))

        return np.array(all_proba)

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
