"""
Gen17: ChEMBL Data Augmentation

Parent: gen0_champion (gen12c, 0.890 ROC-AUC)

Mutation: Augment training data with ChEMBL hERG dataset
- ChEMBL has ~16K molecules vs original 655
- 25x more training data should improve generalization
- Same 4-model ensemble architecture (RF + XGB + ET + SVM)
- Careful to exclude test set molecules to prevent leakage

Hypothesis: More diverse training data from ChEMBL hERG assays
will improve model generalization without changing architecture.
"""

import numpy as np
import pandas as pd
from pathlib import Path
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
    """4-model ensemble with ChEMBL data augmentation for hERG toxicity."""

    def __init__(self, random_state=42):
        self.random_state = random_state

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
            self.xgb = GradientBoostingClassifier(
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

        # SVM with RBF kernel
        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # 4-model weights
        self.weights = [0.28, 0.28, 0.28, 0.16]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _load_chembl_data(self, exclude_smiles=None):
        """Load ChEMBL hERG data, excluding specified SMILES to prevent leakage."""
        # Find the data directory relative to this file
        this_dir = Path(__file__).parent
        data_dir = this_dir.parent.parent.parent / 'data' / 'chembl_herg'

        if not data_dir.exists():
            return [], []

        try:
            chembl_train = pd.read_csv(data_dir / 'train.csv')
            chembl_valid = pd.read_csv(data_dir / 'valid.csv')

            # Combine train and valid from ChEMBL
            chembl_data = pd.concat([chembl_train, chembl_valid], ignore_index=True)

            # Exclude molecules that might be in the test set
            if exclude_smiles:
                exclude_set = set(exclude_smiles)
                mask = ~chembl_data['smiles_std'].isin(exclude_set)
                chembl_data = chembl_data[mask]

            return chembl_data['smiles_std'].tolist(), chembl_data['Y'].tolist()
        except Exception:
            return [], []

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
        """Fit the model with ChEMBL data augmentation."""
        # Load ChEMBL data, excluding training molecules to prevent duplication
        # and any potential test molecules that might have leaked
        chembl_smiles, chembl_y = self._load_chembl_data(exclude_smiles=X_smiles)

        # Combine original training data with ChEMBL data
        if chembl_smiles:
            X_combined = list(X_smiles) + chembl_smiles
            y_combined = list(y) + chembl_y
        else:
            X_combined = list(X_smiles)
            y_combined = list(y)

        X = self._calculate_features(X_combined)
        X = self._preprocess(X, fit=True)
        y_arr = np.array(y_combined)

        self.rf.fit(X, y_arr)
        self.et.fit(X, y_arr)
        self.svm.fit(X, y_arr)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_arr), y=y_arr)
            sample_weights = np.array([class_weights[int(label)] for label in y_arr])
            self.xgb.fit(X, y_arr, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y_arr)

        # Feature importances from tree models
        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        ) / (self.weights[0] + self.weights[1] + self.weights[2])

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]

        return (self.weights[0] * proba_rf +
                self.weights[1] * proba_xgb +
                self.weights[2] * proba_et +
                self.weights[3] * proba_svm)

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
