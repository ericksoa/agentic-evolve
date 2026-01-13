"""
Gen3b: Add hERG-specific pharmacophore features

Parent: gen1a (0.8899 ROC-AUC)

Mutation: Feature addition - add hERG-specific pharmacophore features
- Add pKa-related basicity indicators
- Add extended topological indices (Chi2, Chi3, Kappa indices)
- Add hERG-relevant molecular shape descriptors
- Add charge-related features

Hypothesis: hERG channel blocking is strongly influenced by specific
pharmacophore patterns and 3D molecular properties. Adding features
that capture these patterns (basic center strength, molecular shape,
charge distribution) should improve predictive accuracy.
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
    """4-model ensemble with hERG-specific pharmacophore features."""

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

        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # Keep parent's optimized weights
        self.weights = [0.25, 0.35, 0.22, 0.18]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate fingerprints + hERG descriptors + NEW pharmacophore features."""
        all_features = []

        # MUTATION: Added 10 new hERG-relevant features (total now 35)
        n_new_features = 10
        n_total_descriptors = 25 + n_new_features

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + n_total_descriptors)
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

                # Original 25 descriptors
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

                # MUTATION: Add 10 new hERG-relevant features

                # Extended topological indices - capture molecular branching
                chi2 = Descriptors.Chi2v(mol)
                chi3 = Descriptors.Chi3v(mol)

                # Kappa shape indices - capture molecular shape
                kappa1 = Descriptors.Kappa1(mol)
                kappa2 = Descriptors.Kappa2(mol)
                kappa3 = Descriptors.Kappa3(mol)

                # hERG-specific: tertiary/quaternary amine detection (stronger blockers)
                tertiary_amines = sum(1 for atom in mol.GetAtoms()
                                     if atom.GetAtomicNum() == 7
                                     and atom.GetDegree() >= 3
                                     and not atom.GetIsAromatic())

                # Aromatic-basic nitrogen interaction (key hERG motif)
                aromatic_basic_product = aromatic_rings * basic_nitrogens

                # Hydrophobic surface area proxy
                # High logP with large MW = large hydrophobic surface
                hydrophobic_index = logP * heavy_atoms / 100 if heavy_atoms > 0 else 0

                # Charge-related: maximum partial charge (approximated by heteroatom ratio)
                heteroatom_ratio = Descriptors.NumHeteroatoms(mol) / max(1, heavy_atoms)

                # Molecular flexibility with aromaticity (rigid aromatics + flexible chain = hERG motif)
                flex_aromatic_ratio = Descriptors.NumRotatableBonds(mol) / max(1, aromatic_rings + 1)

                new_features = [
                    chi2, chi3,
                    kappa1, kappa2, kappa3,
                    tertiary_amines,
                    aromatic_basic_product,
                    hydrophobic_index,
                    heteroatom_ratio,
                    flex_aromatic_ratio,
                ]

                features = np.concatenate([morgan_bits, maccs_bits, descriptors, new_features])
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
             # New features
             'Chi2v', 'Chi3v', 'Kappa1', 'Kappa2', 'Kappa3',
             'TertiaryAmines', 'AromaticBasicProduct', 'HydrophobicIndex',
             'HeteroatomRatio', 'FlexAromaticRatio']
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        # Scale all descriptors (now 35 instead of 25)
        n_descriptors = 35
        if fit:
            X[:, -n_descriptors:] = self.scaler.fit_transform(X[:, -n_descriptors:])
        else:
            X[:, -n_descriptors:] = self.scaler.transform(X[:, -n_descriptors:])
        return X

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Only tree models have feature_importances_
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
