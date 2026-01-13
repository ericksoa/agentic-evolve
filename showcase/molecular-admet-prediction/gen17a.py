"""
Gen17a: hERG-Specific Pharmacophore Features

Parent: gen12c (0.890 ROC-AUC)

Mutation: Add hERG-specific pharmacophore features
- 10 new descriptors targeting known hERG blocker characteristics
- Basic nitrogen, hydrophobic aromatic regions, optimal LogP ranges
- Total 35 descriptors (up from 25)

New features:
1. pKa_basic - count of protonatable nitrogens
2. formal_charge - net formal charge
3. sp2_fraction - fraction of sp2 carbons
4. aromatic_n_count - aromatic nitrogen count
5. logP_squared - nonlinear LogP relationship
6. logP_optimal - distance from optimal hERG LogP (~3.5)
7. basic_aromatic_interaction - interaction term
8. mw_per_ring - MW normalized by ring count
9. nitrogen_aromatic_ratio - basic N / aromatic atoms
10. heteroatom_density - heteroatoms / heavy atoms

Hypothesis: hERG-specific features capturing pharmacophore requirements
should improve prediction accuracy.
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

        # Add SVM with RBF kernel
        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # 4-model weights (give SVM lower weight as it's different)
        self.weights = [0.28, 0.28, 0.28, 0.16]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate compact fingerprints + hERG descriptors (35 total)."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # 512 Morgan + 167 MACCS + 35 descriptors
                features = np.zeros(512 + 167 + 35)
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
                ring_count = Lipinski.RingCount(mol)
                num_heteroatoms = Descriptors.NumHeteroatoms(mol)
                sp3_fraction = rdMolDescriptors.CalcFractionCSP3(mol)

                # Original 25 descriptors
                descriptors = [
                    logP, Descriptors.MolMR(mol), logP * mw / 100, mw, heavy_atoms,
                    tpsa, Descriptors.NumRotatableBonds(mol), sp3_fraction,
                    aromatic_atoms, aromatic_rings, aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens, basic_nitrogens * logP, Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol), num_heteroatoms,
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),
                    Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
                    ring_count, rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                # NEW: 10 hERG-specific pharmacophore features
                # 1. pKa_basic - count of protonatable nitrogens (nitrogens with H attached)
                pKa_basic = sum(1 for atom in mol_h.GetAtoms()
                               if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0)

                # 2. formal_charge - net formal charge
                formal_charge = Chem.GetFormalCharge(mol)

                # 3. sp2_fraction - complement to sp3_fraction
                sp2_fraction = 1.0 - sp3_fraction

                # 4. aromatic_n_count - aromatic nitrogen count
                aromatic_n_count = sum(1 for atom in mol.GetAtoms()
                                       if atom.GetAtomicNum() == 7 and atom.GetIsAromatic())

                # 5. logP_squared - nonlinear LogP relationship
                logP_squared = logP * logP

                # 6. logP_optimal - distance from optimal hERG LogP (~3.5)
                logP_optimal = 1.0 / (1 + abs(logP - 3.5))

                # 7. basic_aromatic_interaction - interaction term
                basic_aromatic_interaction = basic_nitrogens * aromatic_rings

                # 8. mw_per_ring - MW normalized by ring count
                mw_per_ring = mw / max(1, ring_count)

                # 9. nitrogen_aromatic_ratio - basic N / aromatic atoms
                nitrogen_aromatic_ratio = basic_nitrogens / max(1, aromatic_atoms)

                # 10. heteroatom_density - heteroatoms / heavy atoms
                heteroatom_density = num_heteroatoms / max(1, heavy_atoms)

                # Add new features to descriptors
                herg_features = [
                    pKa_basic,
                    formal_charge,
                    sp2_fraction,
                    aromatic_n_count,
                    logP_squared,
                    logP_optimal,
                    basic_aromatic_interaction,
                    mw_per_ring,
                    nitrogen_aromatic_ratio,
                    heteroatom_density,
                ]

                descriptors.extend(herg_features)

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
             # New hERG-specific features
             'pKa_basic', 'FormalCharge', 'SP2Fraction', 'AromaticNCount',
             'LogPSquared', 'LogPOptimal', 'BasicAromaticInteraction',
             'MWPerRing', 'NitrogenAromaticRatio', 'HeteroatomDensity']
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        # Scale all 35 descriptors (not fingerprints)
        if fit:
            X[:, -35:] = self.scaler.fit_transform(X[:, -35:])
        else:
            X[:, -35:] = self.scaler.transform(X[:, -35:])
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
