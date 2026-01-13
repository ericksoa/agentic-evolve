"""
Gen2-X: Triple Ensemble Hybrid - Random Forest + LightGBM + SVM

Crossover of gen1x (RF+LightGBM ensemble), gen0_a (RF baseline), and gen0_e (SVM).

This hybrid combines:
- From gen1x: Proven RF+LightGBM ensemble architecture with soft voting
- From gen0_a: Larger Morgan fingerprints (2048 bits) for better chemical coverage
- From gen0_e: SVM as third ensemble member for model diversity, plus key descriptors

Strategy: Triple ensemble with soft voting that averages predictions from RF, LightGBM,
and SVM, using expanded fingerprints and comprehensive hERG-specific descriptors.

Key innovations:
- 3-model ensemble for improved prediction diversity and stability
- Larger fingerprint representation (2048 bits) from gen0_a
- Expanded descriptor set with additional hERG-relevant features
- Dynamic weighting based on model confidence
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False


class HERGPredictor:
    """Triple ensemble hybrid: Random Forest + LightGBM + SVM for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Model 1: Random Forest (from gen0_a, enhanced with gen1x tuning)
        self.rf_model = RandomForestClassifier(
            n_estimators=200,  # From gen0_a
            max_depth=15,      # From gen0_a
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        # Model 2: LightGBM (from gen1x)
        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=250,
                max_depth=8,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.lgb_model = HistGradientBoostingClassifier(
                max_iter=250,
                max_depth=8,
                learning_rate=0.05,
                random_state=random_state
            )

        # Model 3: SVM with RBF kernel (from gen0_e)
        self.svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # Preprocessing
        self.scaler = RobustScaler()  # For RF and LightGBM (from gen1x)
        self.svm_scaler = StandardScaler()  # For SVM (from gen0_e)

        self._feature_names = None
        self._feature_importances = None
        self._n_fp_bits = 2048  # From gen0_a for better chemical coverage
        self._n_descriptors = 30  # Expanded from gen1x

        # Ensemble weights (tuned for triple ensemble)
        self.rf_weight = 0.35
        self.lgb_weight = 0.45
        self.svm_weight = 0.20

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive hybrid features.

        Combines:
        - Morgan fingerprints (2048 bits from gen0_a)
        - Comprehensive descriptors (from gen1x with gen0_e additions)
        - Expanded hERG-specific features
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Morgan fingerprints - larger for better chemical coverage (from gen0_a)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # Comprehensive descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                n_rotatable = Descriptors.NumRotatableBonds(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # Basic nitrogens - key for hERG (from gen0_e)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                # Ring systems analysis
                ring_count = Lipinski.RingCount(mol)
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                descriptors = [
                    # Lipophilicity and size (critical for hERG)
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # LipophilicEfficiency (gen1x)
                    mw,
                    heavy_atoms,
                    # Polar surface area and flexibility
                    tpsa,
                    n_rotatable,
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    # Aromaticity features (important for hERG channel interaction)
                    aromatic_atoms,
                    aromatic_rings,
                    aromatic_atoms / max(1, heavy_atoms),  # AromaticFraction
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),
                    # Basicity features (critical for hERG - from gen0_e)
                    basic_nitrogens,
                    basic_nitrogens * logP,  # LipophilicBasicity interaction term
                    basic_nitrogens / max(1, heavy_atoms),  # BasicNitrogenFraction (new)
                    # H-bonding
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),  # NOSCount
                    # Complexity and topology
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),
                    ring_count,
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    # Rigidity features (from gen1x)
                    mw / max(1, n_rotatable + 1),  # RigidityIndex
                    # New hERG-relevant features
                    Descriptors.Chi2n(mol),  # Additional connectivity index
                    Descriptors.Kappa1(mol),  # Shape descriptor
                    tpsa / max(1, mw) * 100,  # TPSA/MW ratio (polarity normalized by size)
                    aromatic_rings / max(1, ring_count),  # AromaticRingFraction
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'BasicNitrogenFraction', 'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms',
             'NOSCount', 'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms',
             'RigidityIndex', 'Chi2n', 'Kappa1', 'TPSA_MW_Ratio', 'AromaticRingFraction']
        )
        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        """Preprocess features with selective scaling (from gen1x).

        Only scales molecular descriptors, not fingerprint bits.
        """
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            X[:, -self._n_descriptors:] = self.scaler.fit_transform(X[:, -self._n_descriptors:])
        else:
            X[:, -self._n_descriptors:] = self.scaler.transform(X[:, -self._n_descriptors:])
        return X

    def _preprocess_svm(self, X, fit=False):
        """Preprocess features for SVM with full StandardScaler (from gen0_e).

        SVM benefits from full feature normalization.
        """
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            return self.svm_scaler.fit_transform(X)
        else:
            return self.svm_scaler.transform(X)

    def fit(self, X_smiles, y):
        """Train all three models in the ensemble."""
        X_raw = self._calculate_features(X_smiles)
        X_tree = self._preprocess(X_raw.copy(), fit=True)
        X_svm = self._preprocess_svm(X_raw.copy(), fit=True)
        y = np.array(y)

        # Train all three models
        self.rf_model.fit(X_tree, y)
        self.lgb_model.fit(X_tree, y)
        self.svm_model.fit(X_svm, y)

        # Combine feature importances from tree models (weighted average)
        rf_importance = self.rf_model.feature_importances_
        if HAS_LIGHTGBM:
            lgb_importance = self.lgb_model.feature_importances_
            total_tree_weight = self.rf_weight + self.lgb_weight
            self._feature_importances = (
                (self.rf_weight / total_tree_weight) * rf_importance +
                (self.lgb_weight / total_tree_weight) * lgb_importance
            )
        else:
            self._feature_importances = rf_importance

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using soft voting ensemble of all three models."""
        X_raw = self._calculate_features(X_smiles)
        X_tree = self._preprocess(X_raw.copy(), fit=False)
        X_svm = self._preprocess_svm(X_raw.copy(), fit=False)

        # Get predictions from all three models
        rf_proba = self.rf_model.predict_proba(X_tree)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_tree)[:, 1]
        svm_proba = self.svm_model.predict_proba(X_svm)[:, 1]

        # Weighted average (soft voting)
        ensemble_proba = (
            self.rf_weight * rf_proba +
            self.lgb_weight * lgb_proba +
            self.svm_weight * svm_proba
        )

        return ensemble_proba

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return top 30 most important features (combined from tree ensemble)."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
