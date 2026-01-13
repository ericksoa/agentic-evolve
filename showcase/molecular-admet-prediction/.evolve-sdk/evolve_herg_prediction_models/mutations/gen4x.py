"""
Gen4-X: Advanced Triple Ensemble - Random Forest + LightGBM + SVM

Crossover of gen1x (RF+LightGBM ensemble), gen0_a (RF with larger fingerprints),
and gen0_e (SVM with hERG-specific features).

This hybrid combines:
- From gen1x: Ensemble architecture with RF+LightGBM, comprehensive descriptors,
  RobustScaler preprocessing for descriptors only
- From gen0_a: Larger 2048-bit Morgan fingerprints for better chemical resolution,
  more RF estimators
- From gen0_e: SVM with RBF kernel as third ensemble member,
  hERG-specific basic nitrogen features, standardized feature scaling

Strategy: Triple ensemble with calibrated probability predictions.
Each model contributes based on its strengths - RF for robustness,
LightGBM for gradient optimization, SVM for non-linear decision boundaries.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False


class HERGPredictor:
    """Triple ensemble: Random Forest + LightGBM + SVM for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Model 1: Random Forest (from gen0_a: more estimators, from gen1x: tuned params)
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

        # Model 2: LightGBM (from gen1x with enhanced params)
        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=300,  # Increased for larger feature space
                max_depth=10,      # Slightly deeper for 2048-bit FP
                num_leaves=127,    # More leaves for complexity
                learning_rate=0.04,  # Slower for better generalization
                subsample=0.8,
                colsample_bytree=0.6,  # Lower for higher-dim feature space
                reg_alpha=0.2,
                reg_lambda=1.5,
                min_child_samples=15,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.lgb_model = HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=10,
                learning_rate=0.04,
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
        self.descriptor_scaler = RobustScaler()  # From gen1x: for descriptors
        self.svm_scaler = StandardScaler()       # From gen0_e: for SVM

        self._feature_names = None
        self._feature_importances = None
        self._n_fp_bits = 2048  # From gen0_a: larger fingerprints
        self._n_descriptors = 28  # Extended descriptor set

        # Triple ensemble weights (optimized for hERG prediction)
        self.rf_weight = 0.35
        self.lgb_weight = 0.45
        self.svm_weight = 0.20

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive hybrid features.

        Combines:
        - Larger Morgan fingerprints (2048 bits from gen0_a)
        - Comprehensive descriptors (from gen1x)
        - hERG-specific features (basic nitrogens from gen0_e)
        - Additional interaction terms for hERG channel binding
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Larger Morgan fingerprints (from gen0_a)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # Comprehensive descriptors (gen1x + gen0_e enhanced)
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                num_rot_bonds = Descriptors.NumRotatableBonds(mol)

                # Basic nitrogens - key for hERG (from gen0_e)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                # Quaternary nitrogens (permanently charged - high hERG risk)
                quat_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                     if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0)

                descriptors = [
                    # Lipophilicity and size (critical for hERG)
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # LipophilicEfficiency (gen1x)
                    mw,
                    heavy_atoms,
                    # Polar surface area and flexibility
                    tpsa,
                    num_rot_bonds,
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    # Aromaticity features (important for hERG channel interaction)
                    aromatic_atoms,
                    Descriptors.NumAromaticRings(mol),
                    aromatic_atoms / max(1, heavy_atoms),  # AromaticFraction
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),
                    # Basicity features (critical for hERG - from gen0_e)
                    basic_nitrogens,
                    basic_nitrogens * logP,  # LipophilicBasicity interaction term
                    quat_nitrogens,  # New: quaternary nitrogens
                    # H-bonding
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),  # NOSCount
                    # Complexity and topology
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    # Rigidity (from gen1x)
                    mw / max(1, num_rot_bonds + 1),
                    # New hERG-specific derived features
                    logP - tpsa / 100,  # Net lipophilicity (adjusted for polarity)
                    aromatic_atoms * basic_nitrogens,  # Aromatic-basic interaction
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'QuatNitrogens', 'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms',
             'NOSCount', 'BertzCT', 'Chi0', 'Chi1', 'RingCount',
             'NumBridgeheadAtoms', 'RigidityIndex', 'NetLipophilicity',
             'AromaticBasicInteraction']
        )
        return np.array(all_features)

    def _preprocess_for_tree_models(self, X, fit=False):
        """Preprocess features for RF and LightGBM (from gen1x).

        Only scales molecular descriptors, not fingerprint bits.
        """
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            X[:, -self._n_descriptors:] = self.descriptor_scaler.fit_transform(X[:, -self._n_descriptors:])
        else:
            X[:, -self._n_descriptors:] = self.descriptor_scaler.transform(X[:, -self._n_descriptors:])
        return X

    def _preprocess_for_svm(self, X, fit=False):
        """Preprocess features for SVM (from gen0_e).

        Full standardization for all features as SVM is sensitive to scale.
        """
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            X = self.svm_scaler.fit_transform(X)
        else:
            X = self.svm_scaler.transform(X)
        return X

    def fit(self, X_smiles, y):
        """Train all three models in the ensemble."""
        X_raw = self._calculate_features(X_smiles)
        y = np.array(y)

        # Preprocess for tree models
        X_tree = self._preprocess_for_tree_models(X_raw.copy(), fit=True)

        # Preprocess for SVM
        X_svm = self._preprocess_for_svm(X_raw.copy(), fit=True)

        # Train all models
        self.rf_model.fit(X_tree, y)
        self.lgb_model.fit(X_tree, y)
        self.svm_model.fit(X_svm, y)

        # Combine feature importances from RF and LightGBM (weighted average)
        rf_importance = self.rf_model.feature_importances_
        if HAS_LIGHTGBM:
            lgb_importance = self.lgb_model.feature_importances_
            combined_weight = self.rf_weight + self.lgb_weight
            self._feature_importances = (
                (self.rf_weight / combined_weight) * rf_importance +
                (self.lgb_weight / combined_weight) * lgb_importance
            )
        else:
            self._feature_importances = rf_importance

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using triple soft voting ensemble."""
        X_raw = self._calculate_features(X_smiles)

        # Preprocess for each model type
        X_tree = self._preprocess_for_tree_models(X_raw.copy(), fit=False)
        X_svm = self._preprocess_for_svm(X_raw.copy(), fit=False)

        # Get predictions from all models
        rf_proba = self.rf_model.predict_proba(X_tree)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_tree)[:, 1]
        svm_proba = self.svm_model.predict_proba(X_svm)[:, 1]

        # Weighted average (soft voting with triple ensemble)
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
        """Return top 30 most important features (combined from ensemble)."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
