"""
Gen1-X: Ensemble Hybrid - Random Forest + LightGBM

Crossover of gen0_a (Random Forest), gen0_d (LightGBM), and gen0_e (SVM descriptors).

This hybrid combines:
- From gen0_a: Random Forest classifier with balanced weights and feature importance
- From gen0_d: LightGBM with comprehensive feature set and RobustScaler preprocessing
- From gen0_e: Key molecular descriptors relevant to hERG blocking

Strategy: Soft voting ensemble that averages predictions from RF and LightGBM,
using the comprehensive feature set optimized for hERG prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False


class HERGPredictor:
    """Ensemble hybrid: Random Forest + LightGBM for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Model 1: Random Forest (from gen0_a, adapted for hybrid features)
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        # Model 2: LightGBM (from gen0_d)
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

        # Preprocessing (from gen0_d)
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._n_fp_bits = 1024  # Balance between gen0_a (2048) and gen0_e (512)
        self._n_descriptors = 25

        # Ensemble weights (can be tuned)
        self.rf_weight = 0.4
        self.lgb_weight = 0.6

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive hybrid features.

        Combines:
        - Morgan fingerprints (from all parents)
        - Comprehensive descriptors (from gen0_d)
        - hERG-specific features like basic nitrogens (from gen0_e)
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Morgan fingerprints (shared across all parents)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # Comprehensive descriptors (from gen0_d with gen0_e additions)
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

                # Basic nitrogens - key for hERG (from gen0_e)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                descriptors = [
                    # Lipophilicity and size (critical for hERG)
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # LipophilicEfficiency (gen0_d)
                    mw,
                    heavy_atoms,
                    # Polar surface area and flexibility
                    rdMolDescriptors.CalcTPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
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
                    # Rigidity (from gen0_d)
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex']
        )
        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        """Preprocess features with selective scaling (from gen0_d).

        Only scales molecular descriptors, not fingerprint bits.
        """
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -self._n_descriptors:] = self.scaler.fit_transform(X[:, -self._n_descriptors:])
        else:
            X[:, -self._n_descriptors:] = self.scaler.transform(X[:, -self._n_descriptors:])
        return X

    def fit(self, X_smiles, y):
        """Train both models in the ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Train both models
        self.rf_model.fit(X, y)
        self.lgb_model.fit(X, y)

        # Combine feature importances (weighted average)
        rf_importance = self.rf_model.feature_importances_
        if HAS_LIGHTGBM:
            lgb_importance = self.lgb_model.feature_importances_
            self._feature_importances = (
                self.rf_weight * rf_importance + self.lgb_weight * lgb_importance
            )
        else:
            self._feature_importances = rf_importance

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using soft voting ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]

        # Weighted average (soft voting)
        ensemble_proba = self.rf_weight * rf_proba + self.lgb_weight * lgb_proba

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
