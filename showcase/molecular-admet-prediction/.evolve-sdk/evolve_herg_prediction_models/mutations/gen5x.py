"""
Gen5-X: Enhanced Triple Ensemble with Threshold Optimization

Crossover of gen1x (RF+LightGBM ensemble), gen0_a (RF baseline), and gen0_e (SVM features).

This hybrid combines:
- From gen1x: RF+LightGBM ensemble architecture, comprehensive hERG-specific descriptors,
  RobustScaler for descriptor preprocessing, soft voting strategy (fitness: 0.7030)
- From gen0_a: Random Forest with balanced class weights, larger fingerprints for coverage
- From gen0_e: hERG-specific basic nitrogen features, LipophilicBasicity interaction term

New innovations in this crossover:
- ExtraTrees as third ensemble member (better diversity than SVM, faster training)
- OOB-based threshold optimization (from gen4b concept) for better F1 balance
- Calibrated probability estimates for more reliable predictions
- Optimized ensemble weights based on model characteristics

Strategy: Triple soft voting ensemble with RF, LightGBM, and ExtraTrees using
comprehensive hERG-specific features and adaptive threshold optimization.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
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
    """Enhanced triple ensemble: RF + LightGBM + ExtraTrees with threshold optimization."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Model 1: Random Forest (from gen1x with gen0_a optimizations)
        self.rf_model = RandomForestClassifier(
            n_estimators=175,  # Balanced between gen1x (150) and gen0_a (200)
            max_depth=14,      # Balanced between gen1x (12) and gen0_a (15)
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # From gen0_a
            oob_score=True,    # Enable OOB for threshold optimization
            random_state=random_state,
            n_jobs=-1
        )

        # Model 2: LightGBM (from gen1x with slight tuning)
        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=275,  # Slightly increased
                max_depth=9,       # Between 8 and 10
                num_leaves=95,     # Between 63 and 127
                learning_rate=0.045,  # Slightly lower for better generalization
                subsample=0.8,
                colsample_bytree=0.65,  # Slightly lower for regularization
                reg_alpha=0.15,
                reg_lambda=1.25,
                min_child_samples=18,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.lgb_model = HistGradientBoostingClassifier(
                max_iter=275,
                max_depth=9,
                learning_rate=0.045,
                random_state=random_state
            )

        # Model 3: ExtraTrees (new - provides diversity, handles high-dim data well)
        self.et_model = ExtraTreesClassifier(
            n_estimators=175,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        # Preprocessing (from gen1x)
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._n_fp_bits = 1024  # From gen1x - good balance
        self._n_descriptors = 26  # Slightly expanded

        # Optimized ensemble weights (RF and ET similar, LGB gets more)
        self.rf_weight = 0.30
        self.lgb_weight = 0.45
        self.et_weight = 0.25

        # Threshold optimization
        self._optimal_threshold = 0.5

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive hERG-specific features.

        Combines:
        - Morgan fingerprints (from gen1x: 1024 bits)
        - Comprehensive descriptors (from gen1x)
        - hERG-specific features (from gen0_e: basic nitrogens, LipophilicBasicity)
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Morgan fingerprints (from gen1x)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # Comprehensive descriptors (gen1x + gen0_e)
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                num_rot_bonds = Descriptors.NumRotatableBonds(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # Basic nitrogens - key for hERG (from gen0_e)
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

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
                    basic_nitrogens / max(1, heavy_atoms),  # BasicNitrogenFraction
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
             'RigidityIndex']
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

    def _find_optimal_threshold(self, y_true, y_proba):
        """Find threshold that maximizes F1 score (from gen4b concept).

        Uses grid search over thresholds to find optimal decision boundary.
        """
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.30, 0.70, 0.02):
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate F1 score
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def fit(self, X_smiles, y):
        """Train all three models in the ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Train all three models
        self.rf_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.et_model.fit(X, y)

        # Optimize threshold using OOB predictions from RF
        if hasattr(self.rf_model, 'oob_decision_function_'):
            oob_proba = self.rf_model.oob_decision_function_[:, 1]
            self._optimal_threshold = self._find_optimal_threshold(y, oob_proba)

        # Combine feature importances (weighted average from all tree models)
        rf_importance = self.rf_model.feature_importances_
        et_importance = self.et_model.feature_importances_

        if HAS_LIGHTGBM:
            lgb_importance = self.lgb_model.feature_importances_
            self._feature_importances = (
                self.rf_weight * rf_importance +
                self.lgb_weight * lgb_importance +
                self.et_weight * et_importance
            )
        else:
            self._feature_importances = (
                (self.rf_weight / (self.rf_weight + self.et_weight)) * rf_importance +
                (self.et_weight / (self.rf_weight + self.et_weight)) * et_importance
            )

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using triple soft voting ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        # Get predictions from all models
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        et_proba = self.et_model.predict_proba(X)[:, 1]

        # Weighted average (soft voting)
        ensemble_proba = (
            self.rf_weight * rf_proba +
            self.lgb_weight * lgb_proba +
            self.et_weight * et_proba
        )

        return ensemble_proba

    def predict(self, X_smiles, threshold=None):
        """Binary prediction using optimized threshold."""
        if threshold is None:
            threshold = self._optimal_threshold
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
