"""
Gen2-A: Ensemble with Optimized Decision Threshold

Mutation from: gen1x (Ensemble Hybrid - Random Forest + LightGBM)
Mutation type: Threshold optimization
Change: Added automatic threshold calibration during training using validation split

Hypothesis: The default 0.5 threshold may not be optimal for the hERG dataset which
likely has class imbalance. By finding the threshold that maximizes the F1-score on
a held-out validation set during training, we can improve the balance between
precision and recall, leading to better overall fitness.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False


class HERGPredictor:
    """Ensemble hybrid with optimized decision threshold for hERG prediction."""

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

        # Optimized threshold (MUTATION: will be calibrated during training)
        self.optimal_threshold = 0.5

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

    def _find_optimal_threshold(self, y_true, y_proba):
        """Find threshold that maximizes F1 score."""
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.3, 0.7, 0.02):
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate F1 score manually to avoid sklearn dependency
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
        """Train both models in the ensemble with threshold optimization."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Split data for threshold calibration (MUTATION)
        if len(X) > 100:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        # Train both models on training portion
        self.rf_model.fit(X_train, y_train)
        self.lgb_model.fit(X_train, y_train)

        # Calibrate threshold on validation set (MUTATION)
        rf_proba_val = self.rf_model.predict_proba(X_val)[:, 1]
        lgb_proba_val = self.lgb_model.predict_proba(X_val)[:, 1]
        ensemble_proba_val = self.rf_weight * rf_proba_val + self.lgb_weight * lgb_proba_val
        self.optimal_threshold = self._find_optimal_threshold(y_val, ensemble_proba_val)

        # Retrain on full data for final model
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

    def predict(self, X_smiles, threshold=None):
        """Binary prediction using optimized threshold."""
        if threshold is None:
            threshold = self.optimal_threshold
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
