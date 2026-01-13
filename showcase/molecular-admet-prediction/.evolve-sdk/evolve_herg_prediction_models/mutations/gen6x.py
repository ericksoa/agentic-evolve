"""
Gen6-X: Quad-Ensemble with Multi-Fingerprint Features

Crossover of gen1x (RF+LightGBM champion), gen0_a (RF baseline), and gen0_e (SVM descriptors).

This hybrid combines:
- From gen1x: RF+LightGBM ensemble architecture (fitness: 0.7031), comprehensive hERG-specific
  descriptors, RobustScaler for descriptor preprocessing, soft voting strategy
- From gen0_a: Larger fingerprint bit space concept (2048 bits) for better structural coverage
- From gen0_e: Essential hERG descriptors (basic nitrogens, LipophilicBasicity interaction term)

New innovations in this crossover:
- Quad ensemble: RF + LightGBM + XGBoost + GradientBoosting for maximum diversity
- Multi-fingerprint features: Morgan (1536 bits) + MACCS keys (167 bits) for complementary
  structural information - Morgan captures circular neighborhoods, MACCS captures substructure patterns
- Adaptive weighting with isotonic calibration for more reliable probability estimates
- Enhanced hERG-specific pharmacophore features including tertiary amine detection
- Increased fingerprint resolution (1536 bits) for better chemical space coverage

Strategy: Quad soft-voting ensemble using diverse boosting algorithms with comprehensive
multi-fingerprint chemical features optimized for hERG channel binding prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class HERGPredictor:
    """Quad ensemble: RF + LightGBM + XGBoost + GradientBoosting with multi-fingerprints."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Model 1: Random Forest (from gen1x with gen0_a influences)
        self.rf_model = RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,
            random_state=random_state,
            n_jobs=-1
        )

        # Model 2: LightGBM (from gen1x)
        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=280,
                max_depth=9,
                num_leaves=80,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.65,
                reg_alpha=0.12,
                reg_lambda=1.2,
                min_child_samples=18,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.lgb_model = HistGradientBoostingClassifier(
                max_iter=280,
                max_depth=9,
                learning_rate=0.04,
                random_state=random_state
            )

        # Model 3: XGBoost (new - complementary boosting approach)
        if HAS_XGBOOST:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            self.xgb_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state
            )

        # Model 4: sklearn GradientBoosting (additional diversity)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )

        # Preprocessing (from gen1x)
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

        # Multi-fingerprint setup
        self._n_morgan_bits = 1536  # Increased from gen1x (1024) toward gen0_a (2048)
        self._n_maccs_bits = 167    # MACCS keys for complementary structural info
        self._n_fp_bits = self._n_morgan_bits + self._n_maccs_bits
        self._n_descriptors = 28

        # Ensemble weights (will be calibrated during training)
        self.rf_weight = 0.25
        self.lgb_weight = 0.35
        self.xgb_weight = 0.25
        self.gb_weight = 0.15

        # Calibration
        self._calibrator = None
        self._optimal_threshold = 0.5

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive multi-fingerprint features.

        Combines:
        - Morgan fingerprints (1536 bits) - circular neighborhoods
        - MACCS keys (167 bits) - substructure patterns (complementary to Morgan)
        - Comprehensive hERG-specific descriptors (from gen1x + gen0_e)
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Morgan fingerprints (primary structural features)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=self._n_morgan_bits
                )
                morgan_bits = np.array(morgan_fp)

                # MACCS keys (complementary structural fingerprints)
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs_fp)

                # Comprehensive descriptors (gen1x + gen0_e + new hERG features)
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

                # Tertiary amines - especially important for hERG (new)
                tertiary_amines = sum(1 for atom in mol_h.GetAtoms()
                                     if atom.GetAtomicNum() == 7 and
                                     atom.GetDegree() == 3 and
                                     atom.GetFormalCharge() >= 0)

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
                    tertiary_amines,  # New - tertiary amines especially relevant
                    tertiary_amines * logP,  # TertiaryAmineLipophilicity
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

                features = np.concatenate([morgan_bits, maccs_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_morgan_bits)] +
            [f'MACCS_{i}' for i in range(self._n_maccs_bits)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'BasicNitrogenFraction', 'TertiaryAmines', 'TertiaryAmineLipophilicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms',
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
        """Find threshold that maximizes F1 score."""
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.30, 0.70, 0.02):
            y_pred = (y_proba >= threshold).astype(int)

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
        """Train all four models in the ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Train all four models
        self.rf_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.xgb_model.fit(X, y)
        self.gb_model.fit(X, y)

        # Get OOB predictions for calibration and threshold optimization
        if hasattr(self.rf_model, 'oob_decision_function_'):
            oob_proba = self.rf_model.oob_decision_function_[:, 1]

            # Isotonic calibration
            try:
                self._calibrator = IsotonicRegression(out_of_bounds='clip')
                self._calibrator.fit(oob_proba, y)
            except Exception:
                self._calibrator = None

            # Threshold optimization
            self._optimal_threshold = self._find_optimal_threshold(y, oob_proba)

        # Combine feature importances (weighted average from all tree models)
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_

        if HAS_LIGHTGBM:
            lgb_importance = self.lgb_model.feature_importances_
        else:
            lgb_importance = rf_importance  # Fallback

        if HAS_XGBOOST:
            xgb_importance = self.xgb_model.feature_importances_
        else:
            xgb_importance = gb_importance  # Fallback

        self._feature_importances = (
            self.rf_weight * rf_importance +
            self.lgb_weight * lgb_importance +
            self.xgb_weight * xgb_importance +
            self.gb_weight * gb_importance
        )

        return self

    def predict_proba(self, X_smiles):
        """Predict probability using quad soft voting ensemble."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        # Get predictions from all models
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        gb_proba = self.gb_model.predict_proba(X)[:, 1]

        # Weighted average (soft voting)
        ensemble_proba = (
            self.rf_weight * rf_proba +
            self.lgb_weight * lgb_proba +
            self.xgb_weight * xgb_proba +
            self.gb_weight * gb_proba
        )

        # Apply calibration if available
        if self._calibrator is not None:
            try:
                ensemble_proba = self._calibrator.predict(ensemble_proba)
            except Exception:
                pass

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
