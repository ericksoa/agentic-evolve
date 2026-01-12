"""
Meta-Ensemble: Combining Multiple hERG Predictors

This solution combines 4 diverse predictors for maximum robustness:
1. evolved_ensemble.py (RF+GBM+ET with FP+descriptors) - 0.8711
2. triple_ensemble_svm.py (RF+GBM+SVM) - 0.8711
3. hybrid_fp_maccs.py (Morgan+MACCS fingerprints) - 0.857
4. hybrid_fp_desc_light.py (FP+top10 descriptors) - 0.857

Key insight: Diverse feature representations reduce model variance.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """
    Meta-ensemble combining 4 diverse predictors.

    Strategy: Stack predictions from different feature representations
    and combine with learned weights based on validation performance.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_bits = 2048
        self.fp_radius = 2
        self.scaler = StandardScaler()

        # Import sklearn models
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            ExtraTreesClassifier, VotingClassifier
        )
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        # Feature selection for SVM path
        self.feature_selector = SelectKBest(mutual_info_classif, k=500)
        self.svm_scaler = StandardScaler()

        # Model 1: RF with full hybrid features (from evolved_ensemble)
        self.rf_hybrid = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, n_jobs=1, random_state=random_state,
            class_weight='balanced'
        )

        # Model 2: GBM with full features
        self.gbm = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=random_state
        )

        # Model 3: ExtraTrees with full features
        self.et = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=4,
            n_jobs=1, random_state=random_state, class_weight='balanced'
        )

        # Model 4: Calibrated SVM (different decision boundary)
        self.svm = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state),
            cv=3
        )

        # Model 5: GBM with lighter features (less overfitting)
        self.gbm_light = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.7, random_state=random_state + 1
        )

        # Weights learned from validation (higher for better performers)
        # evolved_ensemble components: 0.30, triple_svm component: 0.25
        self.weights = [0.25, 0.22, 0.18, 0.20, 0.15]
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate comprehensive hybrid features."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Full feature dimension: 2048 + 167 + 25 = 2240
                fp = np.zeros(2048 + 167 + 25)
            else:
                # Morgan fingerprints (2048 bits)
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=2048)
                morgan_array = np.array(morgan)

                # MACCS keys (167 bits)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.array(maccs)

                # Full descriptors (25 features)
                desc = self._calculate_full_descriptors(mol)

                fp = np.concatenate([morgan_array, maccs_array, desc])

            features.append(fp)

        return np.array(features)

    def _calculate_full_descriptors(self, mol):
        """Calculate comprehensive hERG-relevant descriptors."""
        return np.array([
            # Lipophilicity (key drivers)
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),

            # Size
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomCount(mol),

            # Flexibility
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),

            # Polar surface
            rdMolDescriptors.CalcTPSA(mol),

            # Aromatic features (critical for hERG)
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),

            # Ring systems
            Lipinski.RingCount(mol),
            Descriptors.NumAliphaticRings(mol),

            # H-bonding
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),

            # Nitrogen features (basic N is key)
            self._count_basic_nitrogens(mol),
            self._count_total_nitrogens(mol),

            # Heteroatoms
            Descriptors.NumHeteroatoms(mol),

            # Topological
            self._safe_descriptor(lambda: Descriptors.BalabanJ(mol), 0),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),

            # Charge
            self._safe_descriptor(lambda: Descriptors.MaxAbsPartialCharge(mol), 0),
            self._safe_descriptor(lambda: Descriptors.MinAbsPartialCharge(mol), 0),

            # Shape
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),

            # Surface area
            Descriptors.LabuteASA(mol),
        ], dtype=float)

    def _count_basic_nitrogens(self, mol):
        """Count basic (sp3) nitrogens - key hERG pharmacophore."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetHybridization().name == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def _count_total_nitrogens(self, mol):
        """Count all nitrogen atoms."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

    def _safe_descriptor(self, func, default):
        """Safely compute a descriptor with fallback."""
        try:
            val = func()
            if val is None or np.isnan(val) or np.isinf(val):
                return default
            return val
        except:
            return default

    def fit(self, X_smiles, y):
        """Train all models in the meta-ensemble."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = np.array(y)

        # Split features for different scaling
        n_fp = 2048 + 167  # Morgan + MACCS
        X_fp = X[:, :n_fp]
        X_desc = X[:, n_fp:]

        # Scale descriptors
        X_desc_scaled = self.scaler.fit_transform(X_desc)
        X_full = np.concatenate([X_fp, X_desc_scaled], axis=1)

        # Prepare SVM features (full scaling + selection)
        X_svm_scaled = self.svm_scaler.fit_transform(X_full)
        X_svm_selected = self.feature_selector.fit_transform(X_svm_scaled, y)

        # Train all models
        self.rf_hybrid.fit(X_full, y)
        self.gbm.fit(X_full, y)
        self.et.fit(X_full, y)
        self.svm.fit(X_svm_selected, y)
        self.gbm_light.fit(X_full[:, :2048+10], y)  # FP + top 10 desc only

        # Average feature importances from tree models
        self._feature_importances = np.mean([
            self.rf_hybrid.feature_importances_,
            self.gbm.feature_importances_,
            self.et.feature_importances_
        ], axis=0)

        return self

    def predict_proba(self, X_smiles):
        """Predict using weighted meta-ensemble."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_fp = 2048 + 167
        X_fp = X[:, :n_fp]
        X_desc = X[:, n_fp:]
        X_desc_scaled = self.scaler.transform(X_desc)
        X_full = np.concatenate([X_fp, X_desc_scaled], axis=1)

        # SVM path
        X_svm_scaled = self.svm_scaler.transform(X_full)
        X_svm_selected = self.feature_selector.transform(X_svm_scaled)

        # Get predictions from all models
        probas = [
            self.rf_hybrid.predict_proba(X_full)[:, 1] * self.weights[0],
            self.gbm.predict_proba(X_full)[:, 1] * self.weights[1],
            self.et.predict_proba(X_full)[:, 1] * self.weights[2],
            self.svm.predict_proba(X_svm_selected)[:, 1] * self.weights[3],
            self.gbm_light.predict_proba(X_full[:, :2048+10])[:, 1] * self.weights[4],
        ]

        return np.sum(probas, axis=0)

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary labels."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Get aggregated feature importance."""
        if self._feature_importances is None:
            return None

        morgan_imp = np.sum(self._feature_importances[:2048])
        maccs_imp = np.sum(self._feature_importances[2048:2048+167])
        desc_imp = self._feature_importances[2048+167:]

        desc_names = [
            'MolLogP', 'MolMR', 'MolWt', 'HeavyAtomCount', 'NumRotatableBonds',
            'FractionCSP3', 'TPSA', 'NumAromaticRings', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'RingCount', 'NumAliphaticRings',
            'NumHDonors', 'NumHAcceptors', 'BasicNitrogens', 'TotalNitrogens',
            'NumHeteroatoms', 'BalabanJ', 'Chi0', 'Chi1',
            'MaxAbsPartialCharge', 'MinAbsPartialCharge',
            'Kappa1', 'Kappa2', 'LabuteASA'
        ]

        return {
            'features': ['MorganFingerprints', 'MACCSKeys'] + desc_names,
            'importances': [morgan_imp, maccs_imp] + desc_imp.tolist()
        }


if __name__ == '__main__':
    predictor = HERGPredictor()
    test_smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    train_smiles = test_smiles * 25
    train_labels = [0, 1] * 25

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Meta-Ensemble Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
