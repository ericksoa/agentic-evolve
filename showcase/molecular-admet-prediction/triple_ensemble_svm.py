"""
Solution: Triple Ensemble with SVM Predictor
Approach: RF + GBM + SVM ensemble (different from RF+GBM+ET)

Key differences from evolved_ensemble.py:
- Replaces ExtraTrees with SVM (RBF kernel)
- SVM captures different decision boundaries
- Calibrated SVM for proper probability estimates
- Feature selection to reduce SVM complexity
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    Triple ensemble: RF + GBM + Calibrated SVM.

    SVM provides different decision boundary than tree-based models,
    potentially improving ensemble diversity and robustness.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_bits = 1024  # Reduced for SVM efficiency
        self.fp_radius = 2
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(mutual_info_classif, k=500)

        # Three diverse models
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            n_jobs=1,
            random_state=random_state,
            class_weight='balanced'
        )

        self.gbm = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            random_state=random_state
        )

        # Calibrated SVM for probability estimates
        self.svm = CalibratedClassifierCV(
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=random_state
            ),
            cv=3
        )

        self.weights = [0.40, 0.35, 0.25]  # RF, GBM, SVM
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan FP + descriptors."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = np.zeros(self.fp_bits + 15)
            else:
                # Morgan fingerprints (smaller for SVM)
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
                morgan_array = np.array(morgan)

                # Key descriptors
                desc = self._calculate_descriptors(mol)
                fp = np.concatenate([morgan_array, desc])

            features.append(fp)

        return np.array(features)

    def _calculate_descriptors(self, mol):
        """Calculate 15 key descriptors for hERG prediction."""
        return np.array([
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            self._count_basic_nitrogens(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol),
            Lipinski.RingCount(mol),
            Descriptors.MolMR(mol),
            Descriptors.Chi0(mol),
            Descriptors.Kappa1(mol),
        ], dtype=float)

    def _count_basic_nitrogens(self, mol):
        """Count sp3 nitrogens."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetHybridization().name == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def fit(self, X_smiles, y):
        """Train the triple ensemble."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale all features (important for SVM)
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection (helps SVM performance)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)

        y = np.array(y)

        # Train RF and GBM on full features
        self.rf.fit(X_scaled, y)
        self.gbm.fit(X_scaled, y)

        # Train SVM on selected features
        self.svm.fit(X_selected, y)

        self._feature_importances = self.rf.feature_importances_

        return self

    def predict_proba(self, X_smiles):
        """Predict with weighted ensemble."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        # Weighted average
        rf_proba = self.rf.predict_proba(X_scaled)[:, 1] * self.weights[0]
        gbm_proba = self.gbm.predict_proba(X_scaled)[:, 1] * self.weights[1]
        svm_proba = self.svm.predict_proba(X_selected)[:, 1] * self.weights[2]

        return rf_proba + gbm_proba + svm_proba

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary labels."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Get RF feature importance."""
        if self._feature_importances is None:
            return None

        fp_imp = np.sum(self._feature_importances[:self.fp_bits])
        desc_imp = self._feature_importances[self.fp_bits:]

        desc_names = [
            'MolLogP', 'MolWt', 'TPSA', 'NumAromaticRings', 'NumHDonors',
            'NumHAcceptors', 'NumRotatableBonds', 'BasicNitrogens',
            'FractionCSP3', 'HeavyAtomCount', 'NumHeteroatoms', 'RingCount',
            'MolMR', 'Chi0', 'Kappa1'
        ]

        return {
            'features': ['MorganFingerprints'] + desc_names,
            'importances': [fp_imp] + desc_imp.tolist()
        }


if __name__ == '__main__':
    predictor = HERGPredictor()
    test_smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    train_smiles = test_smiles * 20
    train_labels = [0, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Triple Ensemble (RF+GBM+SVM) Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
