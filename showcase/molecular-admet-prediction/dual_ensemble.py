"""
Dual Ensemble: Combining the two best predictors (0.8711 each)

A simpler ensemble that averages:
1. evolved_ensemble approach (RF+GBM+ET with FP+descriptors)
2. triple_ensemble_svm approach (RF+GBM+SVM)

Hypothesis: Simpler combination may reduce overfitting.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """
    Dual ensemble: RF+GBM+ET averaged with RF+GBM+SVM.
    Uses 6 models total with equal weighting.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fp_bits = 2048
        self.fp_radius = 2
        self.scaler = StandardScaler()

        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
        )
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV

        # Ensemble 1: RF + GBM + ExtraTrees (from evolved_ensemble)
        self.rf1 = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, n_jobs=1, random_state=random_state,
            class_weight='balanced'
        )
        self.gbm1 = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=random_state
        )
        self.et = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=4,
            n_jobs=1, random_state=random_state, class_weight='balanced'
        )

        # Ensemble 2: RF + GBM + SVM (from triple_ensemble_svm)
        self.rf2 = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            n_jobs=1, random_state=random_state+1, class_weight='balanced'
        )
        self.gbm2 = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=random_state+1
        )
        self.svm = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state),
            cv=3
        )

        # Equal weights (simpler than learned)
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate Morgan FP + descriptors."""
        features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = np.zeros(self.fp_bits + 25)
            else:
                morgan = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
                morgan_array = np.array(morgan)
                desc = self._calculate_descriptors(mol)
                fp = np.concatenate([morgan_array, desc])
            features.append(fp)

        return np.array(features)

    def _calculate_descriptors(self, mol):
        """Calculate 25 hERG-relevant descriptors."""
        return np.array([
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Lipinski.RingCount(mol),
            Descriptors.NumAliphaticRings(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            self._count_basic_nitrogens(mol),
            self._count_total_nitrogens(mol),
            Descriptors.NumHeteroatoms(mol),
            self._safe_descriptor(lambda: Descriptors.BalabanJ(mol), 0),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            self._safe_descriptor(lambda: Descriptors.MaxAbsPartialCharge(mol), 0),
            self._safe_descriptor(lambda: Descriptors.MinAbsPartialCharge(mol), 0),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.LabuteASA(mol),
        ], dtype=float)

    def _count_basic_nitrogens(self, mol):
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetHybridization().name == 'SP3' and atom.GetFormalCharge() >= 0:
                    count += 1
        return count

    def _count_total_nitrogens(self, mol):
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

    def _safe_descriptor(self, func, default):
        try:
            val = func()
            return default if (val is None or np.isnan(val) or np.isinf(val)) else val
        except:
            return default

    def fit(self, X_smiles, y):
        """Train all 6 models."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y)

        # Scale descriptors
        X_fp = X[:, :self.fp_bits]
        X_desc = self.scaler.fit_transform(X[:, self.fp_bits:])
        X_full = np.concatenate([X_fp, X_desc], axis=1)

        # Train all models
        self.rf1.fit(X_full, y)
        self.gbm1.fit(X_full, y)
        self.et.fit(X_full, y)
        self.rf2.fit(X_full, y)
        self.gbm2.fit(X_full, y)
        self.svm.fit(X_full, y)

        self._feature_importances = np.mean([
            self.rf1.feature_importances_,
            self.rf2.feature_importances_,
            self.et.feature_importances_
        ], axis=0)

        return self

    def predict_proba(self, X_smiles):
        """Simple average of all 6 models."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_fp = X[:, :self.fp_bits]
        X_desc = self.scaler.transform(X[:, self.fp_bits:])
        X_full = np.concatenate([X_fp, X_desc], axis=1)

        # Average all predictions
        probas = np.mean([
            self.rf1.predict_proba(X_full)[:, 1],
            self.gbm1.predict_proba(X_full)[:, 1],
            self.et.predict_proba(X_full)[:, 1],
            self.rf2.predict_proba(X_full)[:, 1],
            self.gbm2.predict_proba(X_full)[:, 1],
            self.svm.predict_proba(X_full)[:, 1],
        ], axis=0)

        return probas

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        if self._feature_importances is None:
            return None
        fp_imp = np.sum(self._feature_importances[:self.fp_bits])
        desc_imp = self._feature_importances[self.fp_bits:]
        desc_names = [
            'MolLogP', 'MolMR', 'MolWt', 'HeavyAtomCount', 'NumRotatableBonds',
            'FractionCSP3', 'TPSA', 'NumAromaticRings', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'RingCount', 'NumAliphaticRings',
            'NumHDonors', 'NumHAcceptors', 'BasicNitrogens', 'TotalNitrogens',
            'NumHeteroatoms', 'BalabanJ', 'Chi0', 'Chi1', 'MaxAbsPartialCharge',
            'MinAbsPartialCharge', 'Kappa1', 'Kappa2', 'LabuteASA'
        ]
        return {
            'features': ['MorganFingerprints'] + desc_names,
            'importances': [fp_imp] + desc_imp.tolist()
        }


if __name__ == '__main__':
    predictor = HERGPredictor()
    test_smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    train_smiles = test_smiles * 25
    train_labels = [0, 1] * 25
    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)
    print("Dual Ensemble Predictor:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")
