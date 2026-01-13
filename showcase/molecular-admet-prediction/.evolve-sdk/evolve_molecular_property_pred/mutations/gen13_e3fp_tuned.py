"""
Gen13-E3FP-Tuned: Optimized E3FP Parameters

Parent: gen11_e3fp (0.8814 ROC-AUC)

Mutation: Optimize E3FP parameters and ensemble weights based on gen11 results.

Changes from gen11_e3fp:
1. E3FP bits: 1024 -> 2048 (capture more 3D patterns)
2. E3FP level: 5 -> 6 (deeper circular substructures)
3. XGBoost weight increased (best single model for high-dim data)
4. Increased n_estimators for all tree models
5. Slightly deeper trees for RF/ET to capture complex 3D patterns

Hypothesis: E3FP's 3D patterns show promise - tuning parameters and giving
more weight to XGBoost (handles high-dimensional sparse data well) should
improve accuracy further.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys

# E3FP imports
try:
    from e3fp.pipeline import fprints_from_mol
    HAS_E3FP = True
except ImportError:
    HAS_E3FP = False


class HERGPredictor:
    """Tuned E3FP ensemble for hERG toxicity."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Tuned E3FP configuration
        self.e3fp_bits = 2048  # Increased from 1024
        self.e3fp_level = 6    # Increased from 5
        self.e3fp_radius_multiplier = 1.718
        self.e3fp_first_radius = 2.0

        self._e3fp_cache = {}

        # Increased n_estimators and slightly deeper trees
        self.rf = RandomForestClassifier(
            n_estimators=120,  # Increased from 80
            max_depth=7,       # Increased from 6
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=120,  # Increased from 80
                max_depth=4,       # Increased from 3
                learning_rate=0.025,
                subsample=0.7,
                colsample_bytree=0.55,
                reg_alpha=0.35,
                reg_lambda=0.45,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.025,
                random_state=random_state
            )

        self.et = ExtraTreesClassifier(
            n_estimators=120,  # Increased from 80
            max_depth=6,       # Increased from 5
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=1
        )

        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )

        # Increased XGBoost weight (best for high-dim sparse)
        self.weights = [0.25, 0.35, 0.25, 0.15]

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _get_e3fp_fingerprint(self, smiles):
        """Generate E3FP fingerprint for a molecule."""
        if smiles in self._e3fp_cache:
            return self._e3fp_cache[smiles]

        if not HAS_E3FP:
            return np.zeros(self.e3fp_bits)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.e3fp_bits)

            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_state
            params.maxAttempts = 50
            params.numThreads = 1

            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                params = AllChem.ETKDGv2()
                params.randomSeed = self.random_state
                result = AllChem.EmbedMolecule(mol, params)
                if result == -1:
                    return np.zeros(self.e3fp_bits)

            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass

            fprints = fprints_from_mol(
                mol,
                fprint_params={
                    'bits': self.e3fp_bits,
                    'level': self.e3fp_level,
                    'radius_multiplier': self.e3fp_radius_multiplier,
                    'first': self.e3fp_first_radius
                }
            )

            if fprints and len(fprints) > 0:
                fp_array = fprints[0].to_vector(sparse=False, dtype=np.uint8)
                self._e3fp_cache[smiles] = fp_array
                return fp_array

            return np.zeros(self.e3fp_bits)

        except Exception as e:
            return np.zeros(self.e3fp_bits)

    def _calculate_features(self, smiles_list):
        """Calculate fingerprints + E3FP + descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + self.e3fp_bits + 25)
            else:
                # Morgan fingerprints
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                # MACCS keys
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # E3FP fingerprints
                e3fp_bits = self._get_e3fp_fingerprint(smi)

                # Molecular descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                descriptors = [
                    logP, Descriptors.MolMR(mol), logP * mw / 100, mw, heavy_atoms,
                    tpsa, Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                    aromatic_atoms, aromatic_rings, aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens, basic_nitrogens * logP, Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol), Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),
                    Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                features = np.concatenate([morgan_bits, maccs_bits, e3fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            [f'E3FP_{i}' for i in range(self.e3fp_bits)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex']
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        # Scale continuous features (last 25 descriptors)
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])
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

        tree_weight_sum = self.weights[0] + self.weights[1] + self.weights[2]
        self._feature_importances = (
            self.weights[0] * self.rf.feature_importances_ +
            self.weights[1] * self.xgb.feature_importances_ +
            self.weights[2] * self.et.feature_importances_
        ) / tree_weight_sum

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
