"""
Gen0-H: Extra Trees with MACCS Keys

Approach: ExtraTreesClassifier with MACCS structural keys.
MACCS keys encode known chemical substructure patterns (167 bits).

Key features:
- MACCS keys (167 bits) - interpretable structural patterns
- Combined with Morgan fingerprints for complementary coverage
- ExtraTrees with extra randomization for diversity
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors


class HERGPredictor:
    """ExtraTrees with MACCS keys for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._n_morgan = 512
        self._n_maccs = 167
        self._n_descriptors = 15

    def _calculate_features(self, smiles_list):
        """Calculate MACCS + Morgan + descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_morgan + self._n_maccs + self._n_descriptors)
            else:
                # Morgan fingerprints (compact)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_morgan)
                morgan_bits = np.array(morgan_fp)

                # MACCS keys
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # Key descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                descriptors = [
                    logP, mw, heavy_atoms,
                    rdMolDescriptors.CalcTPSA(mol), Descriptors.NumRotatableBonds(mol),
                    aromatic_atoms, Descriptors.NumAromaticRings(mol),
                    basic_nitrogens, basic_nitrogens * logP,
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol), Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol), Descriptors.Chi1(mol)
                ]

                features = np.concatenate([morgan_bits, maccs_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_morgan)] +
            [f'MACCS_{i}' for i in range(self._n_maccs)] +
            ['MolLogP', 'MolWt', 'HeavyAtomCount', 'TPSA', 'NumRotatableBonds',
             'AromaticAtoms', 'NumAromaticRings', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'BertzCT', 'Chi0', 'Chi1']
        )
        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        """Preprocess features."""
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -self._n_descriptors:] = self.scaler.fit_transform(X[:, -self._n_descriptors:])
        else:
            X[:, -self._n_descriptors:] = self.scaler.transform(X[:, -self._n_descriptors:])
        return X

    def fit(self, X_smiles, y):
        """Train the model."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return top 30 most important features."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
