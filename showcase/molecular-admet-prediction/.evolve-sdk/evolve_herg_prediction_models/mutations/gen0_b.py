"""
Gen0-B: Gradient Boosting with Molecular Descriptors

Approach: GradientBoostingClassifier with handcrafted molecular descriptors.
Uses domain knowledge about hERG binding (basic nitrogens, lipophilicity, aromaticity).

Key features:
- 25 carefully selected molecular descriptors
- Known hERG pharmacophore features (basic N, aromatic rings, logP)
- RobustScaler for handling outliers in physicochemical properties
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """Gradient Boosting with molecular descriptors for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=random_state
        )
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_features(self, smiles_list):
        """Calculate molecular descriptors for hERG prediction."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(25)
            else:
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                features = [
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # Lipophilic efficiency proxy
                    mw,
                    heavy_atoms,
                    rdMolDescriptors.CalcTPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    aromatic_atoms,
                    Descriptors.NumAromaticRings(mol),
                    aromatic_atoms / max(1, heavy_atoms),  # Aromatic fraction
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens,
                    basic_nitrogens * logP,  # Lipophilic basicity
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),  # NOS count
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),  # Rigidity index
                ]
            all_features.append(features)

        self._feature_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]
        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)
        X = self.scaler.fit_transform(X)
        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return feature importances."""
        if self._feature_importances is None or self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance],
            'importances': [x[1] for x in sorted_importance]
        }
