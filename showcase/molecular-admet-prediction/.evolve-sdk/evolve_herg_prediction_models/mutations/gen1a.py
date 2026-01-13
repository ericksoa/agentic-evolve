"""
Gen1-A: Random Forest with Morgan Fingerprints + hERG-Relevant Descriptors

Approach: Combines Morgan fingerprints with handcrafted molecular descriptors
known to be predictive of hERG blocking activity.

Mutation from gen0_a: Added 15 molecular descriptors to complement fingerprints.
- Morgan fingerprints capture local chemical environments
- Descriptors add global physicochemical properties (logP, TPSA, basic N, aromaticity)

Hypothesis: Combining fingerprints with domain-knowledge descriptors should
improve predictive power as hERG binding correlates with lipophilicity,
basicity, and molecular size.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


class HERGPredictor:
    """Random Forest with Morgan fingerprints + molecular descriptors for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self._feature_names = None
        self._feature_importances = None

    def _calculate_descriptors(self, mol):
        """Calculate hERG-relevant molecular descriptors."""
        if mol is None:
            return np.zeros(15)

        try:
            mol_h = Chem.AddHs(mol)
            heavy_atoms = Descriptors.HeavyAtomCount(mol)
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                  if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
            logP = Descriptors.MolLogP(mol)

            descriptors = [
                logP,
                Descriptors.MolWt(mol),
                rdMolDescriptors.CalcTPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                aromatic_atoms,
                Descriptors.NumAromaticRings(mol),
                aromatic_atoms / max(1, heavy_atoms),  # Aromatic fraction
                basic_nitrogens,
                basic_nitrogens * logP,  # Lipophilic basicity - key hERG feature
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Descriptors.NumHeteroatoms(mol),
                Lipinski.RingCount(mol),
                heavy_atoms,
                rdMolDescriptors.CalcFractionCSP3(mol),
            ]
            return np.array(descriptors)
        except:
            return np.zeros(15)

    def _calculate_features(self, smiles_list):
        """Calculate Morgan fingerprints + molecular descriptors."""
        all_features = []
        n_fp_bits = 2048
        n_desc = 15

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(n_fp_bits + n_desc)
            else:
                # Morgan fingerprints
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_fp_bits)
                fp_array = np.array(fp)

                # Molecular descriptors
                desc_array = self._calculate_descriptors(mol)

                # Concatenate
                features = np.concatenate([fp_array, desc_array])
            all_features.append(features)

        # Feature names
        fp_names = [f'Morgan2_{i}' for i in range(n_fp_bits)]
        desc_names = [
            'MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'RingCount',
            'HeavyAtomCount', 'FractionCSP3'
        ]
        self._feature_names = fp_names + desc_names

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)
        y = np.array(y)
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)
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
