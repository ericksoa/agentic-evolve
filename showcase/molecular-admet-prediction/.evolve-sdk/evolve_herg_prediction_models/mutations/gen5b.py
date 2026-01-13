"""
Gen5-B: Random Forest with hERG-Specific Physicochemical Features

Mutation from: gen0_a (Random Forest with Morgan Fingerprints)
Mutation type: Feature addition
Change: Added key physicochemical descriptors known to correlate with hERG binding

Hypothesis: The parent (gen0_a) uses only Morgan fingerprints (2048 bits), which
capture structural patterns but miss important physicochemical properties. hERG
channel binding is well-known to correlate with:
1. Lipophilicity (logP) - hydrophobic drugs bind better to the hERG channel
2. Molecular weight - larger molecules often have higher hERG liability
3. Basic nitrogen count - positive charge at physiological pH increases binding
4. Aromatic ring count - pi-stacking with channel residues

Adding these 6 targeted descriptors should improve prediction without significantly
increasing model complexity. This is a more focused approach than using many
descriptors, reducing risk of overfitting while capturing domain knowledge.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


class HERGPredictor:
    """Random Forest with Morgan fingerprints + hERG-specific features."""

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
        self._n_fp_bits = 2048  # Same as parent
        self._n_descriptors = 6  # MUTATION: Added descriptors

    def _calculate_features(self, smiles_list):
        """Calculate Morgan fingerprints + hERG-specific physicochemical features.

        MUTATION: Added 6 targeted descriptors for hERG prediction.
        """
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_descriptors)
            else:
                # Morgan fingerprints (same as parent)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # MUTATION: hERG-specific physicochemical descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)

                # Basic nitrogen count - critical for hERG binding
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                # Aromatic features - pi-stacking with hERG channel
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                descriptors = [
                    logP,                    # Lipophilicity
                    mw,                      # Molecular weight
                    basic_nitrogens,         # Basicity
                    aromatic_rings,          # Aromaticity
                    basic_nitrogens * logP,  # Lipophilic basicity interaction
                    rdMolDescriptors.CalcTPSA(mol),  # Polar surface area
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            ['MolLogP', 'MolWt', 'BasicNitrogens', 'NumAromaticRings',
             'LipophilicBasicity', 'TPSA']
        )
        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)  # Handle any NaN values
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
