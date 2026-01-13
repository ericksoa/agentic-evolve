"""
Gen3-C: SVM with Tuned Hyperparameters and Enhanced Features

Mutation from: gen0_e (SVM with RBF Kernel)
Mutation type: Hyperparameter tuning
Change: Adjusted C=10.0 (from 1.0), gamma='auto', and increased fingerprint bits to 1024

Hypothesis: The parent SVM with C=1.0 and gamma='scale' may be underfitting or
having poor margin separation. Increasing C to 10.0 creates a harder margin that
prioritizes correctly classifying training examples, which is important for
distinguishing hERG blockers. Using gamma='auto' (1/n_features) instead of 'scale'
provides more localized decision boundaries. More fingerprint bits (1024) capture
finer structural details important for hERG binding.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


class HERGPredictor:
    """SVM with tuned hyperparameters for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = SVC(
            C=10.0,              # MUTATION: increased from 1.0 for harder margin
            kernel='rbf',
            gamma='auto',       # MUTATION: changed from 'scale' to 'auto'
            class_weight='balanced',
            probability=True,
            random_state=random_state,
            cache_size=500      # Increased cache for faster training
        )
        self.scaler = StandardScaler()
        self._feature_names = None
        self._n_fp_bits = 1024  # MUTATION: increased from 512 for more detail

    def _calculate_features(self, smiles_list):
        """Calculate features optimized for SVM."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + 15)
            else:
                # Morgan fingerprints with more bits
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # Essential molecular descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                descriptors = [
                    logP, mw, rdMolDescriptors.CalcTPSA(mol),
                    Descriptors.NumRotatableBonds(mol), aromatic_atoms,
                    Descriptors.NumAromaticRings(mol), basic_nitrogens,
                    basic_nitrogens * logP, heavy_atoms,
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol), Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol), Descriptors.Chi1(mol)
                ]

                features = np.concatenate([fp_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'AromaticAtoms',
             'NumAromaticRings', 'BasicNitrogens', 'LipophilicBasicity', 'HeavyAtomCount',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'BertzCT', 'Chi0', 'Chi1']
        )
        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0)
        X = self.scaler.fit_transform(X)
        y = np.array(y)
        self.model.fit(X, y)
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
        """SVM doesn't have direct feature importance."""
        return None
