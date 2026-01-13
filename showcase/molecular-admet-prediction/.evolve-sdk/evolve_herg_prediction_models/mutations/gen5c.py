"""
Gen5-C: SVM with Polynomial Kernel

Mutation from: gen0_e (SVM with RBF Kernel)
Mutation type: Algorithm swap (kernel change)
Change: Replaced RBF kernel with polynomial kernel (degree=3)

Hypothesis: The parent uses an RBF kernel which creates radially symmetric
decision boundaries. For molecular fingerprint data where feature interactions
are important (e.g., combinations of specific substructures that together
indicate hERG liability), a polynomial kernel can explicitly model these
feature interactions through polynomial expansion. A degree-3 polynomial can
capture higher-order interactions between fingerprint bits and molecular
descriptors that may be critical for hERG channel binding - such as the
combination of lipophilicity, basic nitrogen, and aromatic ring features
that together define hERG blockers. The 'coef0=1' ensures cross-terms are
included in the polynomial expansion.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


class HERGPredictor:
    """SVM with polynomial kernel for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = SVC(
            C=1.0,
            kernel='poly',       # MUTATION: changed from 'rbf' to 'poly'
            degree=3,            # MUTATION: added polynomial degree
            coef0=1,             # MUTATION: added to include cross-terms
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._feature_names = None
        self._n_fp_bits = 512  # Smaller for SVM efficiency

    def _calculate_features(self, smiles_list):
        """Calculate features optimized for SVM."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + 15)
            else:
                # Compact Morgan fingerprints
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
