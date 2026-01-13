"""
Gen2-C: SVM with Enhanced hERG-Specific Features

Mutation from: gen0_e (SVM with RBF Kernel)
Mutation type: Feature addition
Change: Added MACCS keys fingerprints and additional charge/electrostatic descriptors

Hypothesis: hERG channel binding is strongly influenced by molecular electrostatics
and the presence of specific pharmacophoric features. MACCS keys capture predefined
structural patterns that may be more interpretable for hERG binding. Adding
Gasteiger partial charges and electrostatic features should improve prediction
since hERG binding often involves positively charged moieties interacting with
the channel's selectivity filter.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys


class HERGPredictor:
    """SVM with enhanced hERG-specific features for hERG prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._feature_names = None
        self._n_fp_bits = 512  # Morgan fingerprint bits
        self._n_maccs_bits = 167  # MACCS keys

    def _calculate_features(self, smiles_list):
        """Calculate features with enhanced hERG-specific descriptors."""
        all_features = []
        n_descriptors = 20  # Expanded from 15 to 20

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(self._n_fp_bits + self._n_maccs_bits + n_descriptors)
            else:
                # Morgan fingerprints (circular)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._n_fp_bits)
                fp_bits = np.array(fp)

                # MACCS keys (structural patterns)
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                # Essential molecular descriptors
                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)

                # Calculate Gasteiger charges for electrostatic features
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
                    charges = [c if not np.isnan(c) and not np.isinf(c) else 0.0 for c in charges]
                    max_positive_charge = max(charges) if charges else 0.0
                    max_negative_charge = min(charges) if charges else 0.0
                    charge_range = max_positive_charge - max_negative_charge
                except:
                    max_positive_charge = 0.0
                    max_negative_charge = 0.0
                    charge_range = 0.0

                descriptors = [
                    logP, mw, rdMolDescriptors.CalcTPSA(mol),
                    Descriptors.NumRotatableBonds(mol), aromatic_atoms,
                    Descriptors.NumAromaticRings(mol), basic_nitrogens,
                    basic_nitrogens * logP,  # Lipophilic basicity
                    heavy_atoms,
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol), Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol), Descriptors.Chi1(mol),
                    # New electrostatic features
                    max_positive_charge,
                    max_negative_charge,
                    charge_range,
                    rdMolDescriptors.CalcLabuteASA(mol),  # Accessible surface area
                    rdMolDescriptors.CalcNumAmideBonds(mol),  # Amide bonds
                ]

                features = np.concatenate([fp_bits, maccs_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan2_{i}' for i in range(self._n_fp_bits)] +
            [f'MACCS_{i}' for i in range(self._n_maccs_bits)] +
            ['MolLogP', 'MolWt', 'TPSA', 'NumRotatableBonds', 'AromaticAtoms',
             'NumAromaticRings', 'BasicNitrogens', 'LipophilicBasicity', 'HeavyAtomCount',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'BertzCT', 'Chi0', 'Chi1',
             'MaxPositiveCharge', 'MaxNegativeCharge', 'ChargeRange', 'LabuteASA', 'NumAmideBonds']
        )
        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        y = np.array(y)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_features(X_smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """SVM doesn't have direct feature importance."""
        return None
