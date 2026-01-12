"""
Baseline hERG Predictor using Molecular Descriptors + Gradient Boosting.

This approach uses interpretable physicochemical descriptors known to be
relevant for hERG binding: lipophilicity, molecular weight, charge, and
topological properties.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class HERGPredictor:
    """
    hERG toxicity predictor using molecular descriptors and Gradient Boosting.

    Uses RDKit molecular descriptors that are known to correlate with hERG
    binding: lipophilicity (logP), molecular size, charge distribution,
    and aromatic content.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_descriptors(self, smiles_list):
        """Calculate molecular descriptors from SMILES."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use NaN (will be imputed)
                desc = [np.nan] * 20
            else:
                desc = [
                    # Lipophilicity - key for hERG binding
                    Descriptors.MolLogP(mol),
                    Descriptors.MolMR(mol),  # Molar refractivity

                    # Size and shape
                    Descriptors.MolWt(mol),
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcTPSA(mol),  # Topological polar surface area

                    # Aromatic features - relevant for hERG binding pocket
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),
                    Lipinski.RingCount(mol),

                    # H-bond capacity
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),

                    # Charge-related
                    Descriptors.NumHeteroatoms(mol),
                    rdMolDescriptors.CalcNumAmideBonds(mol),

                    # Topological indices
                    Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) != 0 else 0,
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),

                    # Fraction of sp3 carbons (flexibility)
                    rdMolDescriptors.CalcFractionCSP3(mol),

                    # Basic nitrogen count (key for hERG)
                    len([atom for atom in mol.GetAtoms()
                         if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0])
                ]
            descriptors.append(desc)

        self._feature_names = [
            'MolLogP', 'MolMR', 'MolWt', 'HeavyAtomCount', 'NumRotatableBonds',
            'TPSA', 'NumAromaticRings', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'RingCount', 'NumHDonors', 'NumHAcceptors',
            'NumHeteroatoms', 'NumAmideBonds', 'BalabanJ', 'BertzCT',
            'Chi0', 'Chi1', 'FractionCSP3', 'BasicNitrogenCount'
        ]

        return np.array(descriptors)

    def _impute_and_scale(self, X, fit=False):
        """Handle missing values and scale features."""
        # Simple imputation - replace NaN with column median
        X = np.array(X, dtype=float)
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if not np.isnan(median_val) else 0

        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings and labels."""
        X = self._calculate_descriptors(X_smiles)
        X = self._impute_and_scale(X, fit=True)
        y = np.array(y)

        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_descriptors(X_smiles)
        X = self._impute_and_scale(X, fit=False)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance with descriptor names."""
        if self._feature_importances is None or self._feature_names is None:
            return None

        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance],
            'importances': [x[1] for x in sorted_importance]
        }


if __name__ == '__main__':
    # Quick test
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen
    ]

    train_smiles = test_smiles * 10
    train_labels = [0, 1, 0] * 10

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")

    print("\nFeature importance:")
    fi = predictor.get_feature_importance()
    for name, imp in zip(fi['features'][:10], fi['importances'][:10]):
        print(f"  {name}: {imp:.4f}")
