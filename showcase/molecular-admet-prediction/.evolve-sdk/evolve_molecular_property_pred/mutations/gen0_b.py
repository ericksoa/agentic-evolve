"""
Generation 0 Variant B: Gradient Boosting with Molecular Descriptors
Approach: Interpretable physicochemical features with boosting
Strategy: Use domain knowledge of hERG-relevant molecular properties
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Gradient Boosting predictor using molecular descriptors.

    Focuses on physicochemical properties known to be relevant for hERG binding:
    - Lipophilicity (LogP) - affects membrane permeability
    - Molecular size/weight - affects binding pocket fit
    - Charge properties - ionic interactions with channel
    - Aromatic content - π-π stacking interactions
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Gradient boosting with conservative parameters
        self.model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=random_state
        )

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _calculate_descriptors(self, smiles_list):
        """Calculate hERG-relevant molecular descriptors from SMILES."""
        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use placeholder values
                desc = [np.nan] * 22
            else:
                desc = [
                    # Core lipophilicity - critical for hERG binding
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),  # Topological polar surface area

                    # Molecular size and flexibility
                    Descriptors.MolWt(mol),
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.NumRotatableBonds(mol),

                    # Aromatic characteristics
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),

                    # Hydrogen bonding potential
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),

                    # Charge and electronic properties
                    rdMolDescriptors.BertzCT(mol),  # Complexity
                    Descriptors.BalabanJ(mol),  # Topological index

                    # Nitrogen content (important for hERG)
                    Descriptors.fr_NH0(mol),  # Tertiary amines
                    Descriptors.fr_NH1(mol),  # Secondary amines
                    Descriptors.fr_NH2(mol),  # Primary amines
                    Descriptors.fr_Ndealkylation1(mol),
                    Descriptors.fr_Ndealkylation2(mol),

                    # Ring properties
                    Descriptors.RingCount(mol),
                    rdMolDescriptors.Ipc(mol),  # Information content

                    # Additional hERG-relevant features
                    Descriptors.LabuteASA(mol),  # Accessible surface area
                    Descriptors.PEOE_VSA1(mol),  # Partial charge descriptors
                    Descriptors.VSA_EState1(mol),  # E-state indices
                ]

            descriptors.append(desc)

        return np.array(descriptors)

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._calculate_descriptors(X_smiles)
        y = np.array(y)

        # Handle missing values and scale
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._calculate_descriptors(X_smiles)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get most important molecular descriptors."""
        feature_names = [
            'MolLogP', 'TPSA', 'MolWt', 'HeavyAtomCount', 'NumRotatableBonds',
            'NumAromaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
            'NumHDonors', 'NumHAcceptors', 'BertzCT', 'BalabanJ',
            'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
            'RingCount', 'Ipc', 'LabuteASA', 'PEOE_VSA1', 'VSA_EState1'
        ]

        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]

        return {
            'top_features': [feature_names[i] for i in top_idx],
            'importances': importances[top_idx].tolist()
        }


if __name__ == '__main__':
    # Quick test to ensure it works
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    ]

    # Dummy training
    predictor.fit(test_smiles * 5, [0, 1] * 5)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 0B test: {proba}")
    print(f"Top features: {predictor.get_feature_importance()}")