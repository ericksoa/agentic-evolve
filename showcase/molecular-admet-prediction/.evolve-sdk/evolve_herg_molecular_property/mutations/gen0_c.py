"""
Lightweight Neural Network with Combined Features

This solution uses a simple neural network combining multiple molecular representations:
Morgan fingerprints, pharmacophore fingerprints, and key physicochemical descriptors.
Optimized for speed with minimal hidden layers and dropout for regularization.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class HERGPredictor:
    """
    hERG toxicity predictor using Neural Network with combined molecular features.

    Combines multiple molecular representations:
    - Morgan fingerprints (substructure)
    - Pharmacophore-like features
    - Key physicochemical descriptors

    Uses lightweight MLP optimized for fast inference.
    """

    def __init__(self, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),  # Lightweight architecture
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._feature_names = None

    def _calculate_combined_features(self, smiles_list):
        """Calculate combined molecular features."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use default values
                features = np.zeros(1024 + 50)  # Morgan + descriptors
            else:
                # 1. Morgan fingerprints (1024 bits)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                morgan_bits = np.array(morgan_fp)

                # 2. Key physicochemical descriptors (25 features)
                mol_h = Chem.AddHs(mol)

                # Basic properties
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # hERG-relevant features
                basic_nitrogens = len([atom for atom in mol_h.GetAtoms()
                                     if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0])

                descriptors = [
                    logP, mw, tpsa,
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    basic_nitrogens,
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumHeteroatoms(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    Descriptors.BertzCT(mol),

                    # Interaction features
                    logP * basic_nitrogens,  # Lipophilic basicity
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),  # Rigidity
                    tpsa / mw if mw > 0 else 0,  # Polar surface efficiency
                ]

                # 3. Pharmacophore-like features (25 features)
                pharmaco_features = []

                # Count specific atom patterns
                patterns = [
                    '[#7+]',  # Positive nitrogen
                    '[#7]',   # Any nitrogen
                    '[OH]',   # Hydroxyl
                    'c1ccccc1',  # Benzene ring
                    '[#16]',  # Sulfur
                    '[F,Cl,Br,I]',  # Halogens
                    '[#7]~[#6]~[#6]~[#7]',  # N-C-C-N pattern
                    'c1ccc([#7])cc1',  # Aniline-like
                    '[#6]=[#6]',  # Double bond
                    '[#6]#[#6]',  # Triple bond
                ]

                for pattern in patterns:
                    try:
                        matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                        pharmaco_features.append(matches)
                    except:
                        pharmaco_features.append(0)

                # Extend to 25 features with additional calculated properties
                while len(pharmaco_features) < 25:
                    pharmaco_features.append(0)

                # Combine all features
                features = np.concatenate([
                    morgan_bits,
                    descriptors,
                    pharmaco_features[:25]
                ])

            all_features.append(features)

        # Feature names for interpretability
        morgan_names = [f'Morgan_{i}' for i in range(1024)]
        descriptor_names = [
            'MolLogP', 'MolWt', 'TPSA', 'HeavyAtomCount', 'NumRotatableBonds',
            'NumAromaticRings', 'BasicNitrogens', 'NumHDonors', 'NumHAcceptors',
            'NumHeteroatoms', 'FractionCSP3', 'BertzCT', 'LipophilicBasicity',
            'Rigidity', 'PolarSurfaceEfficiency'
        ]
        pharmaco_names = [f'Pharmaco_{i}' for i in range(25)]

        self._feature_names = morgan_names + descriptor_names + pharmaco_names

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings and labels."""
        X = self._calculate_combined_features(X_smiles)
        y = np.array(y)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X = self.scaler.fit_transform(X)

        # Fit neural network
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_combined_features(X_smiles)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get approximate feature importance using input weights."""
        if not hasattr(self.model, 'coefs_') or self._feature_names is None:
            return None

        # Use first layer weights as proxy for feature importance
        input_weights = np.abs(self.model.coefs_[0]).mean(axis=1)
        top_indices = np.argsort(input_weights)[-50:][::-1]

        return {
            'features': [self._feature_names[i] for i in top_indices],
            'importances': input_weights[top_indices].tolist()
        }


if __name__ == '__main__':
    # Quick test
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'  # Basic lipophilic compound
    ]

    train_smiles = test_smiles * 20
    train_labels = [0, 1, 0, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Neural Network + Combined Features Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")

    print("\nTop feature weights:")
    fi = predictor.get_feature_importance()
    if fi:
        for name, imp in zip(fi['features'][:8], fi['importances'][:8]):
            print(f"  {name}: {imp:.4f}")