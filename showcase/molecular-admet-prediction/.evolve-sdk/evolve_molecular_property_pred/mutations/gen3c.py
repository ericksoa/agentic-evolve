"""
Generation 3 Variant C: Deep Regularized Neural Network
Approach: Deeper architecture with stronger regularization and dropout
Strategy: Combat overfitting with deeper network and aggressive regularization
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
import warnings
warnings.filterwarnings('ignore')


class HERGPredictor:
    """
    Deep Regularized Neural Network predictor with multi-scale feature representation.

    Uses multiple fingerprint types (Morgan, MACCS, Avalon) combined with
    molecular descriptors. This version employs a deeper architecture with
    stronger regularization to prevent overfitting.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.morgan_bits = 1024
        self.morgan_radius = 2

        # Deeper neural network with stronger regularization
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32, 16),  # Deeper architecture
            activation='relu',
            solver='adam',
            alpha=0.01,  # Stronger L2 regularization (10x increase)
            learning_rate_init=0.0005,  # Lower learning rate for stability
            max_iter=500,  # More iterations
            early_stopping=True,
            validation_fraction=0.2,  # Larger validation set
            n_iter_no_change=15,  # More patience
            random_state=random_state,
            beta_1=0.9,  # Adam parameters
            beta_2=0.999
        )

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def _morgan_fingerprint(self, mol):
        """Generate Morgan fingerprint."""
        if mol is None:
            return np.zeros(self.morgan_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
        return np.array(fp)

    def _maccs_fingerprint(self, mol):
        """Generate MACCS structural keys."""
        if mol is None:
            return np.zeros(167)  # MACCS has 167 keys
        try:
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)

    def _avalon_fingerprint(self, mol):
        """Generate Avalon fingerprint (if available)."""
        if mol is None:
            return np.zeros(512)
        try:
            fp = pyAvalonTools.GetAvalonFP(mol, 512)
            return np.array(fp)
        except:
            return np.zeros(512)

    def _advanced_descriptors(self, mol):
        """Calculate advanced molecular descriptors."""
        if mol is None:
            return [np.nan] * 20

        try:
            return [
                # Core physicochemical properties
                Descriptors.MolLogP(mol),
                Descriptors.MolWt(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),

                # Electronic and topological
                Descriptors.BalabanJ(mol),
                rdMolDescriptors.BertzCT(mol),
                Descriptors.Ipc(mol),  # Information content
                rdMolDescriptors.Kappa1(mol),  # Molecular connectivity
                rdMolDescriptors.Kappa2(mol),

                # Aromatic and ring features
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeterocycles(mol),
                Descriptors.RingCount(mol),

                # hERG-specific features
                Descriptors.fr_NH0(mol),  # Tertiary amines
                Descriptors.fr_NH1(mol),  # Secondary amines
                Descriptors.fr_NH2(mol),  # Primary amines
                Descriptors.fr_piperdine(mol),  # Piperidine (hERG pharmacophore)
                Descriptors.fr_piperzine(mol),  # Piperazine

                # Surface area and volume
                Descriptors.LabuteASA(mol),
            ]
        except:
            return [np.nan] * 20

    def _extract_features(self, smiles_list):
        """Extract comprehensive multi-scale features."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)

            # Different fingerprint types
            morgan_fp = self._morgan_fingerprint(mol)
            maccs_fp = self._maccs_fingerprint(mol)
            avalon_fp = self._avalon_fingerprint(mol)
            descriptors = self._advanced_descriptors(mol)

            # Concatenate all features
            combined = np.concatenate([
                morgan_fp,
                maccs_fp,
                avalon_fp,
                descriptors
            ])

            all_features.append(combined)

        return np.array(all_features)

    def fit(self, X_smiles, y):
        """Train on SMILES strings and binary labels."""
        X = self._extract_features(X_smiles)
        y = np.array(y)

        # Handle missing values in descriptor portion
        X = self.imputer.fit_transform(X)

        # Scale features (important for neural networks)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking."""
        X = self._extract_features(X_smiles)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_architecture_info(self):
        """Get information about the neural network architecture."""
        total_features = self.morgan_bits + 167 + 512 + 20  # Morgan + MACCS + Avalon + descriptors
        return {
            'hidden_layers': (256, 128, 64, 32, 16),
            'total_features': total_features,
            'morgan_bits': self.morgan_bits,
            'maccs_keys': 167,
            'avalon_bits': 512,
            'descriptors': 20,
            'activation': 'ReLU',
            'optimizer': 'Adam',
            'regularization': 'L2 (alpha=0.01)',
            'learning_rate': 0.0005,
            'validation_fraction': 0.2
        }


if __name__ == '__main__':
    # Quick test to ensure it works
    predictor = HERGPredictor()

    # Test molecules
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # Chloroquine
    ]

    # Dummy training with more data for neural network
    train_smiles = test_smiles * 20  # Neural nets need more data
    train_labels = [0, 1, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print(f"Generation 3C test: {proba}")
    print(f"Architecture: {predictor.get_architecture_info()}")