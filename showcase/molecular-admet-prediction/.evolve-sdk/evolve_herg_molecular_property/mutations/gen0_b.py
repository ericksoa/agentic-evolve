"""
XGBoost with Physicochemical Descriptors and Feature Engineering

This solution uses XGBoost with engineered molecular descriptors specifically
targeting hERG-relevant properties: lipophilicity, aromaticity, basic nitrogen content.
Includes sophisticated feature engineering and class weight balancing.
"""

import numpy as np
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight


class HERGPredictor:
    """
    hERG toxicity predictor using XGBoost with hERG-specific descriptors.

    Focuses on physicochemical properties known to correlate with hERG binding:
    - Basic nitrogen content (crucial for hERG channel blocking)
    - Lipophilicity and size
    - Aromatic ring systems
    - Molecular flexibility
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            # Fallback to GradientBoosting if XGBoost not available
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state
            )

        self.scaler = RobustScaler()  # More robust to outliers
        self._feature_names = None
        self._feature_importances = None

    def _calculate_herg_descriptors(self, smiles_list):
        """Calculate hERG-specific molecular descriptors."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Invalid SMILES - use median values
                desc = [np.nan] * 25
            else:
                # Add hydrogens for accurate calculations
                mol = Chem.AddHs(mol)

                # Basic descriptors
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # hERG-specific: Basic nitrogen count (key predictor)
                basic_nitrogens = 0
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 7:  # Nitrogen
                        # Count potentially basic nitrogens
                        if atom.GetFormalCharge() >= 0:
                            basic_nitrogens += 1

                # Aromatic features (hERG pocket is hydrophobic/aromatic)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                desc = [
                    # Lipophilicity (crucial for hERG)
                    logP,
                    Descriptors.MolMR(mol),  # Molar refractivity
                    logP * mw / 100,  # Lipophilic efficiency

                    # Size and shape
                    mw,
                    Descriptors.HeavyAtomCount(mol),
                    tpsa,
                    Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),

                    # Aromatic character
                    aromatic_atoms,
                    aromatic_rings,
                    aromatic_atoms / max(1, Descriptors.HeavyAtomCount(mol)),  # Aromatic fraction
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),

                    # Basic nitrogen (most important for hERG)
                    basic_nitrogens,
                    basic_nitrogens * logP,  # Lipophilic basicity

                    # H-bonding
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),

                    # Charge and electronics
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),  # N,O,S count

                    # Complexity indices
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),

                    # Ring features
                    Lipinski.RingCount(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),

                    # Flexibility
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),  # Rigidity index
                ]

            descriptors.append(desc)

        self._feature_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
            'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex'
        ]

        return np.array(descriptors)

    def _impute_and_scale(self, X, fit=False):
        """Handle missing values and scale features."""
        X = np.array(X, dtype=float)

        # Impute NaN values with column median
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
        X = self._calculate_herg_descriptors(X_smiles)
        X = self._impute_and_scale(X, fit=True)
        y = np.array(y)

        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = np.array([class_weights[int(label)] for label in y])

        if HAS_XGBOOST:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            # GradientBoosting doesn't support sample_weight, use class_weight
            pos_weight = class_weights[1] / class_weights[0]
            # Approximate with duplicating minority class samples
            if pos_weight > 1:
                # Oversample positive class
                pos_indices = np.where(y == 1)[0]
                n_duplicate = int(pos_weight)
                if n_duplicate > 1:
                    dup_indices = np.tile(pos_indices, n_duplicate)[:len(pos_indices) * (n_duplicate - 1)]
                    X = np.vstack([X, X[dup_indices]])
                    y = np.hstack([y, y[dup_indices]])

            self.model.fit(X, y)

        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_herg_descriptors(X_smiles)
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
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin - likely non-toxic
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine - basic nitrogen
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen - likely non-toxic
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1'  # Synthetic basic lipophilic - likely toxic
    ]

    train_smiles = test_smiles * 15
    train_labels = [0, 1, 0, 1] * 15

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("XGBoost + hERG Descriptors Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:30]}... -> {p:.3f}")

    print("\nTop feature importances:")
    fi = predictor.get_feature_importance()
    for name, imp in zip(fi['features'][:8], fi['importances'][:8]):
        print(f"  {name}: {imp:.4f}")