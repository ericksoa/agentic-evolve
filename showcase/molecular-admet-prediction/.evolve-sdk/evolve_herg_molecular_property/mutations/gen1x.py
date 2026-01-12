"""
Hybrid XGBoost with Extended Feature Engineering
Fitness improvement target: 0.85+

This hybrid solution combines the best features from multiple parent solutions:
- XGBoost architecture from gen0_b (best performer, 0.8434 fitness)
- Extended feature set combining hERG-specific descriptors + Morgan fingerprints
- Enhanced pharmacophore patterns from gen0_c
- Robust preprocessing and class balancing

Key improvements:
1. Multi-modal features: hERG descriptors + Morgan fingerprints + pharmacophores
2. Enhanced basic nitrogen detection with formal charge consideration
3. Optimized XGBoost hyperparameters based on feature dimensionality
4. Advanced class balancing with sample weighting
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
    hERG toxicity predictor using XGBoost with extended molecular features.

    Combines multiple molecular representations optimized for hERG prediction:
    - hERG-specific physicochemical descriptors (from gen0_b)
    - Morgan fingerprints for substructure capture (from gen0_c)
    - Enhanced pharmacophore patterns for drug-like interactions
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        if HAS_XGBOOST:
            # Optimized hyperparameters for extended feature set
            self.model = xgb.XGBClassifier(
                n_estimators=200,  # Increased for more complex features
                max_depth=8,       # Deeper trees for feature interactions
                learning_rate=0.08,  # Slightly lower for stability
                subsample=0.85,
                colsample_bytree=0.7,  # Lower for high-dimensional features
                gamma=1,           # Regularization for overfitting control
                reg_alpha=0.1,     # L1 regularization
                reg_lambda=1,      # L2 regularization
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.85,
                random_state=random_state
            )

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

    def _calculate_herg_descriptors(self, smiles_list):
        """Calculate comprehensive hERG-specific molecular descriptors."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                desc = [np.nan] * 28  # Extended descriptor count
            else:
                mol = Chem.AddHs(mol)

                # Basic descriptors
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                # Enhanced basic nitrogen detection (critical for hERG)
                basic_nitrogens = 0
                total_nitrogens = 0
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 7:  # Nitrogen
                        total_nitrogens += 1
                        # More sophisticated basicity detection
                        if atom.GetFormalCharge() >= 0:
                            # Check hybridization and environment
                            if atom.GetHybridization() in [Chem.HybridizationType.SP3, Chem.HybridizationType.SP2]:
                                basic_nitrogens += 1

                # Aromatic features
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)

                desc = [
                    # Core lipophilicity features
                    logP,
                    Descriptors.MolMR(mol),
                    logP * mw / 100,  # Lipophilic efficiency

                    # Size and shape
                    mw,
                    Descriptors.HeavyAtomCount(mol),
                    tpsa,
                    Descriptors.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),

                    # Aromatic character (enhanced)
                    aromatic_atoms,
                    aromatic_rings,
                    aromatic_atoms / max(1, Descriptors.HeavyAtomCount(mol)),
                    Descriptors.NumAromaticCarbocycles(mol),
                    Descriptors.NumAromaticHeterocycles(mol),

                    # Basic nitrogen features (enhanced)
                    basic_nitrogens,
                    basic_nitrogens * logP,  # Lipophilic basicity
                    total_nitrogens,
                    basic_nitrogens / max(1, total_nitrogens),  # Basicity fraction

                    # H-bonding
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),

                    # Electronic features
                    Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),

                    # Complexity indices
                    Descriptors.BertzCT(mol),
                    Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),

                    # Ring and flexibility
                    Lipinski.RingCount(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),  # Rigidity

                    # Additional hERG-relevant features
                    tpsa / mw if mw > 0 else 0,  # Polar surface efficiency
                ]

            descriptors.append(desc)

        return np.array(descriptors)

    def _calculate_morgan_fingerprints(self, smiles_list, radius=2, n_bits=512):
        """Calculate Morgan fingerprints (from gen0_c but optimized size)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp = np.zeros(n_bits)
            else:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fp = np.array(morgan_fp)
            fingerprints.append(fp)

        return np.array(fingerprints)

    def _calculate_pharmacophore_features(self, smiles_list):
        """Calculate enhanced pharmacophore-like features."""
        from rdkit import Chem

        all_features = []

        # Enhanced pattern list focusing on hERG-relevant substructures
        patterns = [
            '[#7+]',  # Positive nitrogen (key hERG blocker)
            '[#7H2]',  # Primary amine
            '[#7H]',   # Secondary amine
            '[#7]([#6])[#6]',  # Tertiary amine
            '[#7]c1ccccc1',  # Aromatic nitrogen
            'c1ccccc1',  # Benzene ring
            'c1ccc2ccccc2c1',  # Naphthalene
            '[OH]',   # Hydroxyl
            '[#16]',  # Sulfur
            '[F]',    # Fluorine
            '[Cl]',   # Chlorine
            '[#7]~[#6]~[#6]~[#7]',  # N-C-C-N pattern
            'c1ccc([#7])cc1',  # Aniline-like
            '[#6]=[#6]',  # Double bond
            'O=C',    # Carbonyl
            '[#7]C(=O)',  # Amide nitrogen
            'S(=O)(=O)',  # Sulfone
            'C#N',    # Nitrile
        ]

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            features = []

            if mol is None:
                features = [0] * len(patterns)
            else:
                for pattern in patterns:
                    try:
                        matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                        features.append(matches)
                    except:
                        features.append(0)

            all_features.append(features)

        return np.array(all_features)

    def _calculate_extended_features(self, smiles_list):
        """Combine all feature types into extended feature matrix."""
        # Get all feature types
        herg_desc = self._calculate_herg_descriptors(smiles_list)
        morgan_fp = self._calculate_morgan_fingerprints(smiles_list)
        pharmaco = self._calculate_pharmacophore_features(smiles_list)

        # Combine features
        extended_features = np.hstack([herg_desc, morgan_fp, pharmaco])

        # Create feature names for interpretability
        herg_names = [
            'MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
            'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
            'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
            'TotalNitrogens', 'BasicityFraction', 'NumHDonors', 'NumHAcceptors',
            'NumHeteroatoms', 'NOSCount', 'BertzCT', 'Chi0', 'Chi1', 'RingCount',
            'NumBridgeheadAtoms', 'RigidityIndex', 'PolarSurfaceEfficiency'
        ]
        morgan_names = [f'Morgan_{i}' for i in range(512)]
        pharmaco_names = [f'Pharmaco_{i}' for i in range(18)]

        self._feature_names = herg_names + morgan_names + pharmaco_names

        return extended_features

    def _impute_and_scale(self, X, fit=False):
        """Enhanced preprocessing with robust handling."""
        X = np.array(X, dtype=float)

        # Impute NaN values with column median
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if not np.isnan(median_val) else 0

        # Handle infinite values
        X = np.clip(X, -1e6, 1e6)

        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def fit(self, X_smiles, y):
        """Train the model on SMILES strings and labels."""
        X = self._calculate_extended_features(X_smiles)
        X = self._impute_and_scale(X, fit=True)
        y = np.array(y)

        # Enhanced class balancing
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = np.array([class_weights[int(label)] for label in y])

        if HAS_XGBOOST:
            # Use scale_pos_weight for better class balancing in XGBoost
            pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            self.model.set_params(scale_pos_weight=pos_weight)
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            # Handle class imbalance for sklearn
            if len(class_weights) > 1:
                pos_weight = class_weights[1] / class_weights[0]
                if pos_weight > 1:
                    pos_indices = np.where(y == 1)[0]
                    n_duplicate = min(int(pos_weight), 3)  # Cap duplication
                    if n_duplicate > 1:
                        dup_indices = np.tile(pos_indices, n_duplicate)[:len(pos_indices) * (n_duplicate - 1)]
                        X = np.vstack([X, X[dup_indices]])
                        y = np.hstack([y, y[dup_indices]])

            self.model.fit(X, y)

        self._feature_importances = self.model.feature_importances_
        return self

    def predict_proba(self, X_smiles):
        """Predict probability of hERG blocking for each molecule."""
        X = self._calculate_extended_features(X_smiles)
        X = self._impute_and_scale(X, fit=False)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict(self, X_smiles, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X_smiles)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance with enhanced interpretability."""
        if self._feature_importances is None or self._feature_names is None:
            return None

        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            'features': [x[0] for x in sorted_importance],
            'importances': [x[1] for x in sorted_importance]
        }


if __name__ == '__main__':
    # Comprehensive test
    predictor = HERGPredictor()

    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin - likely non-toxic
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine - basic nitrogen
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen - likely non-toxic
        'CCN(CC)CCCC(C#N)(c1ccccc1)c1ccc(F)cc1',  # Synthetic basic lipophilic - likely toxic
        'CC1=CC(=C(C=C1)C)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C',  # Complex drug-like molecule
    ]

    train_smiles = test_smiles * 20
    train_labels = [0, 1, 0, 1, 1] * 20

    predictor.fit(train_smiles, train_labels)
    proba = predictor.predict_proba(test_smiles)

    print("Hybrid XGBoost + Extended Features Test predictions:")
    for smi, p in zip(test_smiles, proba):
        print(f"  {smi[:40]}... -> {p:.3f}")

    print("\nTop feature importances:")
    fi = predictor.get_feature_importance()
    for name, imp in zip(fi['features'][:10], fi['importances'][:10]):
        print(f"  {name}: {imp:.4f}")