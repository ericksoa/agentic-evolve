"""
Gen18b: Neural Network Ensemble (Evolved)

Parent: gen18_neural (0.867 ROC-AUC)

Mutation: Hyperparameter evolution for small dataset
- Fewer epochs (30 vs 50) to reduce overfitting
- Higher dropout (0.4-0.5) for regularization
- Smaller networks for limited data
- Stronger weight decay
- Higher XGBoost weight in ensemble

Hypothesis: With only ~500 training samples, smaller/regularized
networks should generalize better.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys
import warnings
warnings.filterwarnings('ignore')


class MLP(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, input_size, hidden_sizes, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HERGPredictor:
    """Neural network ensemble for hERG toxicity prediction (evolved)."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # EVOLVED: Smaller networks, higher dropout for small dataset
        self.mlp_configs = [
            {'hidden_sizes': [128, 64], 'dropout': 0.5, 'lr': 0.001},
            {'hidden_sizes': [256, 128], 'dropout': 0.4, 'lr': 0.0008},
            {'hidden_sizes': [64, 32], 'dropout': 0.5, 'lr': 0.002},
        ]
        self.n_epochs = 30  # EVOLVED: fewer epochs
        self.batch_size = 32
        self.weight_decay = 5e-4  # EVOLVED: stronger regularization

        # EVOLVED: Give more weight to XGBoost (proven performer)
        self.weights = [0.20, 0.20, 0.20, 0.40]  # 3 MLPs + XGBoost

        self.mlps = None

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.65,
                colsample_bytree=0.5,
                reg_alpha=0.4,
                reg_lambda=0.5,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03,
                random_state=random_state
            )

        self.scaler = RobustScaler()
        self._feature_names = None

    def _calculate_features(self, smiles_list):
        """Calculate compact fingerprints + hERG descriptors."""
        all_features = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features = np.zeros(512 + 167 + 25)
            else:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=512)
                morgan_bits = np.array(morgan_fp)

                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_bits = np.array(maccs)

                mol_h = Chem.AddHs(mol)
                logP = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)

                basic_nitrogens = sum(1 for atom in mol_h.GetAtoms()
                                      if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0)
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                heavy_atoms = Descriptors.HeavyAtomCount(mol)

                descriptors = [
                    logP, Descriptors.MolMR(mol), logP * mw / 100, mw, heavy_atoms,
                    tpsa, Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                    aromatic_atoms, aromatic_rings, aromatic_atoms / max(1, heavy_atoms),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
                    basic_nitrogens, basic_nitrogens * logP, Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol), Descriptors.NumHeteroatoms(mol),
                    len([a for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16]]),
                    Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
                    Lipinski.RingCount(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    mw / max(1, Descriptors.NumRotatableBonds(mol) + 1),
                ]

                features = np.concatenate([morgan_bits, maccs_bits, descriptors])
            all_features.append(features)

        self._feature_names = (
            [f'Morgan3_{i}' for i in range(512)] +
            [f'MACCS_{i}' for i in range(167)] +
            ['MolLogP', 'MolMR', 'LipophilicEfficiency', 'MolWt', 'HeavyAtomCount',
             'TPSA', 'NumRotatableBonds', 'FractionCSP3', 'AromaticAtoms',
             'NumAromaticRings', 'AromaticFraction', 'NumAromaticCarbocycles',
             'NumAromaticHeterocycles', 'BasicNitrogens', 'LipophilicBasicity',
             'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NOSCount',
             'BertzCT', 'Chi0', 'Chi1', 'RingCount', 'NumBridgeheadAtoms', 'RigidityIndex']
        )

        return np.array(all_features)

    def _preprocess(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X[:, -25:] = self.scaler.fit_transform(X[:, -25:])
        else:
            X[:, -25:] = self.scaler.transform(X[:, -25:])
        return X

    def _train_mlp(self, X, y, config):
        """Train a single MLP."""
        input_size = X.shape[1]
        mlp = MLP(input_size, config['hidden_sizes'], config['dropout']).to(self.device)

        # Compute class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(mlp.parameters(), lr=config['lr'], weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Create data loader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        mlp.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step(epoch_loss)

        mlp.eval()
        return mlp

    def fit(self, X_smiles, y):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Train MLPs with different architectures
        self.mlps = []
        for config in self.mlp_configs:
            mlp = self._train_mlp(X, y, config)
            self.mlps.append(mlp)

        # Train XGBoost
        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        return self

    def predict_proba(self, X_smiles):
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Get predictions from each MLP
        mlp_probas = []
        for mlp in self.mlps:
            mlp.eval()
            with torch.no_grad():
                logits = mlp(X_tensor)
                proba = torch.sigmoid(logits).cpu().numpy().flatten()
            mlp_probas.append(proba)

        # Get XGBoost predictions
        xgb_proba = self.xgb.predict_proba(X)[:, 1]

        # Weighted ensemble
        ensemble_proba = (
            self.weights[0] * mlp_probas[0] +
            self.weights[1] * mlp_probas[1] +
            self.weights[2] * mlp_probas[2] +
            self.weights[3] * xgb_proba
        )

        return ensemble_proba

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        """Return XGBoost feature importance."""
        if self._feature_names is None:
            return None
        importance_dict = dict(zip(self._feature_names, self.xgb.feature_importances_.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
