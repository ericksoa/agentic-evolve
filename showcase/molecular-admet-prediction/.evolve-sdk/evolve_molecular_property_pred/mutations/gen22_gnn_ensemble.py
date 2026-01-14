"""
Gen22: GNN Ensemble Component

Parent: gen20c_ensemble (0.8894 test)

Mutation: Add Graph Neural Network to ensemble
- Top TDC models (MapLight, AttentiveFP) use GNNs
- GNNs learn molecular representations from graph structure
- Add GCN/GAT as 6th ensemble member

Hypothesis: GNN can capture structural patterns that fingerprints miss,
improving ensemble diversity and overall performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
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


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 1-118
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}


def one_hot_encoding(x, allowable_set):
    """One-hot encoding with unknown handling."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    """Get atom features as a vector."""
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    return features


def mol_to_graph(smiles):
    """Convert SMILES to PyG graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty graph for invalid molecules
        return Data(
            x=torch.zeros((1, 138), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edges (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class GNN(nn.Module):
    """Graph Neural Network for molecular property prediction."""

    def __init__(self, input_dim=138, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for mean+max pooling
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Graph convolutions
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global pooling (mean + max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP head
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HERGPredictor:
    """Ensemble with GNN component for improved performance."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Tree-based models (from gen20c)
        self.rf = RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
            random_state=random_state, n_jobs=1
        )

        if HAS_XGBOOST:
            self.xgb = xgb.XGBClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03,
                subsample=0.65, colsample_bytree=0.5, reg_alpha=0.4,
                reg_lambda=0.5, random_state=random_state,
                eval_metric='logloss', n_jobs=1
            )
        else:
            self.xgb = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.03,
                random_state=random_state
            )

        self.et = ExtraTreesClassifier(
            n_estimators=80, max_depth=5, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=random_state, n_jobs=1
        )

        self.svm = SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
            probability=True, random_state=random_state
        )

        # MLP config
        self.mlp_config = {'hidden_sizes': [128, 64, 32], 'dropout': 0.4, 'lr': 0.0006}
        self.mlp = None

        # GNN config
        self.gnn_config = {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.3, 'lr': 0.001}
        self.gnn = None
        self.n_epochs = 50
        self.batch_size = 32
        self.weight_decay = 2e-3

        # Ensemble weights (adjusted to include GNN)
        # Original gen20c: tree=0.75, svm=0.10, mlp=0.15
        # Now: tree=0.60, svm=0.08, mlp=0.12, gnn=0.20
        self.tree_weights = [0.20, 0.20, 0.20]  # RF, XGB, ET
        self.svm_weight = 0.08
        self.mlp_weight = 0.12
        self.gnn_weight = 0.20

        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._smiles_cache = None

    def _calculate_features(self, smiles_list):
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
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    def _create_graph_batch(self, smiles_list):
        """Convert SMILES list to batched graph."""
        graphs = [mol_to_graph(smi) for smi in smiles_list]
        return Batch.from_data_list(graphs)

    def _train_mlp(self, X, y):
        torch.manual_seed(42)
        np.random.seed(42)

        input_size = X.shape[1]
        mlp = MLP(input_size, self.mlp_config['hidden_sizes'], self.mlp_config['dropout']).to(self.device)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(mlp.parameters(), lr=self.mlp_config['lr'], weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        mlp.train()
        for epoch in range(self.n_epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        mlp.eval()
        return mlp

    def _train_gnn(self, smiles_list, y):
        """Train GNN on molecular graphs."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Get input dimension from first graph
        sample_graph = mol_to_graph(smiles_list[0])
        input_dim = sample_graph.x.shape[1]

        gnn = GNN(
            input_dim=input_dim,
            hidden_dim=self.gnn_config['hidden_dim'],
            num_layers=self.gnn_config['num_layers'],
            dropout=self.gnn_config['dropout']
        ).to(self.device)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(gnn.parameters(), lr=self.gnn_config['lr'], weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        # Create mini-batches
        indices = np.arange(len(smiles_list))

        gnn.train()
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_smiles = [smiles_list[i] for i in batch_idx]
                batch_y = torch.tensor(y[batch_idx], dtype=torch.float32).unsqueeze(1).to(self.device)

                batch = self._create_graph_batch(batch_smiles).to(self.device)

                optimizer.zero_grad()
                outputs = gnn(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

        gnn.eval()
        return gnn

    def fit(self, X_smiles, y):
        # Store SMILES for GNN
        self._smiles_cache = list(X_smiles)

        # Calculate fingerprint features
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=True)
        y = np.array(y)

        # Train tree models
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.svm.fit(X, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X, y)

        # Train MLP
        self.mlp = self._train_mlp(X, y)

        # Train GNN
        self.gnn = self._train_gnn(self._smiles_cache, y)

        # Feature importance (from tree models only)
        self._feature_importances = (
            self.tree_weights[0] * self.rf.feature_importances_ +
            self.tree_weights[1] * self.xgb.feature_importances_ +
            self.tree_weights[2] * self.et.feature_importances_
        ) / sum(self.tree_weights)
        return self

    def predict_proba(self, X_smiles):
        X_smiles = list(X_smiles)
        X = self._calculate_features(X_smiles)
        X = self._preprocess(X, fit=False)

        # Tree model predictions
        proba_rf = self.rf.predict_proba(X)[:, 1]
        proba_xgb = self.xgb.predict_proba(X)[:, 1]
        proba_et = self.et.predict_proba(X)[:, 1]
        proba_svm = self.svm.predict_proba(X)[:, 1]

        # MLP prediction
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(X_tensor)
            proba_mlp = torch.sigmoid(logits).cpu().numpy().flatten()

        # GNN prediction
        self.gnn.eval()
        proba_gnn = []
        with torch.no_grad():
            for start in range(0, len(X_smiles), self.batch_size):
                batch_smiles = X_smiles[start:start + self.batch_size]
                batch = self._create_graph_batch(batch_smiles).to(self.device)
                logits = self.gnn(batch.x, batch.edge_index, batch.batch)
                proba_gnn.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        proba_gnn = np.array(proba_gnn)

        # Ensemble
        return (self.tree_weights[0] * proba_rf +
                self.tree_weights[1] * proba_xgb +
                self.tree_weights[2] * proba_et +
                self.svm_weight * proba_svm +
                self.mlp_weight * proba_mlp +
                self.gnn_weight * proba_gnn)

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        if self._feature_importances is None:
            return None
        importance_dict = dict(zip(self._feature_names, self._feature_importances.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {
            'features': [x[0] for x in sorted_importance[:30]],
            'importances': [x[1] for x in sorted_importance[:30]]
        }
