"""
Gen27: AttentiveFP - State-of-the-art Graph Attention Network

This implements AttentiveFP, the architecture used by top TDC leaderboard models.
AttentiveFP uses graph attention with a GRU-based readout for molecular property prediction.

Reference: "Pushing the Boundaries of Molecular Representation for Drug Discovery
with the Graph Attention Mechanism" (Xiong et al., 2019)

Target: Beat MapLight+GNN (0.880 AUROC) on TDC hERG benchmark
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')


# Atom features for AttentiveFP
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

# Bond features
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [0, 1, 2, 3, 4, 5],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
}


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    """Get comprehensive atom features for AttentiveFP."""
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
    features += one_hot(atom.IsInRing(), ATOM_FEATURES['is_in_ring'])
    return features


def get_bond_features(bond):
    """Get bond features for AttentiveFP."""
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo'])
    features += one_hot(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated'])
    features += one_hot(bond.IsInRing(), BOND_FEATURES['is_in_ring'])
    return features


def mol_to_graph(smiles):
    """Convert SMILES to PyG graph with atom and bond features."""
    mol = Chem.MolFromSmiles(smiles)

    # Calculate feature dimensions
    atom_dim = (len(ATOM_FEATURES['atomic_num']) + len(ATOM_FEATURES['degree']) +
                len(ATOM_FEATURES['formal_charge']) + len(ATOM_FEATURES['chiral_tag']) +
                len(ATOM_FEATURES['num_Hs']) + len(ATOM_FEATURES['hybridization']) +
                len(ATOM_FEATURES['is_aromatic']) + len(ATOM_FEATURES['is_in_ring']))

    bond_dim = (len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['stereo']) +
                len(BOND_FEATURES['is_conjugated']) + len(BOND_FEATURES['is_in_ring']))

    if mol is None:
        # Return dummy graph for invalid molecules
        return Data(
            x=torch.zeros((1, atom_dim), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, bond_dim), dtype=torch.float),
        )

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features and edge indices
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, bond_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class AttentiveFPModel(nn.Module):
    """AttentiveFP model for molecular property prediction."""

    def __init__(self, in_channels, hidden_channels=200, edge_dim=14,
                 num_layers=2, num_timesteps=2, dropout=0.2):
        super().__init__()

        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

        # MLP head for classification
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # AttentiveFP returns graph-level representation
        out = self.attentive_fp(x, edge_index, edge_attr, batch)
        return self.head(out)


class HERGPredictor:
    """AttentiveFP-based hERG predictor."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model configuration
        self.hidden_channels = 200
        self.num_layers = 2
        self.num_timesteps = 2
        self.dropout = 0.2
        self.n_epochs = 100
        self.batch_size = 64
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.patience = 15  # Early stopping

        self.model = None
        self.atom_dim = None
        self.edge_dim = None

    def _create_graph_batch(self, smiles_list):
        """Convert SMILES list to batched graph."""
        graphs = [mol_to_graph(smi) for smi in smiles_list]
        return Batch.from_data_list(graphs)

    def fit(self, X_smiles, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_smiles = list(X_smiles)
        y = np.array(y)

        # Get feature dimensions from first molecule
        sample_graph = mol_to_graph(X_smiles[0])
        self.atom_dim = sample_graph.x.shape[1]
        self.edge_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.shape[0] > 0 else 14

        # Initialize model
        self.model = AttentiveFPModel(
            in_channels=self.atom_dim,
            hidden_channels=self.hidden_channels,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout
        ).to(self.device)

        # Loss with class weighting
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Training loop with mini-batches
        indices = np.arange(len(X_smiles))
        best_auc = 0
        patience_counter = 0

        self.model.train()
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            epoch_loss = 0

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_smiles = [X_smiles[i] for i in batch_idx]
                batch_y = torch.tensor(y[batch_idx], dtype=torch.float32).unsqueeze(1).to(self.device)

                batch = self._create_graph_batch(batch_smiles).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            # Validation (on training data for now - proper validation should use held-out set)
            self.model.eval()
            with torch.no_grad():
                all_proba = []
                for start in range(0, len(X_smiles), self.batch_size):
                    batch_smiles = X_smiles[start:start + self.batch_size]
                    batch = self._create_graph_batch(batch_smiles).to(self.device)
                    logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    proba = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_proba.extend(proba)

                train_auc = roc_auc_score(y, all_proba)

            scheduler.step(train_auc)

            # Early stopping
            if train_auc > best_auc:
                best_auc = train_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

            self.model.train()

        # Load best model
        self.model.load_state_dict(best_state)
        self.model.eval()

        return self

    def predict_proba(self, X_smiles):
        X_smiles = list(X_smiles)

        self.model.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X_smiles), self.batch_size):
                batch_smiles = X_smiles[start:start + self.batch_size]
                batch = self._create_graph_batch(batch_smiles).to(self.device)
                logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                proba = torch.sigmoid(logits).cpu().numpy().flatten()
                all_proba.extend(proba)

        return np.array(all_proba)

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        # AttentiveFP doesn't have direct feature importance
        return None
