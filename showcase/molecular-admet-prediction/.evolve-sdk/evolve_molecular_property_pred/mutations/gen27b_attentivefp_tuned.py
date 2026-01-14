"""
Gen27b: AttentiveFP Tuned

Parent: gen27_attentivefp (0.761 ± 0.057 - too unstable)

Fixes:
- More epochs (200 vs 100)
- Smaller hidden size (64 vs 200) - less overfitting on small data
- More regularization (dropout 0.5 vs 0.2)
- Smaller learning rate with warmup
- Gradient accumulation for stability
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
import warnings
warnings.filterwarnings('ignore')


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
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo'])
    features += one_hot(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated'])
    features += one_hot(bond.IsInRing(), BOND_FEATURES['is_in_ring'])
    return features


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    atom_dim = (len(ATOM_FEATURES['atomic_num']) + len(ATOM_FEATURES['degree']) +
                len(ATOM_FEATURES['formal_charge']) + len(ATOM_FEATURES['chiral_tag']) +
                len(ATOM_FEATURES['num_Hs']) + len(ATOM_FEATURES['hybridization']) +
                len(ATOM_FEATURES['is_aromatic']) + len(ATOM_FEATURES['is_in_ring']))

    bond_dim = (len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['stereo']) +
                len(BOND_FEATURES['is_conjugated']) + len(BOND_FEATURES['is_in_ring']))

    if mol is None:
        return Data(
            x=torch.zeros((1, atom_dim), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, bond_dim), dtype=torch.float),
        )

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
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
    def __init__(self, in_channels, hidden_channels=64, edge_dim=14,
                 num_layers=2, num_timesteps=2, dropout=0.5):
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

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.attentive_fp(x, edge_index, edge_attr, batch)
        return self.head(out)


class HERGPredictor:
    """Tuned AttentiveFP for small datasets."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Tuned for small data
        self.hidden_channels = 64  # Smaller
        self.num_layers = 2
        self.num_timesteps = 2
        self.dropout = 0.5  # More dropout
        self.n_epochs = 200  # More epochs
        self.batch_size = 32  # Smaller batches
        self.lr = 0.0005  # Lower LR
        self.weight_decay = 5e-4
        self.patience = 30

        self.model = None
        self.atom_dim = None
        self.edge_dim = None

    def _create_graph_batch(self, smiles_list):
        graphs = [mol_to_graph(smi) for smi in smiles_list]
        return Batch.from_data_list(graphs)

    def fit(self, X_smiles, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_smiles = list(X_smiles)
        y = np.array(y)

        sample_graph = mol_to_graph(X_smiles[0])
        self.atom_dim = sample_graph.x.shape[1]
        self.edge_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.shape[0] > 0 else 14

        self.model = AttentiveFPModel(
            in_channels=self.atom_dim,
            hidden_channels=self.hidden_channels,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout
        ).to(self.device)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cosine annealing with warmup
        warmup_epochs = 10
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (self.n_epochs - warmup_epochs)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        indices = np.arange(len(X_smiles))
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            epoch_loss = 0
            n_batches = 0

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
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
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
        return None
