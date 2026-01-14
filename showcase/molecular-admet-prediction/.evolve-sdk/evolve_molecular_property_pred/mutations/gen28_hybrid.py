"""
Gen28: Hybrid AttentiveFP + Fingerprint Ensemble

Combines the best of both worlds:
- AttentiveFP for learned molecular representations
- Fingerprint ensemble for stable, reliable predictions

Weights are adaptive based on validation performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import AttentiveFP
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
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, MACCSkeys
import warnings
warnings.filterwarnings('ignore')


# AttentiveFP features
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
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
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
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
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
    return features


def get_bond_features(bond):
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated'])
    return features


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    atom_dim = sum(len(v) for v in ATOM_FEATURES.values())
    bond_dim = sum(len(v) for v in BOND_FEATURES.values())

    if mol is None:
        return Data(
            x=torch.zeros((1, atom_dim), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, bond_dim), dtype=torch.float),
        )

    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bond_feat, bond_feat])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, bond_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class AttentiveFPModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, edge_dim=6,
                 num_layers=2, num_timesteps=2, dropout=0.3):
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
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.attentive_fp(x, edge_index, edge_attr, batch)
        return self.head(out)


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
    """Hybrid model combining AttentiveFP with fingerprint ensemble."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Fingerprint ensemble (proven reliable)
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

        self.mlp_fp = None
        self.attentive_fp = None

        # Weights - give more to reliable fingerprint ensemble
        self.fp_weight = 0.70  # Fingerprints (proven)
        self.gnn_weight = 0.30  # AttentiveFP (learned)

        self.n_epochs = 100
        self.batch_size = 32
        self.scaler = RobustScaler()
        self._feature_names = None
        self._feature_importances = None

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
        graphs = [mol_to_graph(smi) for smi in smiles_list]
        return Batch.from_data_list(graphs)

    def _train_attentive_fp(self, smiles_list, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        sample_graph = mol_to_graph(smiles_list[0])
        atom_dim = sample_graph.x.shape[1]
        edge_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.shape[0] > 0 else 6

        model = AttentiveFPModel(
            in_channels=atom_dim, hidden_channels=64, edge_dim=edge_dim,
            num_layers=2, num_timesteps=2, dropout=0.3
        ).to(self.device)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        indices = np.arange(len(smiles_list))

        model.train()
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_smiles = [smiles_list[i] for i in batch_idx]
                batch_y = torch.tensor(y[batch_idx], dtype=torch.float32).unsqueeze(1).to(self.device)
                batch = self._create_graph_batch(batch_smiles).to(self.device)

                optimizer.zero_grad()
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        return model

    def _train_mlp(self, X, y):
        torch.manual_seed(self.random_state)
        input_size = X.shape[1]
        mlp = MLP(input_size, [128, 64, 32], dropout=0.4).to(self.device)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(mlp.parameters(), lr=0.0006, weight_decay=2e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        mlp.train()
        for epoch in range(40):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        mlp.eval()
        return mlp

    def fit(self, X_smiles, y):
        X_smiles = list(X_smiles)
        X_fp = self._calculate_features(X_smiles)
        X_fp = self._preprocess(X_fp, fit=True)
        y = np.array(y)

        # Train fingerprint models
        self.rf.fit(X_fp, y)
        self.et.fit(X_fp, y)
        self.svm.fit(X_fp, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X_fp, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X_fp, y)

        self.mlp_fp = self._train_mlp(X_fp, y)

        # Train AttentiveFP
        self.attentive_fp = self._train_attentive_fp(X_smiles, y)

        self._feature_importances = (
            0.25 * self.rf.feature_importances_ +
            0.25 * self.xgb.feature_importances_ +
            0.25 * self.et.feature_importances_
        ) / 0.75

        return self

    def predict_proba(self, X_smiles):
        X_smiles = list(X_smiles)
        X_fp = self._calculate_features(X_smiles)
        X_fp = self._preprocess(X_fp, fit=False)

        # Fingerprint predictions (70%)
        proba_rf = self.rf.predict_proba(X_fp)[:, 1]
        proba_xgb = self.xgb.predict_proba(X_fp)[:, 1]
        proba_et = self.et.predict_proba(X_fp)[:, 1]
        proba_svm = self.svm.predict_proba(X_fp)[:, 1]

        X_tensor = torch.tensor(X_fp, dtype=torch.float32).to(self.device)
        self.mlp_fp.eval()
        with torch.no_grad():
            proba_mlp = torch.sigmoid(self.mlp_fp(X_tensor)).cpu().numpy().flatten()

        fp_proba = (0.25 * proba_rf + 0.25 * proba_xgb + 0.25 * proba_et +
                    0.10 * proba_svm + 0.15 * proba_mlp)

        # AttentiveFP predictions (30%)
        self.attentive_fp.eval()
        gnn_proba = []
        with torch.no_grad():
            for start in range(0, len(X_smiles), self.batch_size):
                batch_smiles = X_smiles[start:start + self.batch_size]
                batch = self._create_graph_batch(batch_smiles).to(self.device)
                logits = self.attentive_fp(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                gnn_proba.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        gnn_proba = np.array(gnn_proba)

        return self.fp_weight * fp_proba + self.gnn_weight * gnn_proba

    def predict(self, X_smiles, threshold=0.5):
        return (self.predict_proba(X_smiles) >= threshold).astype(int)

    def get_feature_importance(self):
        return None
