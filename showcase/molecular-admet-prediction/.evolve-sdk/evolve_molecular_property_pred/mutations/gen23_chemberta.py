"""
Gen23: ChemBERTa Transformer Ensemble

Parent: gen20c_ensemble (0.8894 test)

Mutation: Add ChemBERTa embeddings as additional features
- Pre-trained on 77M SMILES from ZINC
- Captures broader chemical knowledge
- Use [CLS] token embedding (768-dim)

Hypothesis: ChemBERTa embeddings provide complementary
representations to fingerprints, improving ensemble diversity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
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
    """Ensemble with ChemBERTa embeddings."""

    def __init__(self, random_state=42):
        self.random_state = random_state

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load ChemBERTa for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.chemberta = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.chemberta.eval()
        self.chemberta.to(self.device)

        # Freeze ChemBERTa weights
        for param in self.chemberta.parameters():
            param.requires_grad = False

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

        # Two MLPs: one for fingerprints, one for ChemBERTa
        self.mlp_fp_config = {'hidden_sizes': [128, 64, 32], 'dropout': 0.4, 'lr': 0.0006}
        self.mlp_bert_config = {'hidden_sizes': [256, 128, 64], 'dropout': 0.4, 'lr': 0.0004}
        self.mlp_fp = None
        self.mlp_bert = None
        self.n_epochs = 40
        self.batch_size = 32
        self.weight_decay = 2e-3

        # Weights: 70% trees, 8% SVM, 12% MLP-FP, 10% MLP-BERT
        self.tree_weights = [0.23, 0.24, 0.23]  # RF, XGB, ET = 70%
        self.svm_weight = 0.08
        self.mlp_fp_weight = 0.12
        self.mlp_bert_weight = 0.10

        self.scaler_fp = RobustScaler()
        self.scaler_bert = RobustScaler()
        self._feature_names = None
        self._feature_importances = None
        self._bert_embeddings_cache = {}

    def _get_chemberta_embedding(self, smiles_list):
        """Get ChemBERTa [CLS] embeddings for SMILES."""
        embeddings = []
        batch_size = 16

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Get embeddings
                outputs = self.chemberta(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(embeddings)

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

    def _preprocess_fp(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X = self.scaler_fp.fit_transform(X)
        else:
            X = self.scaler_fp.transform(X)
        return X

    def _preprocess_bert(self, X, fit=False):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        if fit:
            X = self.scaler_bert.fit_transform(X)
        else:
            X = self.scaler_bert.transform(X)
        return X

    def _train_mlp(self, X, y, config, input_size):
        torch.manual_seed(42)
        np.random.seed(42)

        mlp = MLP(input_size, config['hidden_sizes'], config['dropout']).to(self.device)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(mlp.parameters(), lr=config['lr'], weight_decay=self.weight_decay)
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

    def fit(self, X_smiles, y):
        X_smiles = list(X_smiles)

        # Calculate fingerprint features
        X_fp = self._calculate_features(X_smiles)
        X_fp = self._preprocess_fp(X_fp, fit=True)

        # Get ChemBERTa embeddings
        X_bert = self._get_chemberta_embedding(X_smiles)
        X_bert = self._preprocess_bert(X_bert, fit=True)

        y = np.array(y)

        # Train tree models on fingerprints
        self.rf.fit(X_fp, y)
        self.et.fit(X_fp, y)
        self.svm.fit(X_fp, y)

        if HAS_XGBOOST:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            self.xgb.fit(X_fp, y, sample_weight=sample_weights)
        else:
            self.xgb.fit(X_fp, y)

        # Train MLP on fingerprints
        self.mlp_fp = self._train_mlp(X_fp, y, self.mlp_fp_config, X_fp.shape[1])

        # Train MLP on ChemBERTa embeddings
        self.mlp_bert = self._train_mlp(X_bert, y, self.mlp_bert_config, X_bert.shape[1])

        self._feature_importances = (
            self.tree_weights[0] * self.rf.feature_importances_ +
            self.tree_weights[1] * self.xgb.feature_importances_ +
            self.tree_weights[2] * self.et.feature_importances_
        ) / sum(self.tree_weights)
        return self

    def predict_proba(self, X_smiles):
        X_smiles = list(X_smiles)

        # Fingerprint features
        X_fp = self._calculate_features(X_smiles)
        X_fp = self._preprocess_fp(X_fp, fit=False)

        # ChemBERTa embeddings
        X_bert = self._get_chemberta_embedding(X_smiles)
        X_bert = self._preprocess_bert(X_bert, fit=False)

        # Tree predictions
        proba_rf = self.rf.predict_proba(X_fp)[:, 1]
        proba_xgb = self.xgb.predict_proba(X_fp)[:, 1]
        proba_et = self.et.predict_proba(X_fp)[:, 1]
        proba_svm = self.svm.predict_proba(X_fp)[:, 1]

        # MLP-FP prediction
        X_fp_tensor = torch.tensor(X_fp, dtype=torch.float32).to(self.device)
        self.mlp_fp.eval()
        with torch.no_grad():
            logits = self.mlp_fp(X_fp_tensor)
            proba_mlp_fp = torch.sigmoid(logits).cpu().numpy().flatten()

        # MLP-BERT prediction
        X_bert_tensor = torch.tensor(X_bert, dtype=torch.float32).to(self.device)
        self.mlp_bert.eval()
        with torch.no_grad():
            logits = self.mlp_bert(X_bert_tensor)
            proba_mlp_bert = torch.sigmoid(logits).cpu().numpy().flatten()

        return (self.tree_weights[0] * proba_rf +
                self.tree_weights[1] * proba_xgb +
                self.tree_weights[2] * proba_et +
                self.svm_weight * proba_svm +
                self.mlp_fp_weight * proba_mlp_fp +
                self.mlp_bert_weight * proba_mlp_bert)

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
