"""
Pre-trained Molecular Embeddings for hERG Prediction

Extracts molecular embeddings using pre-trained transformer models:
- ChemBERTa-2: RoBERTa-style encoder trained on 77M molecules
- MolBERT: BERT-style with chemistry-aware tokenization (more data-efficient)

These embeddings capture learned molecular representations that
complement traditional fingerprints and descriptors.
"""

import numpy as np
from typing import List, Optional, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

# Cache models for efficiency
_model_cache: Dict[str, tuple] = {}


def get_chemberta_model(model_name: str = "DeepChem/ChemBERTa-77M-MTR"):
    """
    Load ChemBERTa model and tokenizer (cached).

    Models available:
    - DeepChem/ChemBERTa-77M-MTR: Multi-task regression pre-trained
    - DeepChem/ChemBERTa-77M-MLM: Masked language model pre-trained
    - seyonec/ChemBERTa-zinc-base-v1: Trained on ZINC dataset

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (tokenizer, model)
    """
    global _model_cache

    if model_name not in _model_cache:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        _model_cache[model_name] = (tokenizer, model, device)
        print(f"Model loaded on {device}")

    return _model_cache[model_name]


def get_smiles_embedding(
    smiles: str,
    model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    pooling: str = "mean"
) -> np.ndarray:
    """
    Get embedding for a single SMILES string.

    Args:
        smiles: SMILES string of the molecule
        model_name: Pre-trained model to use
        pooling: How to pool token embeddings ("mean", "cls", "max")

    Returns:
        Numpy array of shape (embedding_dim,) - typically 384 or 768
    """
    tokenizer, model, device = get_chemberta_model(model_name)

    # Tokenize
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

    # Pool token embeddings
    if pooling == "cls":
        # Use [CLS] token embedding
        embedding = hidden_states[:, 0, :]
    elif pooling == "max":
        # Max pooling across sequence
        embedding = hidden_states.max(dim=1)[0]
    else:  # mean pooling (default)
        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        sum_embeddings = (hidden_states * attention_mask).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        embedding = sum_embeddings / sum_mask

    return embedding.cpu().numpy().squeeze()


def get_batch_embeddings(
    smiles_list: List[str],
    model_name: str = "DeepChem/ChemBERTa-77M-MTR",
    pooling: str = "mean",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Get embeddings for a batch of SMILES strings.

    More efficient than calling get_smiles_embedding repeatedly.

    Args:
        smiles_list: List of SMILES strings
        model_name: Pre-trained model to use
        pooling: How to pool token embeddings
        batch_size: Batch size for processing
        show_progress: Whether to show progress

    Returns:
        Numpy array of shape (n_molecules, embedding_dim)
    """
    tokenizer, model, device = get_chemberta_model(model_name)

    all_embeddings = []
    n_batches = (len(smiles_list) + batch_size - 1) // batch_size

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]

        if show_progress:
            batch_num = i // batch_size + 1
            print(f"\rProcessing batch {batch_num}/{n_batches}", end="", flush=True)

        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        # Pool
        if pooling == "cls":
            embeddings = hidden_states[:, 0, :]
        elif pooling == "max":
            embeddings = hidden_states.max(dim=1)[0]
        else:
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = (hidden_states * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / sum_mask

        all_embeddings.append(embeddings.cpu().numpy())

    if show_progress:
        print()  # Newline after progress

    return np.vstack(all_embeddings)


class MolecularEmbedder:
    """
    High-level interface for molecular embedding extraction.

    Combines multiple embedding strategies:
    1. ChemBERTa transformer embeddings
    2. 3D conformer descriptors (optional)
    3. Pharmacophore distances (optional)

    Example:
        embedder = MolecularEmbedder(include_3d=True)
        features = embedder.embed("CCO")  # Returns combined features
    """

    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        pooling: str = "mean",
        include_3d: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Pre-trained transformer model
            pooling: Token pooling strategy
            include_3d: Whether to include 3D conformer features
            random_seed: Seed for 3D conformer generation
        """
        self.model_name = model_name
        self.pooling = pooling
        self.include_3d = include_3d
        self.random_seed = random_seed

        # Pre-load model
        self.tokenizer, self.model, self.device = get_chemberta_model(model_name)

        # Import 3D features if needed
        if include_3d:
            from conformer_gen import get_all_3d_features, get_3d_feature_names
            self._get_3d_features = get_all_3d_features
            self._get_3d_names = get_3d_feature_names

    def embed(self, smiles: str) -> np.ndarray:
        """
        Get combined embedding for a single molecule.

        Args:
            smiles: SMILES string

        Returns:
            Combined feature vector
        """
        # Get transformer embedding
        transformer_emb = get_smiles_embedding(smiles, self.model_name, self.pooling)

        if self.include_3d:
            # Get 3D features
            features_3d = self._get_3d_features(smiles, self.random_seed)
            features_3d_array = np.array([features_3d[k] for k in self._get_3d_names()])
            return np.concatenate([transformer_emb, features_3d_array])

        return transformer_emb

    def embed_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get combined embeddings for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for transformer
            show_progress: Whether to show progress

        Returns:
            Array of shape (n_molecules, embedding_dim)
        """
        # Get transformer embeddings (batched)
        transformer_embs = get_batch_embeddings(
            smiles_list, self.model_name, self.pooling, batch_size, show_progress
        )

        if self.include_3d:
            # Get 3D features for each molecule
            if show_progress:
                print("Extracting 3D features...")

            features_3d_list = []
            for i, smi in enumerate(smiles_list):
                if show_progress and (i + 1) % 100 == 0:
                    print(f"\r3D features: {i+1}/{len(smiles_list)}", end="", flush=True)

                features = self._get_3d_features(smi, self.random_seed)
                features_array = np.array([features[k] for k in self._get_3d_names()])
                features_3d_list.append(features_array)

            if show_progress:
                print()

            features_3d = np.vstack(features_3d_list)
            return np.concatenate([transformer_embs, features_3d], axis=1)

        return transformer_embs

    def get_embedding_dim(self) -> int:
        """Get the dimension of the combined embedding."""
        # Test with dummy molecule
        emb = self.embed("C")
        return len(emb)

    def get_feature_names(self) -> List[str]:
        """Get names of all features in the embedding."""
        # Transformer features don't have interpretable names
        transformer_dim = len(get_smiles_embedding("C", self.model_name, self.pooling))
        names = [f"transformer_{i}" for i in range(transformer_dim)]

        if self.include_3d:
            names.extend(self._get_3d_names())

        return names


if __name__ == '__main__':
    # Test embedding extraction
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)NCC(O)C1=CC=C(O)C(O)=C1',  # Isoproterenol
    ]

    print("Testing molecular embeddings:")
    print("=" * 60)

    # Test single embedding
    print("\n1. Single molecule embedding (ChemBERTa + 3D):")
    embedder = MolecularEmbedder(include_3d=True)

    for smi in test_smiles:
        emb = embedder.embed(smi)
        print(f"  {smi[:30]}... -> dim={len(emb)}")

    # Test batch embedding
    print("\n2. Batch embedding:")
    batch_embs = embedder.embed_batch(test_smiles, show_progress=False)
    print(f"  Batch shape: {batch_embs.shape}")

    # Show embedding dimension breakdown
    print("\n3. Feature breakdown:")
    print(f"  Transformer features: {len(get_smiles_embedding('C'))}")
    print(f"  3D features: 20")
    print(f"  Total: {embedder.get_embedding_dim()}")
