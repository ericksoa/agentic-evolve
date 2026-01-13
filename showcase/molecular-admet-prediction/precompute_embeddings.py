"""
Pre-compute and Cache Molecular Embeddings for hERG Dataset

This script extracts ChemBERTa + 3D embeddings for all molecules in the
TDC hERG dataset and caches them to disk. This enables fast training
and inference by avoiding repeated embedding computation.

Usage:
    python precompute_embeddings.py

The embeddings are saved to:
    data/embeddings/train_embeddings.npz
    data/embeddings/valid_embeddings.npz
    data/embeddings/test_embeddings.npz
"""

import os
import numpy as np
from pathlib import Path
from tdc.single_pred import Tox
from embeddings_3d import MolecularEmbedder, get_batch_embeddings


def load_herg_data():
    """Load TDC hERG dataset with scaffold split."""
    data = Tox(name='hERG')
    split = data.get_split(method='scaffold', seed=42)
    return split['train'], split['valid'], split['test']


def precompute_and_save(
    smiles_list: list,
    labels: np.ndarray,
    output_path: Path,
    embedder: MolecularEmbedder,
    batch_size: int = 32
):
    """
    Compute embeddings and save to disk.

    Args:
        smiles_list: List of SMILES strings
        labels: Corresponding labels
        output_path: Path to save .npz file
        embedder: MolecularEmbedder instance
        batch_size: Batch size for processing
    """
    print(f"\nComputing embeddings for {len(smiles_list)} molecules...")

    # Get embeddings
    embeddings = embedder.embed_batch(smiles_list, batch_size=batch_size)

    # Save to disk
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=labels,
        smiles=smiles_list
    )

    print(f"Saved to {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """Main function to precompute all embeddings."""
    # Create output directory
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading TDC hERG dataset...")
    train_df, valid_df, test_df = load_herg_data()

    print(f"  Train: {len(train_df)} molecules")
    print(f"  Valid: {len(valid_df)} molecules")
    print(f"  Test: {len(test_df)} molecules")

    # Initialize embedder
    print("\nInitializing embedder...")
    embedder = MolecularEmbedder(
        model_name="DeepChem/ChemBERTa-77M-MTR",
        include_3d=True,
        random_seed=42
    )

    print(f"Embedding dimension: {embedder.get_embedding_dim()}")

    # Process each split
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        output_path = output_dir / f"{name}_embeddings.npz"

        # Skip if already computed
        if output_path.exists():
            print(f"\n{name} embeddings already exist at {output_path}")
            data = np.load(output_path)
            print(f"  Loaded shape: {data['embeddings'].shape}")
            continue

        precompute_and_save(
            smiles_list=df['Drug'].tolist(),
            labels=df['Y'].values,
            output_path=output_path,
            embedder=embedder,
            batch_size=32
        )

    print("\n" + "=" * 60)
    print("Precomputation complete!")
    print("=" * 60)

    # Summary
    total_size = sum(
        os.path.getsize(output_dir / f"{name}_embeddings.npz")
        for name in ["train", "valid", "test"]
        if (output_dir / f"{name}_embeddings.npz").exists()
    )
    print(f"\nTotal cache size: {total_size / 1024 / 1024:.2f} MB")


def load_cached_embeddings(split: str = "train") -> tuple:
    """
    Load pre-computed embeddings from disk.

    Args:
        split: "train", "valid", or "test"

    Returns:
        Tuple of (embeddings, labels, smiles)
    """
    path = Path(f"data/embeddings/{split}_embeddings.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run precompute_embeddings.py first."
        )

    data = np.load(path, allow_pickle=True)
    return data['embeddings'], data['labels'], data['smiles'].tolist()


if __name__ == '__main__':
    main()
