"""
Pairwise Ranking Model for Tree Packing

Instead of predicting absolute side length, this model learns to compare
two packings and predict which one is better. This directly optimizes for
the selection task (choosing the best packing from multiple candidates).

Key insight: Gen102 model predicted well (MAE=0.04) but failed at selection.
Trained for absolute prediction, but we need relative ranking.
"""

import torch
import torch.nn as nn


class PackingEncoder(nn.Module):
    """
    Encodes a packing state into a fixed-size embedding.
    Shared between both packings in the pair.
    """

    def __init__(self, max_n: int = 50, embed_dim: int = 64):
        super().__init__()
        self.max_n = max_n
        self.embed_dim = embed_dim

        # Input: max_n * 3 (positions) + 1 (n_trees) + 1 (target_n)
        input_dim = max_n * 3 + 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
        )

    def forward(self, features: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, max_n * 3 + 1] - packing state
            target_n: [batch, 1] - normalized target n

        Returns:
            embedding: [batch, embed_dim]
        """
        x = torch.cat([features, target_n], dim=-1)
        return self.encoder(x)


class PairwiseRankingNet(nn.Module):
    """
    Siamese network for pairwise ranking.

    Takes two packings, encodes them with shared weights,
    and predicts P(packing_A is better than packing_B).
    """

    def __init__(self, max_n: int = 50, embed_dim: int = 64):
        super().__init__()

        # Shared encoder for both packings
        self.encoder = PackingEncoder(max_n=max_n, embed_dim=embed_dim)

        # Comparison head: takes concatenated embeddings + element-wise product
        # Input: embed_A (64) + embed_B (64) + elem_product (64) + diff (64) = 256
        self.comparator = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode(self, features: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
        """Encode a single packing."""
        return self.encoder(features, target_n)

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        target_n: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features_a: [batch, max_n * 3 + 1] - first packing
            features_b: [batch, max_n * 3 + 1] - second packing
            target_n: [batch, 1] - target n (same for both)

        Returns:
            p_a_better: [batch, 1] - P(A is better than B)
        """
        # Encode both packings
        embed_a = self.encoder(features_a, target_n)  # [batch, embed_dim]
        embed_b = self.encoder(features_b, target_n)  # [batch, embed_dim]

        # Combine embeddings for comparison
        combined = torch.cat([
            embed_a,
            embed_b,
            embed_a * embed_b,  # element-wise product
            embed_a - embed_b,  # difference
        ], dim=-1)

        return self.comparator(combined)


class MarginRankingNet(nn.Module):
    """
    Alternative approach using margin ranking loss.

    Instead of binary classification, learns a score function
    where better packings get higher scores.
    """

    def __init__(self, max_n: int = 50, embed_dim: int = 64):
        super().__init__()

        self.encoder = PackingEncoder(max_n=max_n, embed_dim=embed_dim)

        # Score prediction head
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def score(self, features: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
        """Get quality score for a packing (higher = better)."""
        embed = self.encoder(features, target_n)
        return self.scorer(embed)

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        target_n: torch.Tensor
    ) -> tuple:
        """
        Returns scores for both packings.
        Use margin ranking loss: we want score_better > score_worse + margin
        """
        embed_a = self.encoder(features_a, target_n)
        embed_b = self.encoder(features_b, target_n)

        score_a = self.scorer(embed_a)
        score_b = self.scorer(embed_b)

        return score_a, score_b


def create_ranking_model(model_type: str = "pairwise", max_n: int = 50) -> nn.Module:
    """Factory function for creating ranking models."""
    if model_type == "pairwise":
        return PairwiseRankingNet(max_n=max_n)
    elif model_type == "margin":
        return MarginRankingNet(max_n=max_n)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
