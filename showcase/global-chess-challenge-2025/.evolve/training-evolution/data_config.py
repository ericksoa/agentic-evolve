#!/usr/bin/env python3
"""
Evolvable Training Data Configuration

This defines the training data configuration that will be evolved
to minimize ACPL. Each variant produces a different training dataset.

EVOLUTION TARGETS:
- min_elo: Minimum player Elo (higher = stronger games)
- stockfish_agreement: Require human move in SF top-N
- position_phases: Which game phases to include
- move_quality_filter: Filter by centipawn loss of human move
- augmentation: Data augmentation strategies
- training_epochs: Number of epochs
- lora_rank: LoRA adapter size
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Evolvable training data configuration"""

    # Data filtering
    min_elo: int = 2200  # Minimum average Elo
    max_elo: int = 3000  # Maximum (to filter engine games)
    stockfish_top_n: int = 10  # Human move must be in SF top-N (0=disabled)

    # Position selection
    include_opening: bool = True  # Moves 1-15
    include_middlegame: bool = True  # Moves 16-35
    include_endgame: bool = True  # Moves 36+
    min_pieces: int = 6  # Minimum pieces on board
    max_eval_diff: int = 500  # Skip positions with eval > ±500cp (decided games)

    # Move quality filtering
    max_centipawn_loss: int = 100  # Skip moves with CPL > this (blunders)

    # Data size
    max_examples: int = 50000  # Cap training examples
    validation_split: float = 0.1

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    lora_rank: int = 64
    lora_alpha: int = 128

    # Augmentation
    mirror_positions: bool = False  # Flip board horizontally
    include_losing_side: bool = True  # Include both sides' moves

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DataConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def get_name(self) -> str:
        """Generate descriptive name for this config"""
        parts = [f"elo{self.min_elo}"]
        if self.stockfish_top_n > 0:
            parts.append(f"sf{self.stockfish_top_n}")
        if self.max_centipawn_loss < 100:
            parts.append(f"cpl{self.max_centipawn_loss}")
        parts.append(f"r{self.lora_rank}")
        parts.append(f"e{self.epochs}")
        return "_".join(parts)


# ============================================================
# BASELINE CONFIGURATION
# ============================================================

BASELINE = DataConfig(
    min_elo=2200,
    stockfish_top_n=0,  # No filtering
    include_opening=True,
    include_middlegame=True,
    include_endgame=True,
    max_centipawn_loss=100,  # Allow some inaccuracies
    max_examples=30000,
    epochs=3,
    lora_rank=64,
)


# ============================================================
# CANDIDATE VARIANTS FOR EVOLUTION
# ============================================================

# Higher Elo = Stronger games
ELO_VARIANTS = [2200, 2400, 2500, 2600, 2700]

# Stockfish agreement = Only train on "correct" moves
STOCKFISH_TOP_N_VARIANTS = [0, 1, 3, 5, 10]  # 0=disabled, 1=exact match, 3=top-3, etc.

# Centipawn loss filter = Skip blunders
CPL_VARIANTS = [25, 50, 100, 200]  # Lower = stricter

# Training intensity
EPOCH_VARIANTS = [2, 3, 4, 5]
LORA_RANK_VARIANTS = [32, 64, 128]

# Position phases
PHASE_VARIANTS = [
    {"opening": True, "middlegame": True, "endgame": True},  # All
    {"opening": True, "middlegame": True, "endgame": False},  # No endgame
    {"opening": False, "middlegame": True, "endgame": True},  # No opening
    {"opening": True, "middlegame": False, "endgame": False},  # Opening only
]


def generate_initial_population(n: int = 6) -> List[DataConfig]:
    """Generate diverse initial population"""
    import random

    population = [BASELINE]  # Always include baseline

    while len(population) < n:
        config = DataConfig(
            min_elo=random.choice(ELO_VARIANTS),
            stockfish_top_n=random.choice(STOCKFISH_TOP_N_VARIANTS),
            max_centipawn_loss=random.choice(CPL_VARIANTS),
            epochs=random.choice(EPOCH_VARIANTS),
            lora_rank=random.choice(LORA_RANK_VARIANTS),
            include_opening=random.choice([True, False]) or True,  # Bias toward including
            include_middlegame=True,  # Always include middlegame
            include_endgame=random.choice([True, False]),
        )
        population.append(config)

    return population


if __name__ == "__main__":
    print("Baseline config:")
    print(json.dumps(BASELINE.to_dict(), indent=2))
    print(f"\nConfig name: {BASELINE.get_name()}")

    print("\n\nExample initial population:")
    for i, config in enumerate(generate_initial_population(5)):
        print(f"\n{i+1}. {config.get_name()}")
        print(f"   Elo>={config.min_elo}, SF_top={config.stockfish_top_n}, CPL<={config.max_centipawn_loss}")
        print(f"   Epochs={config.epochs}, LoRA={config.lora_rank}")
