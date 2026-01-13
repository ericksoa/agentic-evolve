"""
Evolution Memory System - Persistent, queryable memory for evolution runs.

Provides crash recovery, mutation pattern memory, and cross-problem learning
using memvid for storage and sentence-transformers for embeddings.

Usage:
    from evolve_sdk.memory import EvolutionMemory

    memory = EvolutionMemory(store_path=".evolve-sdk/problem/evolution.mv2")

    # Store a checkpoint
    memory.checkpoint(
        generation=5,
        population=[...],
        champion="gen5b.py",
        operation="post_evaluation"
    )

    # Recover from crash
    state = memory.recover()

    # Query similar mutations
    similar = memory.find_similar_mutations(parent_code, limit=5)
"""

__version__ = "0.1.0"

from .store import EvolutionMemory, MemoryConfig
from .schemas import FrameType, CheckpointFrame, MutationFrame, ChampionFrame, ExploitFrame, NoteFrame
from .embeddings import CodeEmbedder

__all__ = [
    "EvolutionMemory",
    "MemoryConfig",
    "FrameType",
    "CheckpointFrame",
    "MutationFrame",
    "ChampionFrame",
    "ExploitFrame",
    "NoteFrame",
    "CodeEmbedder",
]
