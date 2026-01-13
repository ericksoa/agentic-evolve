"""
Evolution Memory System - Persistent, queryable memory for evolution runs.

Provides crash recovery, mutation pattern memory, cross-problem learning,
and inter-agent communication using memvid for storage and
sentence-transformers for embeddings.

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

    # Inter-agent messaging
    memory.broadcast(
        from_agent="mutator_a",
        message_type="discovery",
        title="Inlining gave +26% improvement",
        content="Function call overhead dominates. Inline hot functions.",
        generation=1
    )

    # Notify human operator
    memory.notify_human(
        from_agent="runner",
        title="Evolution started",
        priority="info"
    )

    # Get message feed
    print(memory.get_message_feed(limit=20))
"""

__version__ = "0.2.0"

from .store import EvolutionMemory, MemoryConfig
from .schemas import (
    FrameType,
    CheckpointFrame,
    MutationFrame,
    ChampionFrame,
    ExploitFrame,
    NoteFrame,
    MessageFrame,
    MessageType,
    MessagePriority,
    AGENT_IDENTITIES,
)
from .embeddings import CodeEmbedder

__all__ = [
    # Core
    "EvolutionMemory",
    "MemoryConfig",
    # Frame types
    "FrameType",
    "CheckpointFrame",
    "MutationFrame",
    "ChampionFrame",
    "ExploitFrame",
    "NoteFrame",
    # Messaging
    "MessageFrame",
    "MessageType",
    "MessagePriority",
    "AGENT_IDENTITIES",
    # Embeddings
    "CodeEmbedder",
]
