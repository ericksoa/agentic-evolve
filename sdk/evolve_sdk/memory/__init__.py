"""
Evolution Memory System - Persistent, queryable memory for evolution runs.

Provides crash recovery, mutation pattern memory, cross-problem learning,
and inter-agent communication using memvid-sdk for storage and
sentence-transformers for embeddings.

Two levels of memory API:

1. EvolutionMemory - Low-level API for evolution-specific frames
   (checkpoints, mutations, champions, etc.)

2. MemoryManager - High-level API for project-level memory
   (DRIs, submissions, evaluations, config changes)

Usage (High-level - recommended):
    from evolve_sdk.memory import MemoryManager, DRI

    memory = MemoryManager(project_path=".")

    # Get all DRIs - ALWAYS call before submissions
    dris = memory.get_all_dris()

    # Semantic search for relevant DRIs
    relevant = memory.get_relevant_dris("uploading to huggingface")

    # Add a new DRI
    memory.add_dri(DRI(
        id="DRI-015",
        title="Always verify uploads",
        root_cause="Upload succeeded but files were corrupted",
    ))

    # Pre-action safety check
    check = memory.pre_action_check("submission", "uploading v6 model")
    if check["warnings"]:
        print("Review warnings before proceeding:")
        for w in check["warnings"]:
            print(f"  - {w}")

Usage (Low-level - evolution runs):
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

__version__ = "0.3.0"

from .store import EvolutionMemory, MemoryConfig, MEMVID_AVAILABLE
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
from .project_schemas import (
    DRI,
    Submission,
    ModelEvaluation,
    ConfigChange,
)
from .memory_manager import MemoryManager, get_memory_manager
from .embeddings import CodeEmbedder

__all__ = [
    # High-level API (recommended)
    "MemoryManager",
    "get_memory_manager",
    # Project schemas
    "DRI",
    "Submission",
    "ModelEvaluation",
    "ConfigChange",
    # Low-level API
    "EvolutionMemory",
    "MemoryConfig",
    "MEMVID_AVAILABLE",
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
