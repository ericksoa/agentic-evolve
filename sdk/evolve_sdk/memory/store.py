"""
Evolution Memory Store.

Provides persistent, queryable memory for evolution runs using memvid.
Includes crash recovery, mutation pattern memory, and cross-problem learning.

Falls back to JSON-based storage when memvid is not available.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .schemas import (
    FrameType,
    BaseFrame,
    CheckpointFrame,
    MutationFrame,
    FailedMutationFrame,
    ChampionFrame,
    ExploitFrame,
    TrustDecisionFrame,
    HumanDecisionFrame,
    ConfigFrame,
    GenerationFrame,
    NoteFrame,
)

# Check for memvid availability
try:
    import memvid
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False

# Check for embeddings availability
try:
    from .embeddings import CodeEmbedder, get_embedder
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class MemoryConfig:
    """Configuration for evolution memory."""
    enabled: bool = True
    store_path: str | Path = ".evolve-sdk/{problem_id}/evolution.mv2"
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    max_frames: int = 100000
    checkpoint_on_mutation: bool = True
    checkpoint_on_evaluation: bool = True
    checkpoint_on_selection: bool = True
    store_failed_mutations: bool = True
    store_successful_mutations: bool = True


class EvolutionMemory:
    """
    Persistent memory store for evolution runs.

    Provides:
    - Crash recovery via checkpoints
    - Mutation pattern memory
    - Champion library
    - Exploit detection memory
    - Trust decision history

    Uses memvid for storage when available, falls back to JSON.
    """

    def __init__(
        self,
        store_path: str | Path,
        problem_id: str = "",
        mode: str = "size",
        config: MemoryConfig | None = None,
    ):
        """
        Initialize the memory store.

        Args:
            store_path: Path to the .mv2 (or .json fallback) file
            problem_id: Identifier for the current problem
            mode: Evolution mode (size, perf, ml)
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self.problem_id = problem_id
        self.mode = mode

        # Resolve store path
        if isinstance(store_path, str):
            store_path = store_path.replace("{problem_id}", problem_id)
        self.store_path = Path(store_path)

        # Ensure parent directory exists
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embedder if available and enabled
        self._embedder = None
        if self.config.use_embeddings and EMBEDDINGS_AVAILABLE:
            try:
                self._embedder = get_embedder(self.config.embedding_model)
            except Exception:
                pass  # Embeddings not critical

        # Initialize storage backend
        self._backend = self._init_backend()

        # Frame cache for quick access
        self._frame_cache: list[dict] = []
        self._load_cache()

    def _init_backend(self) -> "MemoryBackend":
        """Initialize the appropriate storage backend."""
        if MEMVID_AVAILABLE and str(self.store_path).endswith(".mv2"):
            return MemvidBackend(self.store_path)
        else:
            # Use JSON fallback
            json_path = self.store_path.with_suffix(".json")
            return JSONBackend(json_path)

    def _load_cache(self):
        """Load frames into cache for quick access."""
        try:
            self._frame_cache = self._backend.load_all()
        except Exception:
            self._frame_cache = []

    # ==================== CHECKPOINT OPERATIONS ====================

    def checkpoint(
        self,
        generation: int,
        population: list[dict],
        champion: dict | None = None,
        operation: str = "unknown",
        plateau_count: int = 0,
        notes: str = "",
    ) -> str:
        """
        Create a checkpoint for crash recovery.

        Called after every significant operation to enable resumption.

        Args:
            generation: Current generation number
            population: Current population list
            champion: Current champion solution
            operation: Type of operation (post_mutation, post_evaluation, post_selection)
            plateau_count: Current plateau count
            notes: Optional notes

        Returns:
            Frame ID
        """
        frame = CheckpointFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            generation=generation,
            operation=operation,
            population_json=json.dumps(population),
            champion_file=champion.get("file") if champion else None,
            champion_fitness=champion.get("fitness", 0) if champion else 0,
            plateau_count=plateau_count,
            notes=notes,
        )

        return self._store_frame(frame)

    def recover(self) -> dict | None:
        """
        Recover evolution state from the latest checkpoint.

        Returns:
            Dictionary with recovered state, or None if no checkpoint found.
            Contains: generation, population, champion, operation, plateau_count
        """
        checkpoints = self.query(frame_type=FrameType.CHECKPOINT)
        if not checkpoints:
            return None

        # Get latest checkpoint
        latest = max(checkpoints, key=lambda f: f.get("timestamp", ""))

        try:
            population = json.loads(latest.get("population_json", "[]"))
        except json.JSONDecodeError:
            population = []

        champion = None
        if latest.get("champion_file"):
            champion = {
                "file": latest["champion_file"],
                "fitness": latest.get("champion_fitness", 0),
            }

        return {
            "generation": latest.get("generation", 0),
            "population": population,
            "champion": champion,
            "operation": latest.get("operation", "unknown"),
            "plateau_count": latest.get("plateau_count", 0),
            "timestamp": latest.get("timestamp"),
            "notes": latest.get("notes", ""),
        }

    # ==================== MUTATION OPERATIONS ====================

    def store_mutation(
        self,
        generation: int,
        variant: str,
        parent_file: str,
        child_file: str,
        parent_fitness: float,
        child_fitness: float,
        diff_content: str = "",
        mutation_type: str = "mutation",
        tags: list[str] | None = None,
        broke_plateau: bool = False,
        became_champion: bool = False,
    ) -> str:
        """
        Store a successful mutation pattern.

        Args:
            generation: Generation number
            variant: Mutation variant (a, b, c, x for crossover)
            parent_file: Path to parent solution
            child_file: Path to child solution
            parent_fitness: Parent fitness score
            child_fitness: Child fitness score
            diff_content: Code diff between parent and child
            mutation_type: Type (mutation or crossover)
            tags: Optional tags for categorization
            broke_plateau: Whether this mutation broke a plateau
            became_champion: Whether this became the new champion

        Returns:
            Frame ID
        """
        fitness_delta = child_fitness - parent_fitness
        fitness_delta_pct = (fitness_delta / parent_fitness * 100) if parent_fitness > 0 else 0

        frame = MutationFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            generation=generation,
            variant=variant,
            parent_file=parent_file,
            child_file=child_file,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            fitness_delta=fitness_delta,
            fitness_delta_pct=fitness_delta_pct,
            diff_content=diff_content,
            mutation_type=mutation_type,
            tags=tags or [],
            broke_plateau=broke_plateau,
            became_champion=became_champion,
        )

        return self._store_frame(frame)

    def store_failed_mutation(
        self,
        generation: int,
        parent_file: str,
        child_file: str,
        parent_fitness: float,
        child_fitness: float,
        diff_content: str = "",
        failure_reason: str = "worse",
        parent_code: str = "",
    ) -> str:
        """
        Store a failed mutation to avoid repeating it.

        Args:
            generation: Generation number
            parent_file: Path to parent solution
            child_file: Path to child solution
            parent_fitness: Parent fitness score
            child_fitness: Child fitness score
            diff_content: Code diff
            failure_reason: Why it failed (worse, invalid, rejected_by_trust)
            parent_code: Parent code for hashing

        Returns:
            Frame ID
        """
        parent_hash = ""
        if parent_code and self._embedder:
            from .embeddings import CodeEmbedder
            parent_hash = CodeEmbedder.hash_code(parent_code)

        frame = FailedMutationFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            generation=generation,
            parent_file=parent_file,
            parent_hash=parent_hash,
            child_file=child_file,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            fitness_delta=child_fitness - parent_fitness,
            diff_content=diff_content,
            failure_reason=failure_reason,
        )

        return self._store_frame(frame)

    def find_similar_mutations(
        self,
        code: str,
        limit: int = 5,
        success_only: bool = True,
    ) -> list[dict]:
        """
        Find mutations similar to the given code.

        Uses embedding similarity to find relevant past mutations.

        Args:
            code: Code to find similar mutations for
            limit: Maximum number of results
            success_only: Only return successful mutations

        Returns:
            List of similar mutation frames
        """
        if not self._embedder:
            return []

        # Get embedding for query code
        query_embedding = self._embedder.embed_code(code)

        # Filter mutations
        frame_type = FrameType.MUTATION if success_only else None
        mutations = self.query(frame_type=frame_type)

        if not success_only:
            # Include both successful and failed
            mutations = [
                m for m in self._frame_cache
                if m.get("frame_type") in [FrameType.MUTATION.value, FrameType.FAILED_MUTATION.value]
            ]

        # Score by similarity
        scored = []
        for m in mutations:
            diff = m.get("diff_content", "")
            if diff:
                diff_embedding = self._embedder.embed_diff(diff)
                similarity = self._embedder.similarity(query_embedding, diff_embedding)
                scored.append((similarity, m))

        # Sort by similarity and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def find_failed_mutations_for_parent(
        self,
        parent_code: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find failed mutations that were tried on a similar parent.

        Helps avoid repeating failed approaches.

        Args:
            parent_code: Code of the parent solution
            limit: Maximum number of results

        Returns:
            List of failed mutation frames
        """
        if not self._embedder:
            return []

        from .embeddings import CodeEmbedder
        parent_hash = CodeEmbedder.hash_code(parent_code)

        # Look for exact hash matches first
        failed = self.query(frame_type=FrameType.FAILED_MUTATION)
        exact_matches = [f for f in failed if f.get("parent_hash") == parent_hash]

        if exact_matches:
            return exact_matches[:limit]

        # Fall back to similarity search
        return self.find_similar_mutations(parent_code, limit=limit, success_only=False)

    # ==================== CHAMPION OPERATIONS ====================

    def store_champion(
        self,
        file_path: str,
        code_content: str,
        fitness: float,
        generation: int,
        improvement_over_baseline: float = 0,
        trust_score: float = 1.0,
        cv_mean: float = 0,
        cv_std: float = 0,
        insights: list[str] | None = None,
        problem_description: str = "",
    ) -> str:
        """
        Store a champion solution.

        Args:
            file_path: Path to champion file
            code_content: Champion code
            fitness: Champion fitness
            generation: Generation when it became champion
            improvement_over_baseline: Improvement over starting fitness
            trust_score: Trust score from adversary
            cv_mean: Cross-validation mean
            cv_std: Cross-validation std
            insights: Key insights about what made this champion work
            problem_description: Description of the problem

        Returns:
            Frame ID
        """
        frame = ChampionFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            file_path=file_path,
            code_content=code_content,
            fitness=fitness,
            generation=generation,
            improvement_over_baseline=improvement_over_baseline,
            trust_score=trust_score,
            cv_mean=cv_mean,
            cv_std=cv_std,
            insights=insights or [],
            problem_description=problem_description or self.problem_id,
        )

        return self._store_frame(frame)

    def find_similar_champions(
        self,
        problem_description: str,
        limit: int = 3,
    ) -> list[dict]:
        """
        Find champions from similar problems.

        Useful for bootstrapping new evolution runs.

        Args:
            problem_description: Description of the new problem
            limit: Maximum number of results

        Returns:
            List of champion frames from similar problems
        """
        if not self._embedder:
            return []

        query_embedding = self._embedder.embed_text(problem_description)
        champions = self.query(frame_type=FrameType.CHAMPION)

        scored = []
        for c in champions:
            desc = c.get("problem_description", "")
            if desc:
                desc_embedding = self._embedder.embed_text(desc)
                similarity = self._embedder.similarity(query_embedding, desc_embedding)
                scored.append((similarity, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]

    # ==================== EXPLOIT OPERATIONS ====================

    def store_exploit(
        self,
        file_path: str,
        code_content: str,
        pattern_type: str,
        detection_method: str,
        flags: list[str],
        claimed_fitness: float,
        actual_fitness: float = 0,
    ) -> str:
        """
        Store a detected exploit pattern.

        Args:
            file_path: Path to the exploit file
            code_content: Code that triggered detection
            pattern_type: Type of exploit (hardcoded, timing, etc.)
            detection_method: How it was detected
            flags: Flags raised
            claimed_fitness: Fitness the solution claimed
            actual_fitness: Actual fitness (if determined)

        Returns:
            Frame ID
        """
        frame = ExploitFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            file_path=file_path,
            code_content=code_content,
            pattern_type=pattern_type,
            detection_method=detection_method,
            flags=flags,
            claimed_fitness=claimed_fitness,
            actual_fitness=actual_fitness,
        )

        return self._store_frame(frame)

    def find_similar_exploits(
        self,
        code: str,
        threshold: float = 0.85,
    ) -> list[dict]:
        """
        Check if code resembles known exploit patterns.

        Args:
            code: Code to check
            threshold: Similarity threshold for matching

        Returns:
            List of similar exploit frames
        """
        if not self._embedder:
            return []

        query_embedding = self._embedder.embed_code(code)
        exploits = self.query(frame_type=FrameType.EXPLOIT)

        matches = []
        for e in exploits:
            exploit_code = e.get("code_content", "")
            if exploit_code:
                exploit_embedding = self._embedder.embed_code(exploit_code)
                similarity = self._embedder.similarity(query_embedding, exploit_embedding)
                if similarity >= threshold:
                    matches.append({**e, "_similarity": similarity})

        return sorted(matches, key=lambda x: x.get("_similarity", 0), reverse=True)

    # ==================== TRUST OPERATIONS ====================

    def store_trust_decision(
        self,
        file_path: str,
        generation: int,
        claimed_fitness: float,
        trust_score: float,
        recommendation: str,
        flags: list[str],
        escalation_level: int = 0,
        final_outcome: str = "",
    ) -> str:
        """Store an adversary trust decision."""
        frame = TrustDecisionFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            file_path=file_path,
            generation=generation,
            claimed_fitness=claimed_fitness,
            trust_score=trust_score,
            recommendation=recommendation,
            flags=flags,
            escalation_level=escalation_level,
            final_outcome=final_outcome,
        )

        return self._store_frame(frame)

    def store_human_decision(
        self,
        file_path: str,
        generation: int,
        fitness: float,
        original_trust: float,
        adversary_recommendation: str,
        flags: list[str],
        human_decision: str,
        adjusted_trust: float | None = None,
        reasoning: str = "",
    ) -> str:
        """Store a human escalation decision."""
        frame = HumanDecisionFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            file_path=file_path,
            generation=generation,
            fitness=fitness,
            original_trust=original_trust,
            adversary_recommendation=adversary_recommendation,
            flags=flags,
            human_decision=human_decision,
            adjusted_trust=adjusted_trust,
            reasoning=reasoning,
        )

        return self._store_frame(frame)

    # ==================== GENERATION OPERATIONS ====================

    def store_generation(
        self,
        generation: int,
        population: list[dict],
        best_fitness: float,
        champion_file: str | None = None,
        mutations_tried: int = 0,
        mutations_kept: int = 0,
        plateau_count: int = 0,
        previous_best: float = 0,
    ) -> str:
        """Store a generation snapshot for temporal queries."""
        frame = GenerationFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            generation=generation,
            population_json=json.dumps(population),
            best_fitness=best_fitness,
            champion_file=champion_file,
            mutations_tried=mutations_tried,
            mutations_kept=mutations_kept,
            plateau_count=plateau_count,
            improvement_from_previous=best_fitness - previous_best,
        )

        return self._store_frame(frame)

    def get_generation(self, generation: int) -> dict | None:
        """Get a specific generation snapshot."""
        frames = self.query(frame_type=FrameType.GENERATION)
        for f in frames:
            if f.get("generation") == generation:
                return f
        return None

    def get_fitness_trajectory(self) -> list[tuple[int, float]]:
        """Get fitness progression over generations."""
        frames = self.query(frame_type=FrameType.GENERATION)
        trajectory = [(f.get("generation", 0), f.get("best_fitness", 0)) for f in frames]
        return sorted(trajectory, key=lambda x: x[0])

    # ==================== NOTE OPERATIONS ====================

    def store_note(
        self,
        title: str,
        content: str,
        category: str = "fact",
        priority: str = "normal",
        tags: list[str] | None = None,
        verified: bool = True,
        source: str = "human",
    ) -> str:
        """
        Store a note for institutional knowledge.

        Notes persist across conversations and evolution runs to prevent
        forgetting important facts, constraints, and lessons learned.

        Args:
            title: Short summary (e.g., "Never achieved <100 ACPL")
            content: Full note content with details
            category: Type of note (constraint, fact, lesson, goal, warning)
            priority: Importance (critical, high, normal, low)
            tags: Optional tags for filtering
            verified: Whether this has been verified
            source: Where this info came from (human, evaluation, etc.)

        Returns:
            Frame ID
        """
        frame = NoteFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            category=category,
            title=title,
            content=content,
            priority=priority,
            tags=tags or [],
            verified=verified,
            source=source,
        )

        return self._store_frame(frame)

    def get_notes(
        self,
        category: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve stored notes.

        Args:
            category: Filter by category (constraint, fact, lesson, goal, warning)
            priority: Filter by priority (critical, high, normal, low)
            tags: Filter by tags (returns notes with ANY of these tags)

        Returns:
            List of note frames matching filters
        """
        notes = self.query(frame_type=FrameType.NOTE)

        if category:
            notes = [n for n in notes if n.get("category") == category]

        if priority:
            notes = [n for n in notes if n.get("priority") == priority]

        if tags:
            notes = [
                n for n in notes
                if any(tag in n.get("tags", []) for tag in tags)
            ]

        return notes

    def get_critical_notes(self) -> list[dict]:
        """Get all critical priority notes - these should never be forgotten."""
        return self.get_notes(priority="critical")

    # ==================== QUERY OPERATIONS ====================

    def query(
        self,
        frame_type: FrameType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Query frames from memory.

        Args:
            frame_type: Filter by frame type
            filters: Additional metadata filters
            limit: Maximum number of results

        Returns:
            List of matching frames
        """
        results = self._frame_cache

        if frame_type:
            results = [f for f in results if f.get("frame_type") == frame_type.value]

        if filters:
            for key, value in filters.items():
                results = [f for f in results if f.get(key) == value]

        if limit:
            results = results[:limit]

        return results

    def search_text(self, query: str, limit: int = 10) -> list[dict]:
        """
        Full-text search across frames.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching frames
        """
        query_lower = query.lower()
        matches = []

        for frame in self._frame_cache:
            # Search in common text fields
            searchable = " ".join([
                str(frame.get("diff_content", "")),
                str(frame.get("code_content", "")),
                str(frame.get("notes", "")),
                str(frame.get("problem_description", "")),
                " ".join(frame.get("tags", [])),
                " ".join(frame.get("flags", [])),
                " ".join(frame.get("insights", [])),
            ]).lower()

            if query_lower in searchable:
                matches.append(frame)

        return matches[:limit]

    # ==================== INTERNAL OPERATIONS ====================

    def _store_frame(self, frame: BaseFrame) -> str:
        """Store a frame and return its ID."""
        frame_dict = frame.to_dict()

        # Add embedding if available
        if self._embedder and self.config.use_embeddings:
            content = (
                frame_dict.get("diff_content") or
                frame_dict.get("code_content") or
                frame_dict.get("problem_description") or
                ""
            )
            if content:
                try:
                    frame_dict["_embedding"] = self._embedder.embed_code(content)
                except Exception:
                    pass  # Embeddings not critical

        # Store to backend
        frame_id = self._backend.store(frame_dict)

        # Update cache
        frame_dict["_id"] = frame_id
        self._frame_cache.append(frame_dict)

        return frame_id

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        type_counts = {}
        for frame in self._frame_cache:
            ft = frame.get("frame_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        return {
            "total_frames": len(self._frame_cache),
            "frame_types": type_counts,
            "store_path": str(self.store_path),
            "backend": self._backend.__class__.__name__,
            "embeddings_enabled": self._embedder is not None,
        }

    def close(self):
        """Close the memory store."""
        self._backend.close()


# ==================== STORAGE BACKENDS ====================

class MemoryBackend:
    """Abstract base class for storage backends."""

    def store(self, frame: dict) -> str:
        """Store a frame and return its ID."""
        raise NotImplementedError

    def load_all(self) -> list[dict]:
        """Load all frames."""
        raise NotImplementedError

    def close(self):
        """Close the backend."""
        pass


class JSONBackend(MemoryBackend):
    """
    JSON-based storage backend.

    Simple fallback when memvid is not available.
    Stores frames as a JSON array in a file.
    """

    def __init__(self, path: Path):
        self.path = path
        self._frames: list[dict] = []
        self._load()

    def _load(self):
        """Load existing frames from file."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._frames = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._frames = []

    def _save(self):
        """Save frames to file."""
        # Atomic write with temp file
        temp_path = self.path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(self._frames, f, indent=2)
            temp_path.replace(self.path)
        except IOError:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def store(self, frame: dict) -> str:
        """Store a frame."""
        frame_id = f"{len(self._frames):06d}"
        frame["_id"] = frame_id
        self._frames.append(frame)
        self._save()
        return frame_id

    def load_all(self) -> list[dict]:
        """Load all frames."""
        return self._frames.copy()

    def close(self):
        """Save and close."""
        self._save()


class MemvidBackend(MemoryBackend):
    """
    Memvid-based storage backend.

    Uses memvid for crash-safe, queryable storage with vector search.
    """

    def __init__(self, path: Path):
        self.path = path
        self._store = None
        self._init_store()

    def _init_store(self):
        """Initialize the memvid store."""
        if not MEMVID_AVAILABLE:
            raise ImportError("memvid is required for MemvidBackend")

        # TODO: Initialize memvid store
        # The exact API depends on memvid-sdk version
        # For now, we'll use a placeholder that falls back to JSON
        # self._store = memvid.MemoryStore(str(self.path))

        # Fallback to JSON until we have memvid API details
        self._json_fallback = JSONBackend(self.path.with_suffix(".json"))

    def store(self, frame: dict) -> str:
        """Store a frame."""
        # TODO: Use actual memvid API
        # return self._store.add_frame(
        #     content=json.dumps(frame),
        #     metadata=frame,
        #     embedding=frame.get("_embedding"),
        # )
        return self._json_fallback.store(frame)

    def load_all(self) -> list[dict]:
        """Load all frames."""
        # TODO: Use actual memvid API
        # return [json.loads(f.content) for f in self._store.query_all()]
        return self._json_fallback.load_all()

    def close(self):
        """Close the store."""
        # TODO: self._store.close()
        self._json_fallback.close()
