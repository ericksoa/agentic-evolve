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
    MessageFrame,
    MessageType,
    MessagePriority,
    AGENT_IDENTITIES,
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

    # ==================== MESSAGING OPERATIONS ====================

    def broadcast(
        self,
        from_agent: str,
        message_type: str,
        title: str,
        content: str = "",
        priority: str = "info",
        to_audience: list[str] | None = None,
        generation: int = 0,
        related_file: str = "",
        related_fitness: float | None = None,
        ttl_generations: int = -1,
        tags: list[str] | None = None,
        requires_response: bool = False,
    ) -> str:
        """
        Broadcast a message to agents and/or the human operator.

        This is the primary method for inter-agent communication.

        Args:
            from_agent: Who is sending (e.g., "mutator_a", "runner", "human")
            message_type: Type from MessageType (status, discovery, warning, etc.)
            title: Short summary for the feed
            content: Full message body
            priority: From MessagePriority (debug, info, important, urgent, critical)
            to_audience: Who should see this (e.g., ["human"], ["*"], ["mutator_*"])
            generation: Current generation number
            related_file: Associated file path (if any)
            related_fitness: Fitness value (if relevant)
            ttl_generations: Auto-expire after N generations (-1 = never)
            tags: Optional tags for filtering
            requires_response: Whether this needs acknowledgment

        Returns:
            Message frame ID
        """
        # Get agent emoji from identity constants
        agent_info = AGENT_IDENTITIES.get(from_agent, {})
        agent_emoji = agent_info.get("emoji", "💬")

        frame = MessageFrame(
            problem_id=self.problem_id,
            mode=self.mode,
            from_agent=from_agent,
            agent_emoji=agent_emoji,
            to_audience=to_audience or ["*"],
            message_type=message_type,
            priority=priority,
            title=title,
            content=content,
            generation=generation,
            related_file=related_file,
            related_fitness=related_fitness,
            ttl_generations=ttl_generations,
            tags=tags or [],
            requires_response=requires_response,
        )

        return self._store_frame(frame)

    def notify_human(
        self,
        from_agent: str,
        title: str,
        content: str = "",
        priority: str = "info",
        generation: int = 0,
        related_file: str = "",
        related_fitness: float | None = None,
    ) -> str:
        """
        Send a notification specifically to the human operator.

        Convenience method for agent-to-human communication.

        Args:
            from_agent: Who is sending
            title: Short summary
            content: Full details
            priority: Importance level
            generation: Current generation
            related_file: Associated file
            related_fitness: Fitness value

        Returns:
            Message frame ID
        """
        return self.broadcast(
            from_agent=from_agent,
            message_type="status",
            title=title,
            content=content,
            priority=priority,
            to_audience=["human"],
            generation=generation,
            related_file=related_file,
            related_fitness=related_fitness,
        )

    def announce_milestone(
        self,
        title: str,
        content: str = "",
        generation: int = 0,
        related_file: str = "",
        related_fitness: float | None = None,
    ) -> str:
        """
        Announce a significant milestone to everyone.

        Used for champion promotions, plateaus broken, etc.

        Args:
            title: Milestone summary
            content: Details
            generation: When it happened
            related_file: Champion file (if applicable)
            related_fitness: New fitness

        Returns:
            Message frame ID
        """
        return self.broadcast(
            from_agent="runner",
            message_type="milestone",
            title=title,
            content=content,
            priority="important",
            to_audience=["*"],
            generation=generation,
            related_file=related_file,
            related_fitness=related_fitness,
        )

    def warn(
        self,
        from_agent: str,
        title: str,
        content: str = "",
        generation: int = 0,
        tags: list[str] | None = None,
    ) -> str:
        """
        Send a warning message.

        Used by adversary to share exploit patterns, etc.

        Args:
            from_agent: Who detected the issue
            title: Warning summary
            content: Details
            generation: When detected
            tags: Categories for filtering

        Returns:
            Message frame ID
        """
        return self.broadcast(
            from_agent=from_agent,
            message_type="warning",
            title=title,
            content=content,
            priority="important",
            to_audience=["*"],
            generation=generation,
            tags=tags,
        )

    def share_discovery(
        self,
        from_agent: str,
        title: str,
        content: str = "",
        generation: int = 0,
        related_fitness: float | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Share a useful discovery with other agents.

        Used when an agent finds something that might help others.

        Args:
            from_agent: Who discovered it
            title: Discovery summary
            content: Details about what was found
            generation: When discovered
            related_fitness: Fitness improvement (if applicable)
            tags: Categories

        Returns:
            Message frame ID
        """
        return self.broadcast(
            from_agent=from_agent,
            message_type="discovery",
            title=title,
            content=content,
            priority="info",
            to_audience=["*"],
            generation=generation,
            related_fitness=related_fitness,
            tags=tags,
        )

    def claim_strategy(
        self,
        from_agent: str,
        strategy: str,
        generation: int = 0,
    ) -> str:
        """
        Claim a strategy to avoid duplication with other agents.

        Mutators should call this before starting work.

        Args:
            from_agent: Agent claiming the strategy
            strategy: What approach they're trying
            generation: Current generation

        Returns:
            Message frame ID
        """
        return self.broadcast(
            from_agent=from_agent,
            message_type="strategy",
            title=f"Claiming: {strategy}",
            content=strategy,
            priority="debug",
            to_audience=["mutator_*"],
            generation=generation,
            ttl_generations=1,  # Only valid for current generation
            tags=["strategy_claim"],
        )

    def get_messages(
        self,
        audience: str | None = None,
        message_type: str | None = None,
        priority: str | None = None,
        since_generation: int | None = None,
        from_agent: str | None = None,
        limit: int | None = None,
        include_expired: bool = False,
        current_generation: int = 0,
    ) -> list[dict]:
        """
        Retrieve messages matching filters.

        Args:
            audience: Filter by intended audience (supports wildcards like "mutator_*")
            message_type: Filter by message type
            priority: Filter by minimum priority
            since_generation: Only messages from this generation onwards
            from_agent: Filter by sender
            limit: Maximum results
            include_expired: Include messages past their TTL
            current_generation: Current generation (for TTL calculation)

        Returns:
            List of message frames matching filters
        """
        messages = self.query(frame_type=FrameType.MESSAGE)

        # Filter by audience
        if audience:
            filtered = []
            for m in messages:
                msg_audience = m.get("to_audience", ["*"])
                # Check if audience matches (supports wildcards)
                for target in msg_audience:
                    if target == "*":
                        filtered.append(m)
                        break
                    elif target.endswith("*"):
                        prefix = target[:-1]
                        if audience.startswith(prefix):
                            filtered.append(m)
                            break
                    elif target == audience:
                        filtered.append(m)
                        break
            messages = filtered

        # Filter by message type
        if message_type:
            messages = [m for m in messages if m.get("message_type") == message_type]

        # Filter by priority (show this level and higher)
        if priority:
            priority_order = ["debug", "info", "important", "urgent", "critical"]
            min_level = priority_order.index(priority) if priority in priority_order else 0
            messages = [
                m for m in messages
                if priority_order.index(m.get("priority", "info")) >= min_level
            ]

        # Filter by generation
        if since_generation is not None:
            messages = [m for m in messages if m.get("generation", 0) >= since_generation]

        # Filter by sender
        if from_agent:
            messages = [m for m in messages if m.get("from_agent") == from_agent]

        # Filter expired messages
        if not include_expired and current_generation > 0:
            valid = []
            for m in messages:
                ttl = m.get("ttl_generations", -1)
                if ttl < 0:  # Never expires
                    valid.append(m)
                else:
                    msg_gen = m.get("generation", 0)
                    if current_generation - msg_gen < ttl:
                        valid.append(m)
            messages = valid

        # Sort by timestamp (newest first)
        messages = sorted(messages, key=lambda m: m.get("timestamp", ""), reverse=True)

        if limit:
            messages = messages[:limit]

        return messages

    def get_human_messages(
        self,
        since_generation: int | None = None,
        priority: str = "info",
        limit: int = 50,
        current_generation: int = 0,
    ) -> list[dict]:
        """
        Get messages intended for the human operator.

        Convenience method for displaying the operator feed.

        Args:
            since_generation: Only show messages from this generation
            priority: Minimum priority level
            limit: Maximum messages
            current_generation: For TTL filtering

        Returns:
            List of messages for human
        """
        return self.get_messages(
            audience="human",
            since_generation=since_generation,
            priority=priority,
            limit=limit,
            current_generation=current_generation,
        )

    def get_active_strategies(self, generation: int) -> list[str]:
        """
        Get strategies claimed by agents for the current generation.

        Used by mutators to avoid duplicating work.

        Args:
            generation: Current generation

        Returns:
            List of claimed strategy descriptions
        """
        messages = self.get_messages(
            message_type="strategy",
            since_generation=generation,
            include_expired=False,
            current_generation=generation,
        )
        return [m.get("content", "") for m in messages if m.get("content")]

    def format_message_for_display(self, message: dict) -> str:
        """
        Format a message for terminal display.

        Args:
            message: Message frame dict

        Returns:
            Formatted string for display
        """
        emoji = message.get("agent_emoji", "💬")
        agent = message.get("from_agent", "unknown")
        title = message.get("title", "")
        priority = message.get("priority", "info")
        gen = message.get("generation", 0)
        fitness = message.get("related_fitness")

        # Priority indicators
        priority_prefix = {
            "debug": "  ",
            "info": "  ",
            "important": "❗",
            "urgent": "🔔",
            "critical": "🚨",
        }.get(priority, "  ")

        # Build the display line
        line = f"{priority_prefix} {emoji} [{agent}] {title}"

        if fitness is not None:
            line += f" (fitness: {fitness:,.0f})"

        if gen > 0:
            line += f" [gen{gen}]"

        return line

    def get_message_feed(
        self,
        limit: int = 20,
        current_generation: int = 0,
        priority: str = "info",
    ) -> str:
        """
        Get a formatted message feed for display.

        Args:
            limit: Number of messages
            current_generation: For TTL filtering
            priority: Minimum priority

        Returns:
            Formatted multi-line string
        """
        messages = self.get_human_messages(
            priority=priority,
            limit=limit,
            current_generation=current_generation,
        )

        if not messages:
            return "No messages yet."

        lines = []
        for msg in messages:
            lines.append(self.format_message_for_display(msg))

        return "\n".join(lines)

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
