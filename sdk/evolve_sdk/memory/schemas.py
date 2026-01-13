"""
Frame schemas for evolution memory.

Defines the structure of different frame types stored in the memory system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class FrameType(str, Enum):
    """Types of frames stored in evolution memory."""
    CHECKPOINT = "checkpoint"  # Evolution state snapshots
    MUTATION = "mutation"  # Successful mutation patterns
    FAILED_MUTATION = "failed_mutation"  # Mutations that didn't improve
    CHAMPION = "champion"  # Champion solutions
    EXPLOIT = "exploit"  # Detected exploit patterns
    TRUST_DECISION = "trust_decision"  # Adversary decisions
    HUMAN_DECISION = "human_decision"  # Human escalation decisions
    CONFIG = "config"  # Successful evolution configs (meta-evolution)
    GENERATION = "generation"  # Per-generation snapshots
    NOTE = "note"  # Institutional knowledge and facts to remember


@dataclass
class BaseFrame:
    """Base class for all frames."""
    frame_type: FrameType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    problem_id: str = ""
    mode: str = ""  # size, perf, ml

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["frame_type"] = self.frame_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseFrame":
        """Create from dictionary."""
        data = data.copy()
        if "frame_type" in data:
            data["frame_type"] = FrameType(data["frame_type"])
        return cls(**data)


@dataclass
class CheckpointFrame(BaseFrame):
    """
    Checkpoint frame for crash recovery.

    Stored after every significant operation to enable resumption.
    """
    frame_type: FrameType = field(default=FrameType.CHECKPOINT)
    generation: int = 0
    operation: str = ""  # post_mutation, post_evaluation, post_selection
    population_json: str = ""  # JSON-serialized population
    champion_file: str | None = None
    champion_fitness: float = 0.0
    plateau_count: int = 0
    notes: str = ""


@dataclass
class MutationFrame(BaseFrame):
    """
    Mutation pattern frame.

    Stores successful mutations for cross-problem learning.
    """
    frame_type: FrameType = field(default=FrameType.MUTATION)
    generation: int = 0
    variant: str = ""  # a, b, c, x (crossover)
    parent_file: str = ""
    child_file: str = ""
    parent_fitness: float = 0.0
    child_fitness: float = 0.0
    fitness_delta: float = 0.0
    fitness_delta_pct: float = 0.0
    diff_content: str = ""  # Code diff between parent and child
    mutation_type: str = ""  # mutation, crossover
    tags: list[str] = field(default_factory=list)  # e.g., ["hyperparameter_tune", "ensemble"]
    broke_plateau: bool = False  # Did this mutation break a plateau?
    became_champion: bool = False


@dataclass
class FailedMutationFrame(BaseFrame):
    """
    Failed mutation frame.

    Stores mutations that didn't improve fitness to avoid repeating them.
    """
    frame_type: FrameType = field(default=FrameType.FAILED_MUTATION)
    generation: int = 0
    parent_file: str = ""
    parent_hash: str = ""  # Hash of parent code for quick lookup
    child_file: str = ""
    parent_fitness: float = 0.0
    child_fitness: float = 0.0
    fitness_delta: float = 0.0
    diff_content: str = ""
    failure_reason: str = ""  # overfitting, invalid, worse, rejected_by_trust


@dataclass
class ChampionFrame(BaseFrame):
    """
    Champion solution frame.

    Stores champion solutions for cross-problem bootstrapping.
    """
    frame_type: FrameType = field(default=FrameType.CHAMPION)
    file_path: str = ""
    code_content: str = ""
    fitness: float = 0.0
    generation: int = 0
    improvement_over_baseline: float = 0.0
    trust_score: float = 1.0
    cv_mean: float = 0.0  # Cross-validation mean
    cv_std: float = 0.0  # Cross-validation std
    insights: list[str] = field(default_factory=list)
    problem_description: str = ""


@dataclass
class ExploitFrame(BaseFrame):
    """
    Exploit pattern frame.

    Stores detected exploit patterns for trust system learning.
    """
    frame_type: FrameType = field(default=FrameType.EXPLOIT)
    file_path: str = ""
    code_content: str = ""
    pattern_type: str = ""  # hardcoded, timing, extreme_jump, nan
    detection_method: str = ""  # canary_test, adversary, exploit_detector
    flags: list[str] = field(default_factory=list)
    claimed_fitness: float = 0.0
    actual_fitness: float = 0.0


@dataclass
class TrustDecisionFrame(BaseFrame):
    """
    Trust decision frame.

    Stores adversary decisions for calibration learning.
    """
    frame_type: FrameType = field(default=FrameType.TRUST_DECISION)
    file_path: str = ""
    generation: int = 0
    claimed_fitness: float = 0.0
    trust_score: float = 0.0
    recommendation: str = ""  # accept, challenge, reject
    flags: list[str] = field(default_factory=list)
    escalation_level: int = 0
    final_outcome: str = ""  # accepted, rejected, human_override
    was_correct: bool | None = None  # Did the decision turn out to be correct?


@dataclass
class HumanDecisionFrame(BaseFrame):
    """
    Human escalation decision frame.

    Stores human override decisions for threshold learning.
    """
    frame_type: FrameType = field(default=FrameType.HUMAN_DECISION)
    file_path: str = ""
    generation: int = 0
    fitness: float = 0.0
    original_trust: float = 0.0
    adversary_recommendation: str = ""
    flags: list[str] = field(default_factory=list)
    human_decision: str = ""  # accept, reject
    adjusted_trust: float | None = None
    reasoning: str = ""


@dataclass
class ConfigFrame(BaseFrame):
    """
    Evolution config frame.

    Stores successful evolution configurations for meta-learning.
    """
    frame_type: FrameType = field(default=FrameType.CONFIG)
    config_json: str = ""
    problem_type: str = ""  # small_dataset_ml, code_golf, algorithm_optimization
    success: bool = False
    final_fitness: float = 0.0
    improvement_pct: float = 0.0
    generations_used: int = 0
    plateau_count: int = 0
    key_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationFrame(BaseFrame):
    """
    Generation snapshot frame.

    Stores per-generation state for temporal queries.
    """
    frame_type: FrameType = field(default=FrameType.GENERATION)
    generation: int = 0
    population_json: str = ""
    best_fitness: float = 0.0
    champion_file: str | None = None
    mutations_tried: int = 0
    mutations_kept: int = 0
    plateau_count: int = 0
    improvement_from_previous: float = 0.0


@dataclass
class NoteFrame(BaseFrame):
    """
    Note frame for institutional knowledge.

    Stores important facts, constraints, and lessons learned that should
    persist across conversations and evolution runs.
    """
    frame_type: FrameType = field(default=FrameType.NOTE)
    category: str = ""  # constraint, fact, lesson, goal, warning
    title: str = ""  # Short summary
    content: str = ""  # Full note content
    priority: str = "normal"  # critical, high, normal, low
    tags: list[str] = field(default_factory=list)
    verified: bool = True  # Has this been verified?
    source: str = ""  # Where this info came from (human, evaluation, etc.)
