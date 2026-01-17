"""
Project-level memory schemas for evolve-sdk.

These schemas are for tracking project-level information that persists
across evolution runs, such as:
- Don't Repeat Incidents (DRIs) - hard-learned lessons
- Submissions - competition/benchmark submission records
- Model Evaluations - evaluation results with evidence requirements

Unlike the evolution-specific schemas in schemas.py (checkpoints, mutations, etc.),
these schemas track institutional knowledge and project history.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class DRI:
    """
    Don't Repeat Incident - a hard-learned lesson from a past failure.

    DRIs capture mistakes that wasted submissions, time, or resources.
    They should be consulted BEFORE any major action to avoid repeating
    past incidents.

    Example:
        DRI(
            id="DRI-001",
            title="Template override broke model output",
            root_cause="Simplified chat_template.jinja didn't inject system prompt",
            submissions_wasted=3,
            rules_to_follow=["Always use golden_chat_template.jinja", "Never simplify templates"],
            verification_commands=["curl HF_URL/chat_template.jinja | wc -c  # Should be >2000"],
        )
    """
    id: str
    title: str
    root_cause: str  # WHY it failed - most important field
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    submissions_wasted: int = 0
    time_wasted_hours: float = 0.0
    rules_to_follow: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    related_actions: list[str] = field(default_factory=list)  # Actions that trigger this DRI
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["_schema_type"] = "DRI"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DRI":
        """Create from dictionary."""
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)

    def __str__(self) -> str:
        return f"[{self.id}] {self.title}"


@dataclass
class Submission:
    """
    Competition or benchmark submission record.

    Tracks submission history including status, metrics, and any failures.
    Used to learn from past submissions and avoid repeating mistakes.

    Example:
        Submission(
            id="307752",
            model_path="ericksoa/chess-v6-aicrowd",
            metrics={"acpl": 146.2, "matches": 50},
            status="succeeded",
        )
    """
    id: str
    model_path: str
    status: str  # pending, succeeded, failed, timeout
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    submission_url: str = ""
    failure_reason: str | None = None
    fixes_applied: list[str] = field(default_factory=list)
    verification_done: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["_schema_type"] = "Submission"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Submission":
        """Create from dictionary."""
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def __str__(self) -> str:
        status_emoji = {"succeeded": "+", "failed": "x", "pending": "?", "timeout": "!"}
        return f"[{status_emoji.get(self.status, '?')}] {self.id}: {self.model_path}"


@dataclass
class ModelEvaluation:
    """
    Model evaluation result with evidence requirements.

    Enforces minimum sample sizes to prevent conclusions from insufficient data.
    This addresses DRI-011: "4-position evaluation concluded model was broken
    (actually 2.5x better when tested on 100 positions)".

    Example:
        ModelEvaluation(
            model_name="chess-v6-rs-merged",
            metrics={"acpl": 78.1, "win_rate": 0.65},
            sample_size=100,  # MUST be >= 100
            methodology="Stockfish Level 20 evaluation on 100 random positions",
            verdict="Model is competitive with baseline (71.9 ACPL)",
        )
    """
    model_name: str
    metrics: dict[str, Any]
    sample_size: int  # CRITICAL: Must be >= 100 for conclusions
    methodology: str
    verdict: str
    eval_date: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)  # Model/eval config used
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # Evidence requirements
    MINIMUM_SAMPLE_SIZE = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["_schema_type"] = "ModelEvaluation"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelEvaluation":
        """Create from dictionary."""
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)

    def validate(self) -> list[str]:
        """
        Validate the evaluation meets evidence requirements.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        if self.sample_size < self.MINIMUM_SAMPLE_SIZE:
            errors.append(
                f"Sample size {self.sample_size} < minimum {self.MINIMUM_SAMPLE_SIZE}. "
                f"Conclusions from small samples are unreliable (see DRI-011)."
            )
        if not self.methodology:
            errors.append("Methodology is required - describe how evaluation was conducted.")
        if not self.verdict:
            errors.append("Verdict is required - what conclusion can be drawn?")
        return errors

    def __str__(self) -> str:
        primary_metric = next(iter(self.metrics.values()), "?") if self.metrics else "?"
        return f"{self.model_name}: {primary_metric} (n={self.sample_size})"


@dataclass
class ConfigChange:
    """
    Record of a configuration change with evidence.

    Addresses DRI-002: "Made up config values without measurement".
    Requires evidence for any configuration change.

    Example:
        ConfigChange(
            parameter="max_tokens",
            before_value=2048,
            after_value=64,
            evidence="Training data analysis: p99 response length = 29 tokens",
            rationale="Reduce timeout risk by limiting to measured maximum",
        )
    """
    parameter: str
    before_value: Any
    after_value: Any
    evidence: str  # REQUIRED: Measured data supporting the change
    rationale: str
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""  # Where the evidence came from
    verified: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["_schema_type"] = "ConfigChange"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigChange":
        """Create from dictionary."""
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        return cls(**data)

    def validate(self) -> list[str]:
        """Validate the config change has required evidence."""
        errors = []
        if not self.evidence:
            errors.append("Evidence is required - what data supports this change?")
        if not self.rationale:
            errors.append("Rationale is required - why is this change being made?")
        return errors

    def __str__(self) -> str:
        return f"{self.parameter}: {self.before_value} -> {self.after_value}"


# Schema type registry for deserialization
SCHEMA_TYPES = {
    "DRI": DRI,
    "Submission": Submission,
    "ModelEvaluation": ModelEvaluation,
    "ConfigChange": ConfigChange,
}


def from_dict(data: dict[str, Any]) -> DRI | Submission | ModelEvaluation | ConfigChange:
    """
    Deserialize a dictionary to the appropriate schema type.

    Args:
        data: Dictionary with _schema_type field

    Returns:
        Instance of the appropriate schema class

    Raises:
        ValueError: If _schema_type is missing or unknown
    """
    schema_type = data.get("_schema_type")
    if not schema_type:
        raise ValueError("Missing _schema_type field in data")

    cls = SCHEMA_TYPES.get(schema_type)
    if not cls:
        raise ValueError(f"Unknown schema type: {schema_type}")

    return cls.from_dict(data)
