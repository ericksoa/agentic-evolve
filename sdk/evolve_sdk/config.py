"""Configuration for evolution runs."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any


@dataclass
class TrustConfig:
    """Configuration for trust-aware evolution with Adversary agent."""

    # Enable/disable adversary
    enabled: bool = True

    # Trust thresholds
    accept_threshold: float = 0.7  # Trust >= this: accept without escalation
    challenge_threshold: float = 0.4  # Trust < accept but >= this: escalate
    reject_threshold: float = 0.4  # Trust < this: reject outright

    # Fitness jump that triggers automatic suspicion
    suspicious_jump_pct: float = 20.0  # >20% improvement triggers review

    # Escalation settings
    max_escalation_level: int = 2  # Maximum escalation (0-3)
    extended_test_command: str | None = None  # Command for Level 1+ validation

    # Trust adjustment
    apply_trust_adjustment: bool = True  # Multiply fitness by trust_score

    # Champion gating
    require_adversary_for_champion: bool = True  # New champions must pass adversary

    # === NEW: Variance Gates (A) ===
    # Re-evaluate N times and check consistency
    n_evaluations: int = 1  # Number of evaluation runs (1 = disabled, 3+ recommended)
    variance_threshold: float = 0.05  # Max acceptable coefficient of variation
    require_variance_gate: bool = False  # Reject if variance exceeds threshold

    # === NEW: Canary Test (D) ===
    # Inject known-bad candidate at startup to verify trust system works
    canary_test_enabled: bool = False  # Run canary test before evolution
    canary_test_strict: bool = True  # Fail evolution if canary passes (it shouldn't)

    # === NEW: Exploit Detection (B) - non-sandboxing checks ===
    # Basic exploit detection without sandboxing
    check_timing_anomaly: bool = True  # Flag suspiciously fast evaluations
    timing_anomaly_threshold_ms: float = 10.0  # Below this is suspicious
    check_output_integrity: bool = True  # Check for NaN, extreme values
    output_max_value: float = 1e10  # Fitness above this is rejected
    check_eval_determinism: bool = True  # Compare N evals for consistency

    # === NEW: Trust Dossier (E) ===
    # Generate markdown reports for trust decisions
    generate_dossier: bool = True  # Create trust_dossier.md after champion decisions
    dossier_include_history: bool = True  # Include full trust history in dossier

    # === NEW: Validator Interface (C) ===
    # Pluggable validators for escalation
    validators: list[str] = field(default_factory=lambda: ["default"])  # Validator chain

    # === NEW: Human-in-the-Loop Escalation ===
    # Allow human to override trust decisions on borderline cases
    human_escalation_enabled: bool = True  # Enable human escalation prompts
    human_escalation_on_champion_reject: bool = True  # Ask human when rejecting a potential champion
    human_escalation_min_fitness: float = 0.0  # Only ask for candidates with fitness > this
    human_escalation_timeout: int = 300  # Timeout in seconds (0 = no timeout)


@dataclass
class MemoryConfig:
    """Configuration for evolution memory system."""

    # Enable/disable memory
    enabled: bool = True

    # Storage paths
    store_path: str = ".evolve-sdk/{problem_id}/memory.json"  # .mv2 when memvid available

    # Embedding settings
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

    # Checkpoint settings
    checkpoint_on_mutation: bool = True
    checkpoint_on_evaluation: bool = True
    checkpoint_on_selection: bool = True

    # Mutation memory settings
    store_failed_mutations: bool = True
    store_successful_mutations: bool = True
    inject_mutation_context: bool = True  # Inject memory context into mutator prompts
    max_similar_mutations: int = 3  # Max similar mutations to show
    max_failed_mutations: int = 5  # Max failed mutations to warn about

    # Champion library
    store_champions: bool = True
    bootstrap_from_similar: bool = False  # Bootstrap from similar past problems

    # Exploit memory
    store_exploits: bool = True
    check_exploit_similarity: bool = True

    # Retention
    max_frames: int = 100000
    prune_strategy: str = "lru"  # lru, fifo, none


@dataclass
class StrategicConfig:
    """Configuration for strategic direction analysis (Meta-Strategist extension)."""

    # Enable/disable strategic direction analysis
    enabled: bool = True

    # Trigger interval (analyze every N generations)
    direction_interval: int = 10

    # Auto-trigger when tactical adjustments stall
    auto_trigger_on_stall: bool = True
    tactical_stall_threshold: int = 2  # Trigger after N stalled tactical analyses

    # Direction generation settings
    max_candidate_directions: int = 5
    include_historical_breakthroughs: bool = True

    # Risk tolerance affects ranking (low=prefer safe, high=prefer bold)
    risk_tolerance: str = "medium"  # low|medium|high


@dataclass
class DiversityConfig:
    """Configuration for Diversity Guardian agent (Phase 3)."""

    # Enable/disable diversity monitoring
    enabled: bool = True

    # Diversity thresholds (alert when below)
    genotypic_threshold: float = 0.25  # Code similarity diversity
    phenotypic_threshold: float = 0.20  # Fitness spread diversity

    # Check frequency
    check_interval: int = 1  # Check every N generations (1 = every generation)

    # Intervention settings
    max_injections_per_alert: int = 2  # Max orthogonal solutions to inject
    auto_inject: bool = True  # Automatically inject orthogonal solutions on alert

    # Weights for combined diversity score
    genotypic_weight: float = 0.6  # Weight for code similarity
    phenotypic_weight: float = 0.4  # Weight for fitness spread

    # Alert levels
    warning_multiplier: float = 1.0  # Alert at threshold * this
    critical_multiplier: float = 0.6  # Critical at threshold * this


@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""

    # Problem definition
    problem: str
    mode: Literal["size", "perf", "ml"] = "size"
    minimize: bool = False  # If True, lower fitness is better (e.g., nodes expanded, error rate)

    # Budget controls
    max_generations: int = 50
    plateau_threshold: int = 5  # Stop after N generations without improvement

    # Population settings
    population_size: int = 10
    elite_count: int = 3  # Top N to keep each generation
    mutation_variants: int = 4  # Mutations per generation

    # Agent settings
    max_turns_per_agent: int = 15
    model: str = "claude-opus-4-5-20251101"  # Use Opus 4.5 for best quality
    orchestrator_model: str = "claude-opus-4-5-20251101"  # Main orchestrator

    # Paths
    evolve_dir: Path = field(default_factory=lambda: Path(".evolve-sdk"))

    # Evaluation settings (loaded from evolve_config.json if present)
    test_command: str | None = None  # e.g., "python evaluate_on_lightning.py {solution} --json"
    starter_solutions: list[str] = field(default_factory=list)
    optimization_strategies: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # Trust-aware evolution (Adversary agent)
    trust: TrustConfig = field(default_factory=TrustConfig)

    # Memory system (crash recovery, mutation patterns, cross-problem learning)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Diversity monitoring (Diversity Guardian agent - Phase 3)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)

    # Strategic direction analysis (Meta-Strategist extension)
    strategic: StrategicConfig = field(default_factory=StrategicConfig)

    # Safety
    enable_validation_hooks: bool = True
    blocked_imports: list[str] = field(default_factory=lambda: [
        "os.system", "subprocess", "eval(", "exec(", "__import__",
        "open(", "urllib", "requests", "socket"
    ])

    # Parallelism
    parallel_mutations: bool = True

    def __post_init__(self):
        if isinstance(self.evolve_dir, str):
            self.evolve_dir = Path(self.evolve_dir)

    def is_better(self, a: float, b: float) -> bool:
        """Return True if fitness `a` is better than fitness `b`."""
        if self.minimize:
            return a < b
        return a > b

    def sort_key_reverse(self) -> bool:
        """Return the `reverse` parameter for sorting population by fitness (best first)."""
        return not self.minimize

    @classmethod
    def from_config_file(cls, config_path: Path | str, **overrides) -> "EvolutionConfig":
        """Load config from evolve_config.json file with optional overrides."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        # Map evolve_config.json fields to EvolutionConfig fields
        problem = data.get("description", data.get("problem", {}).get("goal", "Unknown problem"))
        mode = data.get("mode", "size")

        # Extract evaluation config
        eval_config = data.get("evaluation", {})
        test_command = eval_config.get("test_command")

        # Extract cwd for working directory
        cwd = data.get("cwd", overrides.get("cwd"))

        # Extract trust config if present
        trust_data = data.get("trust", {})
        trust_config = TrustConfig(
            enabled=trust_data.get("enabled", True),
            accept_threshold=trust_data.get("accept_threshold", 0.7),
            challenge_threshold=trust_data.get("challenge_threshold", 0.4),
            reject_threshold=trust_data.get("reject_threshold", 0.4),
            suspicious_jump_pct=trust_data.get("suspicious_jump_pct", 20.0),
            max_escalation_level=trust_data.get("max_escalation_level", 2),
            extended_test_command=trust_data.get("extended_test_command"),
            apply_trust_adjustment=trust_data.get("apply_trust_adjustment", True),
            require_adversary_for_champion=trust_data.get("require_adversary_for_champion", True),
            # Variance Gates (A)
            n_evaluations=trust_data.get("n_evaluations", 1),
            variance_threshold=trust_data.get("variance_threshold", 0.05),
            require_variance_gate=trust_data.get("require_variance_gate", False),
            # Canary Test (D)
            canary_test_enabled=trust_data.get("canary_test_enabled", False),
            canary_test_strict=trust_data.get("canary_test_strict", True),
            # Exploit Detection (B)
            check_timing_anomaly=trust_data.get("check_timing_anomaly", True),
            timing_anomaly_threshold_ms=trust_data.get("timing_anomaly_threshold_ms", 10.0),
            check_output_integrity=trust_data.get("check_output_integrity", True),
            output_max_value=trust_data.get("output_max_value", 1e10),
            check_eval_determinism=trust_data.get("check_eval_determinism", True),
            # Trust Dossier (E)
            generate_dossier=trust_data.get("generate_dossier", True),
            dossier_include_history=trust_data.get("dossier_include_history", True),
            # Validator Interface (C)
            validators=trust_data.get("validators", ["default"]),
            # Human-in-the-Loop Escalation
            human_escalation_enabled=trust_data.get("human_escalation_enabled", True),
            human_escalation_on_champion_reject=trust_data.get("human_escalation_on_champion_reject", True),
            human_escalation_min_fitness=trust_data.get("human_escalation_min_fitness", 0.0),
            human_escalation_timeout=trust_data.get("human_escalation_timeout", 300),
        )

        # Extract memory config if present
        memory_data = data.get("memory", {})
        memory_config = MemoryConfig(
            enabled=memory_data.get("enabled", True),
            store_path=memory_data.get("store_path", ".evolve-sdk/{problem_id}/memory.json"),
            use_embeddings=memory_data.get("use_embeddings", True),
            embedding_model=memory_data.get("embedding_model", "all-MiniLM-L6-v2"),
            checkpoint_on_mutation=memory_data.get("checkpoint_on_mutation", True),
            checkpoint_on_evaluation=memory_data.get("checkpoint_on_evaluation", True),
            checkpoint_on_selection=memory_data.get("checkpoint_on_selection", True),
            store_failed_mutations=memory_data.get("store_failed_mutations", True),
            store_successful_mutations=memory_data.get("store_successful_mutations", True),
            inject_mutation_context=memory_data.get("inject_mutation_context", True),
            max_similar_mutations=memory_data.get("max_similar_mutations", 3),
            max_failed_mutations=memory_data.get("max_failed_mutations", 5),
            store_champions=memory_data.get("store_champions", True),
            bootstrap_from_similar=memory_data.get("bootstrap_from_similar", False),
            store_exploits=memory_data.get("store_exploits", True),
            check_exploit_similarity=memory_data.get("check_exploit_similarity", True),
            max_frames=memory_data.get("max_frames", 100000),
            prune_strategy=memory_data.get("prune_strategy", "lru"),
        )

        # Extract diversity config if present (Phase 3)
        diversity_data = data.get("diversity", {})
        diversity_config = DiversityConfig(
            enabled=diversity_data.get("enabled", True),
            genotypic_threshold=diversity_data.get("genotypic_threshold", 0.25),
            phenotypic_threshold=diversity_data.get("phenotypic_threshold", 0.20),
            check_interval=diversity_data.get("check_interval", 1),
            max_injections_per_alert=diversity_data.get("max_injections_per_alert", 2),
            auto_inject=diversity_data.get("auto_inject", True),
            genotypic_weight=diversity_data.get("genotypic_weight", 0.6),
            phenotypic_weight=diversity_data.get("phenotypic_weight", 0.4),
            warning_multiplier=diversity_data.get("warning_multiplier", 1.0),
            critical_multiplier=diversity_data.get("critical_multiplier", 0.6),
        )

        # Extract strategic config if present
        strategic_data = data.get("strategic", {})
        strategic_config = StrategicConfig(
            enabled=strategic_data.get("enabled", True),
            direction_interval=strategic_data.get("direction_interval", 10),
            auto_trigger_on_stall=strategic_data.get("auto_trigger_on_stall", True),
            tactical_stall_threshold=strategic_data.get("tactical_stall_threshold", 2),
            max_candidate_directions=strategic_data.get("max_candidate_directions", 5),
            include_historical_breakthroughs=strategic_data.get("include_historical_breakthroughs", True),
            risk_tolerance=strategic_data.get("risk_tolerance", "medium"),
        )

        # Extract minimize flag
        minimize = data.get("minimize", False)

        # Build config with file data and overrides
        config = cls(
            problem=overrides.get("problem", problem),
            mode=overrides.get("mode", mode),
            minimize=overrides.get("minimize", minimize),
            max_generations=overrides.get("max_generations", 50),
            plateau_threshold=overrides.get("plateau_threshold", 5),
            population_size=overrides.get("population_size", 10),
            evolve_dir=overrides.get("evolve_dir", Path(".evolve-sdk")),
            test_command=test_command,
            starter_solutions=data.get("starter_solutions", []),
            optimization_strategies=data.get("optimization_strategies", []),
            constraints=data.get("constraints", []),
            references=data.get("references", []),
            model=overrides.get("model", "claude-opus-4-5-20251101"),
            trust=trust_config,
            memory=memory_config,
            diversity=diversity_config,
            strategic=strategic_config,
        )
        # Store cwd as extra attribute for runner
        config.cwd = cwd
        return config
