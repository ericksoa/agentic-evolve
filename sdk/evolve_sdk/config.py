"""Configuration for evolution runs."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""

    # Problem definition
    problem: str
    mode: Literal["size", "perf", "ml"] = "size"

    # Budget controls
    max_generations: int = 50
    plateau_threshold: int = 5  # Stop after N generations without improvement

    # Population settings
    population_size: int = 10
    elite_count: int = 3  # Top N to keep each generation
    mutation_variants: int = 4  # Mutations per generation

    # Agent settings
    max_turns_per_agent: int = 15
    model: str = "claude-sonnet-4-20250514"  # Use sonnet for subagents (cost effective)
    orchestrator_model: str = "claude-sonnet-4-20250514"  # Main orchestrator

    # Paths
    evolve_dir: Path = field(default_factory=lambda: Path(".evolve-sdk"))

    # Evaluation settings (loaded from evolve_config.json if present)
    test_command: str | None = None  # e.g., "python evaluate_on_lightning.py {solution} --json"
    starter_solutions: list[str] = field(default_factory=list)
    optimization_strategies: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

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

        # Build config with file data and overrides
        return cls(
            problem=overrides.get("problem", problem),
            mode=overrides.get("mode", mode),
            max_generations=overrides.get("max_generations", 50),
            plateau_threshold=overrides.get("plateau_threshold", 5),
            population_size=overrides.get("population_size", 10),
            evolve_dir=overrides.get("evolve_dir", Path(".evolve-sdk")),
            test_command=test_command,
            starter_solutions=data.get("starter_solutions", []),
            optimization_strategies=data.get("optimization_strategies", []),
            constraints=data.get("constraints", []),
            references=data.get("references", []),
            model=overrides.get("model", "claude-sonnet-4-20250514"),
        )
