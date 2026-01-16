"""
cuopt_config.py - Real cuOpt parameter configuration and genome.

Based on NVIDIA cuOpt documentation:
https://docs.nvidia.com/cuopt/user-guide/latest/lp-milp-settings.html

This module defines the actual tunable parameters for cuOpt LP/MILP solving,
their valid ranges, and the genome representation for evolution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import copy
import random
import math


class SolverMethod(Enum):
    """LP solving methods available in cuOpt."""
    CONCURRENT = 0      # Run PDLP and Dual Simplex concurrently
    PDLP = 1            # Primal-Dual Hybrid Gradient (GPU, large-scale)
    DUAL_SIMPLEX = 2    # Classic simplex (CPU, smaller problems)
    BARRIER = 3         # Interior point method


class PDLPSolverMode(Enum):
    """PDLP internal optimization modes."""
    STABLE3 = "Stable3"         # Most stable, default
    METHODICAL1 = "Methodical1" # Balanced
    FAST1 = "Fast1"             # Fastest but less stable


class BarrierOrdering(Enum):
    """Barrier solver ordering methods."""
    AUTO = -1
    CUDSS = 0
    AMD = 1


# All tunable cuOpt parameters with their domains
CUOPT_PARAMETER_DOMAINS = {
    # Method selection
    "method": {
        "type": "enum",
        "enum_class": SolverMethod,
        "default": SolverMethod.CONCURRENT,
        "description": "LP solving method",
    },
    "pdlp_solver_mode": {
        "type": "enum",
        "enum_class": PDLPSolverMode,
        "default": PDLPSolverMode.STABLE3,
        "description": "PDLP internal mode",
    },

    # Presolve and preprocessing
    "presolve": {
        "type": "bool",
        "default": False,  # cuOpt default for LP
        "description": "Enable presolve",
    },
    "dualize": {
        "type": "int",
        "min": -1, "max": 1,
        "default": -1,
        "description": "Dualize LP in presolve (-1=auto, 0=no, 1=yes)",
    },

    # Crossover and solution refinement
    "crossover": {
        "type": "bool",
        "default": False,
        "description": "Crossover to basic solution after PDLP",
    },
    "save_best_primal": {
        "type": "bool",
        "default": False,
        "description": "Prioritize primal feasible solutions",
    },
    "first_primal_feasible": {
        "type": "bool",
        "default": False,
        "description": "Stop at first feasible solution",
    },

    # Infeasibility handling
    "infeasibility_detection": {
        "type": "bool",
        "default": False,
        "description": "Enable PDLP infeasibility detection",
    },
    "strict_infeasibility": {
        "type": "bool",
        "default": False,
        "description": "Stop if infeasibility detected",
    },

    # Tolerance parameters (PDLP)
    "absolute_primal_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Absolute primal feasibility tolerance",
    },
    "relative_primal_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Relative primal feasibility tolerance",
    },
    "absolute_dual_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Absolute dual feasibility tolerance",
    },
    "relative_dual_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Relative dual feasibility tolerance",
    },
    "absolute_gap_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Absolute duality gap tolerance",
    },
    "relative_gap_tolerance": {
        "type": "log_float",
        "min": 1e-8, "max": 1e-2,
        "default": 1e-4,
        "description": "Relative duality gap tolerance",
    },

    # Iteration and time limits
    "iteration_limit": {
        "type": "int",
        "min": 1000, "max": 10_000_000,
        "default": 1_000_000,
        "description": "Maximum iterations",
    },
    "time_limit": {
        "type": "float",
        "min": 1.0, "max": 3600.0,
        "default": 300.0,
        "description": "Time limit in seconds",
    },

    # Threading
    "num_cpu_threads": {
        "type": "int",
        "min": 1, "max": 32,
        "default": 4,
        "description": "CPU threads for solver",
    },
    "num_gpus": {
        "type": "int",
        "min": 1, "max": 2,
        "default": 1,
        "description": "GPUs for concurrent solving",
    },

    # Barrier-specific settings
    "barrier_folding": {
        "type": "int",
        "min": -1, "max": 1,
        "default": -1,
        "description": "Barrier folding (-1=auto, 0=off, 1=on)",
    },
    "barrier_ordering": {
        "type": "int",
        "min": -1, "max": 1,
        "default": -1,
        "description": "Barrier ordering method",
    },
    "barrier_augmented": {
        "type": "int",
        "min": -1, "max": 1,
        "default": -1,
        "description": "Barrier augmented system",
    },
    "eliminate_dense_columns": {
        "type": "bool",
        "default": True,
        "description": "Remove dense columns before barrier",
    },
    "barrier_deterministic": {
        "type": "bool",
        "default": False,
        "description": "Deterministic barrier for reproducibility",
    },

    # Residual computation
    "per_constraint_residual": {
        "type": "bool",
        "default": False,
        "description": "Per-constraint vs global residual",
    },
}


@dataclass
class CuOptConfig:
    """
    Configuration for cuOpt solver parameters.

    This is the "genome" that gets evolved to optimize cuOpt performance.
    """
    # Method selection
    method: SolverMethod = SolverMethod.CONCURRENT
    pdlp_solver_mode: PDLPSolverMode = PDLPSolverMode.STABLE3

    # Preprocessing
    presolve: bool = False
    dualize: int = -1

    # Solution settings
    crossover: bool = False
    save_best_primal: bool = False
    first_primal_feasible: bool = False

    # Infeasibility
    infeasibility_detection: bool = False
    strict_infeasibility: bool = False

    # Tolerances
    absolute_primal_tolerance: float = 1e-4
    relative_primal_tolerance: float = 1e-4
    absolute_dual_tolerance: float = 1e-4
    relative_dual_tolerance: float = 1e-4
    absolute_gap_tolerance: float = 1e-4
    relative_gap_tolerance: float = 1e-4

    # Limits
    iteration_limit: int = 1_000_000
    time_limit: float = 300.0

    # Resources
    num_cpu_threads: int = 4
    num_gpus: int = 1

    # Barrier settings
    barrier_folding: int = -1
    barrier_ordering: int = -1
    barrier_augmented: int = -1
    eliminate_dense_columns: bool = True
    barrier_deterministic: bool = False

    # Residuals
    per_constraint_residual: bool = False

    def to_cuopt_params(self) -> Dict[str, Any]:
        """Convert to cuOpt parameter dictionary."""
        params = {
            "CUOPT_METHOD": self.method.value,
            "CUOPT_PDLP_SOLVER_MODE": self.pdlp_solver_mode.value,
            "CUOPT_PRESOLVE": self.presolve,
            "CUOPT_DUALIZE": self.dualize,
            "CUOPT_CROSSOVER": self.crossover,
            "CUOPT_SAVE_BEST_PRIMAL_SOLUTION": self.save_best_primal,
            "CUOPT_FIRST_PRIMAL_FEASIBLE": self.first_primal_feasible,
            "CUOPT_INFEASIBILITY_DETECTION": self.infeasibility_detection,
            "CUOPT_STRICT_INFEASIBILITY": self.strict_infeasibility,
            "CUOPT_ABSOLUTE_PRIMAL_TOLERANCE": self.absolute_primal_tolerance,
            "CUOPT_RELATIVE_PRIMAL_TOLERANCE": self.relative_primal_tolerance,
            "CUOPT_ABSOLUTE_DUAL_TOLERANCE": self.absolute_dual_tolerance,
            "CUOPT_RELATIVE_DUAL_TOLERANCE": self.relative_dual_tolerance,
            "CUOPT_ABSOLUTE_GAP_TOLERANCE": self.absolute_gap_tolerance,
            "CUOPT_RELATIVE_GAP_TOLERANCE": self.relative_gap_tolerance,
            "CUOPT_ITERATION_LIMIT": self.iteration_limit,
            "CUOPT_TIME_LIMIT": self.time_limit,
            "CUOPT_NUM_CPU_THREADS": self.num_cpu_threads,
            "CUOPT_NUM_GPUS": self.num_gpus,
            "CUOPT_FOLDING": self.barrier_folding,
            "CUOPT_ORDERING": self.barrier_ordering,
            "CUOPT_AUGMENTED": self.barrier_augmented,
            "CUOPT_ELIMINATE_DENSE_COLUMNS": self.eliminate_dense_columns,
            "CUOPT_CUDSS_DETERMINISTIC": self.barrier_deterministic,
            "CUOPT_PER_CONSTRAINT_RESIDUAL": self.per_constraint_residual,
        }
        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "method": self.method.value,
            "pdlp_solver_mode": self.pdlp_solver_mode.value,
            "presolve": self.presolve,
            "dualize": self.dualize,
            "crossover": self.crossover,
            "save_best_primal": self.save_best_primal,
            "first_primal_feasible": self.first_primal_feasible,
            "infeasibility_detection": self.infeasibility_detection,
            "strict_infeasibility": self.strict_infeasibility,
            "absolute_primal_tolerance": self.absolute_primal_tolerance,
            "relative_primal_tolerance": self.relative_primal_tolerance,
            "absolute_dual_tolerance": self.absolute_dual_tolerance,
            "relative_dual_tolerance": self.relative_dual_tolerance,
            "absolute_gap_tolerance": self.absolute_gap_tolerance,
            "relative_gap_tolerance": self.relative_gap_tolerance,
            "iteration_limit": self.iteration_limit,
            "time_limit": self.time_limit,
            "num_cpu_threads": self.num_cpu_threads,
            "num_gpus": self.num_gpus,
            "barrier_folding": self.barrier_folding,
            "barrier_ordering": self.barrier_ordering,
            "barrier_augmented": self.barrier_augmented,
            "eliminate_dense_columns": self.eliminate_dense_columns,
            "barrier_deterministic": self.barrier_deterministic,
            "per_constraint_residual": self.per_constraint_residual,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CuOptConfig':
        """Create from dictionary."""
        return cls(
            method=SolverMethod(d.get("method", 0)),
            pdlp_solver_mode=PDLPSolverMode(d.get("pdlp_solver_mode", "Stable3")),
            presolve=d.get("presolve", False),
            dualize=d.get("dualize", -1),
            crossover=d.get("crossover", False),
            save_best_primal=d.get("save_best_primal", False),
            first_primal_feasible=d.get("first_primal_feasible", False),
            infeasibility_detection=d.get("infeasibility_detection", False),
            strict_infeasibility=d.get("strict_infeasibility", False),
            absolute_primal_tolerance=d.get("absolute_primal_tolerance", 1e-4),
            relative_primal_tolerance=d.get("relative_primal_tolerance", 1e-4),
            absolute_dual_tolerance=d.get("absolute_dual_tolerance", 1e-4),
            relative_dual_tolerance=d.get("relative_dual_tolerance", 1e-4),
            absolute_gap_tolerance=d.get("absolute_gap_tolerance", 1e-4),
            relative_gap_tolerance=d.get("relative_gap_tolerance", 1e-4),
            iteration_limit=d.get("iteration_limit", 1_000_000),
            time_limit=d.get("time_limit", 300.0),
            num_cpu_threads=d.get("num_cpu_threads", 4),
            num_gpus=d.get("num_gpus", 1),
            barrier_folding=d.get("barrier_folding", -1),
            barrier_ordering=d.get("barrier_ordering", -1),
            barrier_augmented=d.get("barrier_augmented", -1),
            eliminate_dense_columns=d.get("eliminate_dense_columns", True),
            barrier_deterministic=d.get("barrier_deterministic", False),
            per_constraint_residual=d.get("per_constraint_residual", False),
        )

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CuOptConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def baseline(cls) -> 'CuOptConfig':
        """Return cuOpt default configuration."""
        return cls()

    def copy(self) -> 'CuOptConfig':
        """Create a deep copy."""
        return CuOptConfig.from_dict(self.to_dict())


def mutate_config(
    config: CuOptConfig,
    mutation_rate: float = 0.2,
    rng: Optional[random.Random] = None,
) -> CuOptConfig:
    """
    Mutate a cuOpt configuration.

    Args:
        config: Parent configuration
        mutation_rate: Probability of mutating each parameter
        rng: Random number generator

    Returns:
        Mutated configuration copy
    """
    if rng is None:
        rng = random.Random()

    new_config = config.copy()

    for param_name, domain in CUOPT_PARAMETER_DOMAINS.items():
        if rng.random() > mutation_rate:
            continue

        current = getattr(new_config, param_name)
        new_value = _mutate_parameter(current, domain, rng)
        setattr(new_config, param_name, new_value)

    return new_config


def _mutate_parameter(current, domain: Dict, rng: random.Random):
    """Mutate a single parameter according to its domain."""
    param_type = domain["type"]

    if param_type == "bool":
        return not current

    elif param_type == "enum":
        enum_class = domain["enum_class"]
        values = list(enum_class)
        return rng.choice(values)

    elif param_type == "int":
        min_val, max_val = domain["min"], domain["max"]
        sigma = (max_val - min_val) * 0.1
        new_val = current + rng.gauss(0, sigma)
        return int(max(min_val, min(max_val, round(new_val))))

    elif param_type == "float":
        min_val, max_val = domain["min"], domain["max"]
        sigma = (max_val - min_val) * 0.1
        new_val = current + rng.gauss(0, sigma)
        return max(min_val, min(max_val, new_val))

    elif param_type == "log_float":
        min_val, max_val = domain["min"], domain["max"]
        log_current = math.log10(current) if current > 0 else math.log10(min_val)
        log_min, log_max = math.log10(min_val), math.log10(max_val)
        sigma = (log_max - log_min) * 0.15
        log_new = log_current + rng.gauss(0, sigma)
        log_new = max(log_min, min(log_max, log_new))
        return 10 ** log_new

    return current


def crossover_configs(
    config1: CuOptConfig,
    config2: CuOptConfig,
    rng: Optional[random.Random] = None,
) -> CuOptConfig:
    """
    Create child configuration from two parents using uniform crossover.

    Args:
        config1, config2: Parent configurations
        rng: Random number generator

    Returns:
        Child configuration
    """
    if rng is None:
        rng = random.Random()

    d1 = config1.to_dict()
    d2 = config2.to_dict()
    child_dict = {}

    for key in d1:
        child_dict[key] = d1[key] if rng.random() < 0.5 else d2[key]

    return CuOptConfig.from_dict(child_dict)


def random_config(rng: Optional[random.Random] = None) -> CuOptConfig:
    """
    Generate a completely random configuration.

    Args:
        rng: Random number generator

    Returns:
        Random configuration
    """
    if rng is None:
        rng = random.Random()

    config = CuOptConfig()

    for param_name, domain in CUOPT_PARAMETER_DOMAINS.items():
        value = _random_parameter(domain, rng)
        setattr(config, param_name, value)

    return config


def _random_parameter(domain: Dict, rng: random.Random):
    """Generate a random value for a parameter."""
    param_type = domain["type"]

    if param_type == "bool":
        return rng.choice([True, False])

    elif param_type == "enum":
        enum_class = domain["enum_class"]
        return rng.choice(list(enum_class))

    elif param_type == "int":
        return rng.randint(domain["min"], domain["max"])

    elif param_type == "float":
        return rng.uniform(domain["min"], domain["max"])

    elif param_type == "log_float":
        log_min = math.log10(domain["min"])
        log_max = math.log10(domain["max"])
        return 10 ** rng.uniform(log_min, log_max)

    return domain.get("default")


# Feature-conditioned configuration selection
@dataclass
class ConditionalConfig:
    """
    A configuration that selects parameters based on LP features.

    This is the full "genome" - a mapping from LP characteristics to
    cuOpt configurations.
    """
    # Default configuration
    default: CuOptConfig = field(default_factory=CuOptConfig.baseline)

    # Configurations for specific problem types
    small_dense: Optional[CuOptConfig] = None      # < 1000 vars, density > 0.1
    small_sparse: Optional[CuOptConfig] = None     # < 1000 vars, density <= 0.1
    medium_dense: Optional[CuOptConfig] = None     # 1000-10000 vars, density > 0.05
    medium_sparse: Optional[CuOptConfig] = None    # 1000-10000 vars, density <= 0.05
    large_dense: Optional[CuOptConfig] = None      # > 10000 vars, density > 0.01
    large_sparse: Optional[CuOptConfig] = None     # > 10000 vars, density <= 0.01

    # Numerical difficulty adjustments
    hard_numerical: Optional[CuOptConfig] = None   # High coefficient range

    def get_config(self, features: Dict[str, Any]) -> CuOptConfig:
        """
        Select configuration based on LP features.

        Args:
            features: LP feature dictionary with keys:
                - n_vars: number of variables
                - n_cons: number of constraints
                - density: matrix density
                - coef_range_log10: log10 coefficient range

        Returns:
            Appropriate CuOptConfig for this LP
        """
        n_vars = features.get("n_vars", 0)
        density = features.get("density", 0.01)
        coef_range = features.get("coef_range_log10", 0)

        # Check numerical difficulty first
        if coef_range > 5 and self.hard_numerical:
            return self.hard_numerical

        # Size-density based selection
        if n_vars < 1000:
            if density > 0.1:
                return self.small_dense or self.default
            else:
                return self.small_sparse or self.default
        elif n_vars < 10000:
            if density > 0.05:
                return self.medium_dense or self.default
            else:
                return self.medium_sparse or self.default
        else:
            if density > 0.01:
                return self.large_dense or self.default
            else:
                return self.large_sparse or self.default

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = {"default": self.default.to_dict()}
        for name in ["small_dense", "small_sparse", "medium_dense", "medium_sparse",
                     "large_dense", "large_sparse", "hard_numerical"]:
            config = getattr(self, name)
            if config is not None:
                d[name] = config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ConditionalConfig':
        """Deserialize from dictionary."""
        cc = cls(default=CuOptConfig.from_dict(d["default"]))
        for name in ["small_dense", "small_sparse", "medium_dense", "medium_sparse",
                     "large_dense", "large_sparse", "hard_numerical"]:
            if name in d:
                setattr(cc, name, CuOptConfig.from_dict(d[name]))
        return cc

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ConditionalConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
