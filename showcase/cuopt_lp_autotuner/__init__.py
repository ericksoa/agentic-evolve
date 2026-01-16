"""
cuOpt LP AutoTuner - Evolutionary optimization of NVIDIA cuOpt LP solver configurations.

This package evolves robust configurations for the cuOpt GPU-accelerated LP solver
to maximize throughput and minimize latency, without overfitting.

NEW Architecture (cuopt_config, cuopt_solver, evolution):
- cuopt_config: Real cuOpt parameter genome (method, tolerances, PDLP mode, etc.)
- cuopt_solver: Direct cuOpt API integration with scipy HiGHS fallback
- mps_loader: Load LP problems from MPS/LP files
- evolution: Genetic algorithm with train/val/test splits for anti-overfitting

Legacy Architecture (policy, cuopt_runner, evolve):
- feature_extract: Extract structural features from LP instances
- transforms: Safe, solution-preserving formulation transforms
- policy: JSON policy representation
- cuopt_runner: Old solver adapter
- lp_generator: Synthetic LP corpus generation
- benchmark: Multi-objective fitness evaluation
- adversarial: Adversarial instance generation

Quick Start (NEW):
    # Run real cuOpt parameter evolution
    python -m showcase.cuopt_lp_autotuner.evolution --synthetic 100 --generations 30 --verbose

    # With real LP benchmarks
    python -m showcase.cuopt_lp_autotuner.evolution --lp-dir ./benchmarks --generations 50

Quick Start (Legacy):
    # Evaluate baseline policy
    python -m showcase.cuopt_lp_autotuner.benchmark --policy baseline

    # Run policy evolution
    python -m showcase.cuopt_lp_autotuner.evolve --generations 20
"""

__version__ = "0.2.0"

# New cuOpt-native components
try:
    from .cuopt_config import (
        CuOptConfig,
        SolverMethod,
        PDLPSolverMode,
        ConditionalConfig,
        mutate_config,
        crossover_configs,
        random_config,
    )
    from .cuopt_solver import (
        CuOptSolver,
        SolveResult as CuOptSolveResult,
        SolveStatus as CuOptSolveStatus,
        solve_batch,
        is_cuopt_available,
    )
    from .mps_loader import (
        LPData,
        load_mps,
        load_problem,
        get_features as get_lp_features,
    )
    from .evolution import (
        CuOptEvolution,
        EvolutionConfig as CuOptEvolutionConfig,
        LPCorpus,
        FitnessResult as CuOptFitnessResult,
    )
    _NEW_COMPONENTS_AVAILABLE = True
except ImportError as e:
    _NEW_COMPONENTS_AVAILABLE = False
    import logging
    logging.warning(f"New cuOpt components not fully available: {e}")

# Legacy components
try:
    from .transforms import LPInstance, TransformPipeline, TransformType
    from .feature_extract import LPFeatures, extract_features, extract_features_from_lp
    from .policy import Policy, load_policy, mutate_policy, crossover_policies
    from .cuopt_runner import LPSolver, SolveResult, SolveStatus, solve_with_policy
    from .lp_generator import LPGenerator, GeneratorConfig, LPType, create_corpus, lpinstance_to_lpdata
    from .benchmark import Benchmark, BenchmarkMetrics, FitnessResult, evaluate_policy
    from .evolve import Evolution, EvolutionConfig, run_evolution
    from .adversarial import AdversarialGenerator, find_adversarial_instances
    _LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    _LEGACY_COMPONENTS_AVAILABLE = False
    import logging
    logging.warning(f"Legacy components not fully available: {e}")

__all__ = [
    # Version
    "__version__",

    # New cuOpt-native (preferred)
    "CuOptConfig",
    "SolverMethod",
    "PDLPSolverMode",
    "ConditionalConfig",
    "mutate_config",
    "crossover_configs",
    "random_config",
    "CuOptSolver",
    "CuOptSolveResult",
    "CuOptSolveStatus",
    "solve_batch",
    "is_cuopt_available",
    "LPData",
    "load_mps",
    "load_problem",
    "get_lp_features",
    "CuOptEvolution",
    "CuOptEvolutionConfig",
    "LPCorpus",
    "CuOptFitnessResult",

    # Legacy (for backwards compatibility)
    "LPInstance",
    "LPFeatures",
    "Policy",
    "SolveResult",
    "SolveStatus",
    "TransformPipeline",
    "TransformType",
    "extract_features",
    "extract_features_from_lp",
    "load_policy",
    "mutate_policy",
    "crossover_policies",
    "LPSolver",
    "solve_with_policy",
    "LPGenerator",
    "GeneratorConfig",
    "LPType",
    "create_corpus",
    "lpinstance_to_lpdata",
    "Benchmark",
    "BenchmarkMetrics",
    "FitnessResult",
    "evaluate_policy",
    "Evolution",
    "EvolutionConfig",
    "run_evolution",
    "AdversarialGenerator",
    "find_adversarial_instances",
]
