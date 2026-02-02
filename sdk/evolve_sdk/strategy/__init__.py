"""
Strategy selection and orchestration.

Provides intelligent problem analysis and strategy recommendation to determine
whether code evolution or hyperparameter evolution is most appropriate.
"""

from .types import (
    DetectedParameter,
    ProblemContext,
    StrategyRecommendation,
    StrategyDecision,
    PROBLEM_TYPES,
    STRATEGY_TYPES,
    COMPLEXITY_LEVELS,
)

__all__ = [
    "DetectedParameter",
    "ProblemContext",
    "StrategyRecommendation",
    "StrategyDecision",
    "PROBLEM_TYPES",
    "STRATEGY_TYPES",
    "COMPLEXITY_LEVELS",
]
