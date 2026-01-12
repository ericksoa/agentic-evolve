"""Variance Gates - Statistical validation through repeated evaluation.

Evaluates candidates N times and computes statistics to detect:
- Stochastic exploits that only work sometimes
- Timing-dependent behavior
- Non-deterministic fitness gaming

A solution must have consistent fitness across multiple evaluations to pass.
"""

import statistics
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class VarianceStats:
    """Statistics from N evaluations."""

    values: list[float]
    mean: float
    std: float
    cv: float  # Coefficient of variation (std/mean)
    min_val: float
    max_val: float
    n_evals: int
    passed: bool  # True if CV <= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "values": self.values,
            "mean": self.mean,
            "std": self.std,
            "cv": self.cv,
            "min": self.min_val,
            "max": self.max_val,
            "n_evals": self.n_evals,
            "passed": self.passed,
        }


class VarianceGate:
    """
    Variance Gate for statistical validation.

    Re-evaluates candidates multiple times and requires consistent results.
    """

    def __init__(
        self,
        n_evaluations: int = 3,
        variance_threshold: float = 0.05,
        require_gate: bool = False,
    ):
        """
        Initialize the variance gate.

        Args:
            n_evaluations: Number of evaluation runs
            variance_threshold: Maximum acceptable coefficient of variation
            require_gate: If True, reject candidates that fail variance check
        """
        self.n_evaluations = n_evaluations
        self.variance_threshold = variance_threshold
        self.require_gate = require_gate

    def evaluate_with_variance(
        self,
        evaluate_fn: Callable[[], float],
    ) -> VarianceStats:
        """
        Run evaluation N times and compute statistics.

        Args:
            evaluate_fn: Function that returns fitness score

        Returns:
            VarianceStats with computed statistics
        """
        values = []
        for _ in range(self.n_evaluations):
            try:
                fitness = evaluate_fn()
                values.append(fitness)
            except Exception:
                # Failed evaluation counts as 0
                values.append(0.0)

        if not values:
            return VarianceStats(
                values=[],
                mean=0.0,
                std=0.0,
                cv=float("inf"),
                min_val=0.0,
                max_val=0.0,
                n_evals=0,
                passed=False,
            )

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Coefficient of variation (handle zero mean)
        if mean > 0:
            cv = std / mean
        elif std == 0:
            cv = 0.0  # All values are 0 - consistent, but not useful
        else:
            cv = float("inf")  # Non-zero std with zero mean - very inconsistent

        passed = cv <= self.variance_threshold

        return VarianceStats(
            values=values,
            mean=mean,
            std=std,
            cv=cv,
            min_val=min(values),
            max_val=max(values),
            n_evals=len(values),
            passed=passed,
        )

    async def evaluate_with_variance_async(
        self,
        evaluate_fn: Callable[[], Any],
    ) -> VarianceStats:
        """
        Async version of evaluate_with_variance.

        Args:
            evaluate_fn: Async function that returns fitness score

        Returns:
            VarianceStats with computed statistics
        """
        import asyncio

        values = []
        for _ in range(self.n_evaluations):
            try:
                result = evaluate_fn()
                if asyncio.iscoroutine(result):
                    fitness = await result
                else:
                    fitness = result
                values.append(float(fitness))
            except Exception:
                values.append(0.0)

        if not values:
            return VarianceStats(
                values=[],
                mean=0.0,
                std=0.0,
                cv=float("inf"),
                min_val=0.0,
                max_val=0.0,
                n_evals=0,
                passed=False,
            )

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        if mean > 0:
            cv = std / mean
        elif std == 0:
            cv = 0.0
        else:
            cv = float("inf")

        passed = cv <= self.variance_threshold

        return VarianceStats(
            values=values,
            mean=mean,
            std=std,
            cv=cv,
            min_val=min(values),
            max_val=max(values),
            n_evals=len(values),
            passed=passed,
        )

    def check_consistency(self, values: list[float]) -> VarianceStats:
        """
        Check consistency of pre-computed values.

        Args:
            values: List of fitness values from multiple evaluations

        Returns:
            VarianceStats with analysis
        """
        if not values:
            return VarianceStats(
                values=[],
                mean=0.0,
                std=0.0,
                cv=float("inf"),
                min_val=0.0,
                max_val=0.0,
                n_evals=0,
                passed=False,
            )

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        if mean > 0:
            cv = std / mean
        elif std == 0:
            cv = 0.0
        else:
            cv = float("inf")

        passed = cv <= self.variance_threshold

        return VarianceStats(
            values=values,
            mean=mean,
            std=std,
            cv=cv,
            min_val=min(values),
            max_val=max(values),
            n_evals=len(values),
            passed=passed,
        )

    def get_flags(self, stats: VarianceStats) -> list[str]:
        """
        Get warning flags based on variance analysis.

        Args:
            stats: Variance statistics

        Returns:
            List of warning flags
        """
        flags = []

        if not stats.passed:
            flags.append(f"High variance: CV={stats.cv:.3f} > {self.variance_threshold}")

        if stats.n_evals < self.n_evaluations:
            flags.append(f"Incomplete evaluation: {stats.n_evals}/{self.n_evaluations} runs")

        if stats.min_val == 0 and stats.max_val > 0:
            flags.append("Inconsistent: some runs returned 0 fitness")

        range_ratio = (stats.max_val - stats.min_val) / stats.mean if stats.mean > 0 else 0
        if range_ratio > 0.5:
            flags.append(f"Wide range: {stats.min_val:.2f} to {stats.max_val:.2f}")

        return flags
