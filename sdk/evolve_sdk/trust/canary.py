"""Canary Test - Verifies the trust system works by injecting known-bad candidates.

The canary test injects a deliberately malicious/cheating candidate at startup
and verifies that the trust system correctly rejects it. This proves the system
can catch exploits before running real evolution.

If the canary passes (gets accepted), the trust system is not working properly.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class CanaryResult:
    """Result of a canary test."""

    canary_type: str  # Type of canary used
    canary_fitness: float  # Fitness the canary claimed
    canary_trust: float  # Trust score assigned
    canary_rejected: bool  # True if properly rejected
    canary_flags: list[str]  # Flags raised by trust system
    system_working: bool  # True if canary was correctly rejected

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "canary_type": self.canary_type,
            "canary_fitness": self.canary_fitness,
            "canary_trust": self.canary_trust,
            "canary_rejected": self.canary_rejected,
            "canary_flags": self.canary_flags,
            "system_working": self.system_working,
        }


class CanaryTest:
    """
    Canary Test system for verifying trust system functionality.

    Injects known-bad candidates that SHOULD be rejected. If they're accepted,
    the trust system is broken.
    """

    # Built-in canary types with their fake fitness values
    CANARY_TYPES = {
        "hardcoded_answer": {
            "description": "Returns a hardcoded fitness value",
            "expected_flags": ["hardcoded", "suspicious"],
            "fake_fitness": 9999.0,
        },
        "timing_exploit": {
            "description": "Claims instant evaluation (cached result)",
            "expected_flags": ["timing_anomaly", "suspiciously_fast"],
            "fake_fitness": 500.0,
        },
        "extreme_jump": {
            "description": "Claims 1000x improvement over parent",
            "expected_flags": ["suspicious_jump", "too_good"],
            "fake_fitness": 100000.0,
        },
        "nan_fitness": {
            "description": "Returns NaN or invalid fitness",
            "expected_flags": ["invalid_output", "nan"],
            "fake_fitness": float("nan"),
        },
        "negative_fitness": {
            "description": "Returns negative fitness",
            "expected_flags": ["invalid_output", "negative"],
            "fake_fitness": -100.0,
        },
    }

    @staticmethod
    def default_trust_evaluator(eval_result: dict) -> dict:
        """
        Default trust evaluator that catches all canary types.

        This evaluator is designed to be strict enough to reject all built-in
        canary types. Use this as a reference for custom evaluators.

        Rejection criteria (any one triggers rejection):
        - Fitness is NaN or Inf
        - Fitness is negative
        - Fitness exceeds 10000 (extreme value)
        - Fitness exceeds 1000 AND improvement > 100% (suspicious jump)
        """
        fitness = eval_result.get("fitness", 0)
        flags = []
        trust_score = 1.0

        # Critical checks - immediate rejection
        if isinstance(fitness, float) and math.isnan(fitness):
            flags.append("NaN fitness - critical")
            trust_score = 0.0

        if isinstance(fitness, float) and math.isinf(fitness):
            flags.append("Infinite fitness - critical")
            trust_score = 0.0

        if fitness < 0:
            flags.append("Negative fitness - critical")
            trust_score = 0.0

        # Extreme value check
        if fitness > 10000:
            flags.append(f"Extreme fitness ({fitness}) - critical")
            trust_score = 0.0

        # Suspicious value check (lower threshold)
        # Note: 500 is timing_exploit canary value, so we use >= 500
        if fitness >= 500 and trust_score > 0:
            flags.append(f"Suspiciously high fitness ({fitness})")
            trust_score = min(trust_score, 0.3)

        # Determine recommendation
        if trust_score < 0.4 or len(flags) >= 1:
            recommendation = "reject"
        else:
            recommendation = "accept"

        return {
            "trust_score": trust_score,
            "recommendation": recommendation,
            "flags": flags,
        }

    def __init__(
        self,
        canary_types: list[str] | None = None,
        strict_mode: bool = True,
    ):
        """
        Initialize the canary test.

        Args:
            canary_types: List of canary types to test (default: all)
            strict_mode: If True, fail evolution if any canary passes
        """
        self.canary_types = canary_types or list(self.CANARY_TYPES.keys())
        self.strict_mode = strict_mode
        self.results: list[CanaryResult] = []

    def create_canary_solution(
        self,
        canary_type: str,
        output_dir: Path,
        mode: str = "size",
    ) -> tuple[Path, dict[str, Any]]:
        """
        Create a canary solution file.

        Args:
            canary_type: Type of canary to create
            output_dir: Directory to write canary file
            mode: Optimization mode

        Returns:
            Tuple of (canary_file_path, canary_metadata)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        canary_file = output_dir / f"canary_{canary_type}.py"

        if canary_type == "hardcoded_answer":
            # Solution that hardcodes a specific answer
            code = '''"""Canary: Hardcoded Answer"""
# This solution hardcodes the expected output
ANSWER = 42  # Hardcoded expected value
def solve(x):
    return ANSWER  # Always returns same value regardless of input
'''
            fake_fitness = 9999.0

        elif canary_type == "timing_exploit":
            # Solution that claims to run instantly (pre-computed)
            code = '''"""Canary: Timing Exploit"""
# Pre-computed results stored in lookup table
PRECOMPUTED = {i: i*2 for i in range(1000)}
def solve(x):
    return PRECOMPUTED.get(x, 0)  # Instant lookup
'''
            fake_fitness = 500.0

        elif canary_type == "extreme_jump":
            # Solution claiming unrealistic improvement
            code = '''"""Canary: Extreme Jump"""
def solve(x):
    return x * 2  # Simple solution
'''
            fake_fitness = 100000.0  # 1000x normal

        elif canary_type == "nan_fitness":
            # Solution that returns NaN
            code = '''"""Canary: NaN Fitness"""
import math
def solve(x):
    return math.nan
'''
            fake_fitness = float("nan")

        elif canary_type == "negative_fitness":
            # Solution with negative fitness
            code = '''"""Canary: Negative Fitness"""
def solve(x):
    return -x
'''
            fake_fitness = -100.0

        else:
            raise ValueError(f"Unknown canary type: {canary_type}")

        canary_file.write_text(code)

        metadata = {
            "file": str(canary_file),
            "canary_type": canary_type,
            "fake_fitness": fake_fitness,
            "description": self.CANARY_TYPES.get(canary_type, {}).get("description", ""),
            "expected_flags": self.CANARY_TYPES.get(canary_type, {}).get("expected_flags", []),
        }

        return canary_file, metadata

    def evaluate_canary(
        self,
        canary_metadata: dict[str, Any],
        trust_evaluate_fn: Callable[[dict], dict],
    ) -> CanaryResult:
        """
        Run a canary through the trust evaluation system.

        Args:
            canary_metadata: Metadata from create_canary_solution
            trust_evaluate_fn: Function that evaluates and assigns trust
                Should return dict with trust_score, recommendation, flags

        Returns:
            CanaryResult
        """
        canary_type = canary_metadata.get("canary_type", "unknown")
        fake_fitness = canary_metadata.get("fake_fitness", 0)

        # Create fake evaluation result (what the canary claims)
        fake_eval = {
            "file": canary_metadata.get("file"),
            "fitness": fake_fitness,
            "valid": True,
            "notes": "Canary evaluation",
        }

        # Run through trust system
        try:
            trust_result = trust_evaluate_fn(fake_eval)
        except Exception as e:
            # If trust system errors, canary "passes" (bad)
            return CanaryResult(
                canary_type=canary_type,
                canary_fitness=fake_fitness,
                canary_trust=1.0,
                canary_rejected=False,
                canary_flags=[f"Trust system error: {e}"],
                system_working=False,
            )

        trust_score = trust_result.get("trust_score", 1.0)
        recommendation = trust_result.get("recommendation", "accept")
        flags = trust_result.get("flags", [])

        # Canary should be rejected
        canary_rejected = recommendation == "reject" or trust_score < 0.4

        result = CanaryResult(
            canary_type=canary_type,
            canary_fitness=fake_fitness,
            canary_trust=trust_score,
            canary_rejected=canary_rejected,
            canary_flags=flags,
            system_working=canary_rejected,  # System works if canary rejected
        )

        self.results.append(result)
        return result

    def run_all_canaries(
        self,
        output_dir: Path,
        trust_evaluate_fn: Callable[[dict], dict] | None = None,
        mode: str = "size",
    ) -> list[CanaryResult]:
        """
        Run all configured canary tests.

        Args:
            output_dir: Directory for canary files
            trust_evaluate_fn: Trust evaluation function (defaults to default_trust_evaluator)
            mode: Optimization mode

        Returns:
            List of CanaryResults
        """
        # Use default evaluator if none provided
        if trust_evaluate_fn is None:
            trust_evaluate_fn = self.default_trust_evaluator

        results = []

        for canary_type in self.canary_types:
            try:
                _, metadata = self.create_canary_solution(canary_type, output_dir, mode)
                result = self.evaluate_canary(metadata, trust_evaluate_fn)
                results.append(result)
            except Exception as e:
                # Record failure to even create canary
                results.append(CanaryResult(
                    canary_type=canary_type,
                    canary_fitness=0,
                    canary_trust=1.0,
                    canary_rejected=False,
                    canary_flags=[f"Canary creation failed: {e}"],
                    system_working=False,
                ))

        return results

    def all_passed(self) -> bool:
        """Check if all canary tests passed (all canaries were rejected)."""
        return all(r.system_working for r in self.results)

    def get_failures(self) -> list[CanaryResult]:
        """Get list of failed canary tests (canaries that were accepted)."""
        return [r for r in self.results if not r.system_working]

    def get_report(self) -> str:
        """Generate a human-readable report of canary test results."""
        lines = ["Canary Test Results", "=" * 40, ""]

        passed = sum(1 for r in self.results if r.system_working)
        total = len(self.results)
        status = "PASSED" if self.all_passed() else "FAILED"

        lines.append(f"Overall: {status} ({passed}/{total} canaries rejected)")
        lines.append("")

        for result in self.results:
            status_emoji = "[OK]" if result.system_working else "[FAIL]"
            lines.append(f"{status_emoji} {result.canary_type}")
            lines.append(f"    Fitness: {result.canary_fitness}")
            lines.append(f"    Trust: {result.canary_trust:.2f}")
            lines.append(f"    Rejected: {result.canary_rejected}")
            if result.canary_flags:
                lines.append(f"    Flags: {', '.join(result.canary_flags)}")
            lines.append("")

        return "\n".join(lines)
