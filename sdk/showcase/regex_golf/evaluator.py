"""
Regex Golf Evaluator
====================

Evaluates regex solutions for correctness and fitness.

Fitness formula:
  - Base: 1000 - len(regex)  (shorter is better)
  - Penalty: -100 per incorrect match/reject
  - Invalid regex: fitness = 0 (triggers Debugger)
"""

import re
from typing import NamedTuple


class EvalResult(NamedTuple):
    """Result of evaluating a regex solution."""
    valid: bool           # Did regex compile?
    correct: bool         # Did it match all good, reject all bad?
    fitness: float        # Fitness score
    regex_length: int     # Length of regex
    good_matched: int     # How many good strings matched
    good_total: int       # Total good strings
    bad_rejected: int     # How many bad strings rejected
    bad_total: int        # Total bad strings
    error: str | None     # Error message if invalid


def evaluate_regex(pattern: str, good: list[str], bad: list[str]) -> EvalResult:
    """
    Evaluate a regex pattern against good/bad string sets.

    Args:
        pattern: The regex pattern to evaluate
        good: Strings that should match
        bad: Strings that should NOT match

    Returns:
        EvalResult with fitness and diagnostic info
    """
    # Try to compile the regex
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        # Invalid regex - this is where Debugger Agent shines
        return EvalResult(
            valid=False,
            correct=False,
            fitness=0.0,
            regex_length=len(pattern),
            good_matched=0,
            good_total=len(good),
            bad_rejected=0,
            bad_total=len(bad),
            error=f"re.error: {e}",
        )
    except Exception as e:
        return EvalResult(
            valid=False,
            correct=False,
            fitness=0.0,
            regex_length=len(pattern),
            good_matched=0,
            good_total=len(good),
            bad_rejected=0,
            bad_total=len(bad),
            error=f"{type(e).__name__}: {e}",
        )

    # Count matches
    good_matched = sum(1 for s in good if compiled.search(s))
    bad_rejected = sum(1 for s in bad if not compiled.search(s))

    # Calculate fitness
    # Start with length bonus (shorter = better)
    base_fitness = 1000 - len(pattern)

    # Penalty for incorrect behavior
    good_missed = len(good) - good_matched
    bad_matched = len(bad) - bad_rejected
    penalty = (good_missed + bad_matched) * 100

    fitness = max(0, base_fitness - penalty)

    correct = (good_matched == len(good)) and (bad_rejected == len(bad))

    return EvalResult(
        valid=True,
        correct=correct,
        fitness=fitness,
        regex_length=len(pattern),
        good_matched=good_matched,
        good_total=len(good),
        bad_rejected=bad_rejected,
        bad_total=len(bad),
        error=None,
    )


def evaluate_solution_file(filepath: str, good: list[str], bad: list[str]) -> EvalResult:
    """
    Evaluate a solution file containing a regex pattern.

    The file should define a variable `PATTERN` with the regex.
    """
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("solution", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "PATTERN"):
            return EvalResult(
                valid=False,
                correct=False,
                fitness=0.0,
                regex_length=0,
                good_matched=0,
                good_total=len(good),
                bad_rejected=0,
                bad_total=len(bad),
                error="Solution file must define PATTERN variable",
            )

        return evaluate_regex(module.PATTERN, good, bad)

    except Exception as e:
        return EvalResult(
            valid=False,
            correct=False,
            fitness=0.0,
            regex_length=0,
            good_matched=0,
            good_total=len(good),
            bad_rejected=0,
            bad_total=len(bad),
            error=f"{type(e).__name__}: {e}",
        )


if __name__ == "__main__":
    # Quick test
    from problem import DEFAULT_PROBLEM

    print(f"Testing problem: {DEFAULT_PROBLEM['name']}")
    print(f"Good strings: {DEFAULT_PROBLEM['good']}")
    print(f"Bad strings: {DEFAULT_PROBLEM['bad']}")
    print()

    # Test the known optimal
    result = evaluate_regex(
        DEFAULT_PROBLEM["known_optimal"],
        DEFAULT_PROBLEM["good"],
        DEFAULT_PROBLEM["bad"],
    )

    print(f"Known optimal: {DEFAULT_PROBLEM['known_optimal']}")
    print(f"  Valid: {result.valid}")
    print(f"  Correct: {result.correct}")
    print(f"  Fitness: {result.fitness}")
    print(f"  Length: {result.regex_length}")
    print()

    # Test some bad patterns
    bad_patterns = [
        r"[",           # Invalid syntax
        r"Star",        # Too broad
        r".*",          # Matches everything
        r"^$",          # Matches nothing
    ]

    for pattern in bad_patterns:
        result = evaluate_regex(
            pattern,
            DEFAULT_PROBLEM["good"],
            DEFAULT_PROBLEM["bad"],
        )
        status = "INVALID" if not result.valid else ("CORRECT" if result.correct else "WRONG")
        print(f"Pattern: {pattern!r:20} -> {status}, fitness={result.fitness:.0f}")
        if result.error:
            print(f"  Error: {result.error}")
