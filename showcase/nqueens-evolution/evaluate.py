#!/usr/bin/env python3
"""
Evaluator for N-Queens solvers.

Tests the solver across multiple board sizes and measures:
- Correctness (valid solutions)
- Speed (solutions per second)
- Generalization (works on all sizes)

Fitness = weighted score based on speed and correctness.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load_solver(solution_path: str):
    """Dynamically load a solver module."""
    path = Path(solution_path)
    if not path.exists():
        raise FileNotFoundError(f"Solution not found: {solution_path}")

    spec = importlib.util.spec_from_file_location("solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "solve_nqueens"):
        raise AttributeError("Solution must define solve_nqueens(n: int) -> list[int]")

    return module


def verify_solution(n: int, solution: list[int] | None) -> tuple[bool, str]:
    """
    Verify that a solution is valid.

    Returns:
        (is_valid, error_message)
    """
    if solution is None:
        return False, "No solution returned"

    if not isinstance(solution, (list, tuple)):
        return False, f"Expected list, got {type(solution)}"

    if len(solution) != n:
        return False, f"Expected {n} queens, got {len(solution)}"

    for row, col in enumerate(solution):
        if not isinstance(col, int):
            return False, f"Position must be int, got {type(col)}"
        if col < 0 or col >= n:
            return False, f"Column {col} out of range [0, {n})"

        # Check conflicts with previous queens
        for prev_row in range(row):
            prev_col = solution[prev_row]
            if prev_col == col:
                return False, f"Column conflict at rows {prev_row} and {row}"
            if abs(prev_col - col) == abs(prev_row - row):
                return False, f"Diagonal conflict at rows {prev_row} and {row}"

    return True, ""


def evaluate_solver(module, sizes: list[int], timeout_per_size: float = 30.0) -> dict:
    """
    Evaluate a solver across multiple board sizes.

    Args:
        module: Loaded solver module
        sizes: List of board sizes to test
        timeout_per_size: Max time per size in seconds

    Returns:
        Evaluation results dict
    """
    results = {}
    total_time = 0
    valid_count = 0
    total_count = len(sizes)

    for n in sizes:
        start = time.perf_counter()

        try:
            solution = module.solve_nqueens(n)
            elapsed = time.perf_counter() - start

            if elapsed > timeout_per_size:
                results[f"n{n}"] = {
                    "valid": False,
                    "error": f"Timeout ({elapsed:.2f}s > {timeout_per_size}s)",
                    "time_ms": elapsed * 1000,
                }
                continue

            is_valid, error = verify_solution(n, solution)

            results[f"n{n}"] = {
                "valid": is_valid,
                "time_ms": elapsed * 1000,
                "solution": solution if is_valid else None,
                "error": error if not is_valid else None,
            }

            if is_valid:
                valid_count += 1
                total_time += elapsed

        except Exception as e:
            elapsed = time.perf_counter() - start
            results[f"n{n}"] = {
                "valid": False,
                "error": str(e)[:100],
                "time_ms": elapsed * 1000,
            }

    # Calculate fitness
    # Base: correctness ratio
    correctness = valid_count / total_count if total_count > 0 else 0

    # Speed bonus: solutions per second (only for valid solutions)
    speed = valid_count / total_time if total_time > 0 else 0

    # Combined fitness:
    # - Must be correct to get meaningful fitness
    # - Speed is the differentiator
    # - Scale so typical values are in reasonable range
    if correctness < 1.0:
        # Penalty for incorrect solutions
        fitness = correctness * 10  # Max 10 if some are wrong
    else:
        # All correct: fitness based on speed
        # Baseline ~4 solves/sec -> fitness ~40
        # Fast solver ~100 solves/sec -> fitness ~1000
        fitness = speed * 10

    return {
        "valid": valid_count == total_count,
        "fitness": round(fitness, 4),
        "correctness": correctness,
        "speed_solutions_per_sec": round(speed, 4),
        "total_time_ms": round(total_time * 1000, 2),
        "valid_count": valid_count,
        "total_count": total_count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate N-Queens solver")
    parser.add_argument("solution", help="Path to solution file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--sizes",
        type=str,
        default="8,12,16,20",
        help="Comma-separated board sizes (default: 8,12,16,20)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per size in seconds (default: 30)",
    )

    args = parser.parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    try:
        module = load_solver(args.solution)
        result = evaluate_solver(module, sizes, args.timeout)

        if args.json:
            print(json.dumps(result))
        else:
            print(f"Solution: {args.solution}")
            print(f"Valid: {result['valid']}")
            print(f"Fitness: {result['fitness']:.4f}")
            print(f"Speed: {result['speed_solutions_per_sec']:.2f} solutions/sec")
            print(f"Correctness: {result['valid_count']}/{result['total_count']}")
            print(f"\nResults by size:")
            for key, val in result["results"].items():
                status = "OK" if val["valid"] else f"FAIL: {val.get('error', 'unknown')}"
                print(f"  {key}: {val['time_ms']:.2f}ms - {status}")

    except Exception as e:
        if args.json:
            print(json.dumps({"valid": False, "fitness": 0, "error": str(e)}))
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
