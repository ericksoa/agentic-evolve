#!/usr/bin/env python3
"""
Evaluation bridge: evolve-sdk <-> AlgoTune timing_core.

Usage:
    python evaluate_task.py <solution_path> --task <task_name> --json

Output (JSON):
    {"fitness": 3.45, "valid": true, "speedup": 3.45, "error": null, ...}

This script uses AlgoTune's own timing infrastructure for comparable measurements.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure AlgoTune is importable
SHOWCASE_DIR = Path(__file__).resolve().parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
if str(ALGOTUNE_DIR) not in sys.path:
    sys.path.insert(0, str(ALGOTUNE_DIR))


def load_solver_module(solution_path: str):
    """Dynamically load a solver.py file as a module."""
    path = Path(solution_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Solution not found: {path}")

    spec = importlib.util.spec_from_file_location("solver_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_solver_class(module):
    """Extract the Solver class from a loaded module."""
    if not hasattr(module, "Solver"):
        raise AttributeError(
            "Solution must define a 'class Solver' with a 'solve(self, problem)' method"
        )
    return module.Solver


def evaluate_solution(
    solution_path: str,
    task_name: str,
    num_runs: int = 3,
    timeout_seconds: int = 300,
) -> dict:
    """
    Evaluate a solution using AlgoTune's task infrastructure.

    Returns dict with:
        fitness: float  - speedup ratio (higher = better)
        valid: bool     - whether solution passes is_solution() for all inputs
        speedup: float  - same as fitness
        error: str|None - error message if invalid
        details: dict   - per-input timing breakdown
    """
    result = {
        "fitness": 0.0,
        "valid": False,
        "speedup": 0.0,
        "error": None,
        "details": {},
        "evaluation_time_ms": 0.0,
    }

    eval_start = time.time()

    try:
        # Import AlgoTune task infrastructure
        from AlgoTuneTasks.factory import TaskFactory

        # Load the task
        task = TaskFactory(task_name)

        # Load the candidate solver
        solver_module = load_solver_module(solution_path)
        SolverClass = get_solver_class(solver_module)
        solver = SolverClass()

        # Generate test problems using the task's own method.
        # n is the problem SIZE (e.g., matrix dimension), not an index.
        # We use a range of sizes that produce non-trivial timing.
        PROBLEM_SIZES = [10, 25, 50, 100, 200, 300, 500]
        test_inputs = []
        for n in PROBLEM_SIZES:
            for seed in [42, 123]:
                try:
                    problem = task.generate_problem(n=n, random_seed=seed)
                    test_inputs.append(problem)
                except Exception:
                    pass  # Some tasks reject certain sizes
            if len(test_inputs) >= 8:
                break

        if not test_inputs:
            # Fallback: try smaller sizes
            for n in [2, 3, 5, 8]:
                try:
                    problem = task.generate_problem(n=n, random_seed=42)
                    test_inputs.append(problem)
                except Exception:
                    pass

        if not test_inputs:
            result["error"] = f"Could not generate test inputs for task '{task_name}'"
            return result

        # Phase 1: Validate correctness on ALL inputs
        for i, problem in enumerate(test_inputs):
            try:
                candidate_solution = solver.solve(problem)
            except Exception as e:
                result["error"] = f"Solver crashed on input {i}: {e}"
                return result

            try:
                valid = task.is_solution(problem, candidate_solution)
            except Exception as e:
                result["error"] = f"Validation crashed on input {i}: {e}"
                return result

            if not valid:
                result["error"] = f"Solution invalid on input {i}"
                return result

        # Phase 2: Time both reference and candidate solvers
        ref_times = []
        candidate_times = []

        for problem in test_inputs:
            # Time reference solver (multiple runs, take median)
            run_ref_times = []
            for _ in range(num_runs):
                start = time.perf_counter_ns()
                task.solve(problem)
                elapsed = time.perf_counter_ns() - start
                run_ref_times.append(elapsed)
            ref_times.append(sorted(run_ref_times)[len(run_ref_times) // 2])

            # Time candidate solver (multiple runs, take median)
            run_cand_times = []
            for _ in range(num_runs):
                start = time.perf_counter_ns()
                solver.solve(problem)
                elapsed = time.perf_counter_ns() - start
                run_cand_times.append(elapsed)
            candidate_times.append(sorted(run_cand_times)[len(run_cand_times) // 2])

        # Compute per-input speedups
        speedups = []
        per_input = {}
        for i, (ref_t, cand_t) in enumerate(zip(ref_times, candidate_times)):
            if cand_t <= 0:
                cand_t = 1  # Prevent division by zero
            speedup = ref_t / cand_t
            speedups.append(speedup)
            per_input[f"input_{i}"] = {
                "ref_ns": ref_t,
                "candidate_ns": cand_t,
                "speedup": round(speedup, 4),
            }

        # Compute harmonic mean of speedups (matches AlgoTune methodology)
        n = len(speedups)
        harmonic_mean = n / sum(1.0 / s for s in speedups) if speedups else 0.0

        # Variance check: coefficient of variation
        import statistics
        if len(speedups) > 1:
            cv = statistics.stdev(speedups) / statistics.mean(speedups)
        else:
            cv = 0.0

        result["fitness"] = round(harmonic_mean, 4)
        result["valid"] = True
        result["speedup"] = round(harmonic_mean, 4)
        result["details"] = {
            "per_input": per_input,
            "num_inputs": len(test_inputs),
            "num_runs_per_input": num_runs,
            "coefficient_of_variation": round(cv, 4),
            "min_speedup": round(min(speedups), 4),
            "max_speedup": round(max(speedups), 4),
        }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    result["evaluation_time_ms"] = round((time.time() - eval_start) * 1000, 1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a solver solution against an AlgoTune task"
    )
    parser.add_argument("solution", help="Path to solver.py file")
    parser.add_argument("--task", required=True, help="AlgoTune task name")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--runs", type=int, default=3, help="Timing runs per input (default: 3)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout in seconds (default: 300)"
    )
    args = parser.parse_args()

    result = evaluate_solution(
        solution_path=args.solution,
        task_name=args.task,
        num_runs=args.runs,
        timeout_seconds=args.timeout,
    )

    if args.json:
        print(json.dumps(result))
    else:
        status = "PASS" if result["valid"] else "FAIL"
        print(f"[{status}] {args.task}: speedup={result['speedup']}x")
        if result["error"]:
            print(f"  Error: {result['error']}")
        if result["details"]:
            d = result["details"]
            print(
                f"  Inputs: {d.get('num_inputs', '?')}, "
                f"CV: {d.get('coefficient_of_variation', '?')}, "
                f"Range: {d.get('min_speedup', '?')}x - {d.get('max_speedup', '?')}x"
            )


if __name__ == "__main__":
    main()
