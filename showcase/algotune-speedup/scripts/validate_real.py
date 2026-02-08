#!/usr/bin/env python3
"""
Rigorous validation of evolved solutions using methodology closer to AlgoTune's real eval.

Key differences from evaluate_task.py:
1. Enforces OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc. for stable timing
2. Uses calibrated problem sizes (probes until reference takes ~50-100ms)
3. Uses 10 timing runs per input (AlgoTune uses 10 for eval)
4. Uses 20 inputs across diverse sizes
5. Reports both harmonic mean and geometric mean of speedups

Usage:
    python scripts/validate_real.py                     # Validate all 19 valid tasks
    python scripts/validate_real.py --task svm           # Validate one task
    python scripts/validate_real.py --task svm --verbose  # With per-input details
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import statistics
from pathlib import Path

# Force single-threaded BLAS/LAPACK BEFORE importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
EVOLVED_DIR = SHOWCASE_DIR / "evolved"
RESULTS_DIR = SHOWCASE_DIR / "results"

if str(ALGOTUNE_DIR) not in sys.path:
    sys.path.insert(0, str(ALGOTUNE_DIR))

# Valid tasks from Phase 2 results
VALID_TASKS = [
    "cholesky_factorization",
    "convex_hull",
    "convolve_1d",
    "correlate_1d",
    "eigenvectors_real",
    "fft_convolution",
    "kmeans",
    "linear_system_solver",
    "lu_factorization",
    "matrix_multiplication",
    "minimum_spanning_tree",
    "ode_brusselator",
    "ode_lotkavolterra",
    "pagerank",
    "pca",
    "qr_factorization",
    "svd",
    "svm",
    # dijkstra_from_indices had speedup < 1.0 but was valid
    "dijkstra_from_indices",
]

# Number of timing runs per input (AlgoTune uses 10 for eval)
NUM_RUNS = 10
# Target reference time range in ms — we calibrate problem sizes to this
TARGET_REF_MS_MIN = 10
TARGET_REF_MS_MAX = 500
# Number of test inputs
NUM_INPUTS = 16


def load_solver(solution_path: str):
    """Load a solver module from file path."""
    path = Path(solution_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Solution not found: {path}")
    spec = importlib.util.spec_from_file_location("solver_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver()


def calibrate_sizes(task, target_ms_min=10, target_ms_max=500):
    """
    Find problem sizes where the reference solver takes between target_ms_min and target_ms_max.
    Returns a list of (n, seed, problem) tuples.
    """
    calibrated = []
    # Probe sizes exponentially
    candidate_sizes = [5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
    seeds = [42, 123, 7, 256, 999, 31, 77, 444]

    for n in candidate_sizes:
        for seed in seeds:
            try:
                problem = task.generate_problem(n=n, random_seed=seed)
            except Exception:
                continue

            # Quick timing of reference solver (single run)
            try:
                start = time.perf_counter_ns()
                task.solve(problem)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            except Exception:
                continue

            if target_ms_min <= elapsed_ms <= target_ms_max:
                calibrated.append((n, seed, problem))

            if elapsed_ms > target_ms_max * 2:
                break  # No point trying larger sizes with this seed

        if len(calibrated) >= NUM_INPUTS:
            break

    return calibrated[:NUM_INPUTS]


def generate_diverse_inputs(task, num_inputs=NUM_INPUTS):
    """
    Generate diverse test inputs. First tries calibration, falls back to range of sizes.
    """
    # Try calibration first
    calibrated = calibrate_sizes(task)
    if len(calibrated) >= 8:
        return [(n, seed, prob) for n, seed, prob in calibrated]

    # Fallback: use a range of sizes
    inputs = []
    sizes = [5, 10, 25, 50, 100, 200, 300, 500, 750, 1000]
    seeds = [42, 123, 7, 256]

    for n in sizes:
        for seed in seeds:
            try:
                problem = task.generate_problem(n=n, random_seed=seed)
                inputs.append((n, seed, problem))
            except Exception:
                continue
            if len(inputs) >= num_inputs:
                return inputs

    return inputs


def time_function(func, problem, num_runs=NUM_RUNS):
    """
    Time a function with warmup and multiple runs.
    Returns median time in nanoseconds.
    """
    # Warmup
    try:
        func(problem)
    except Exception:
        return None

    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        try:
            result = func(problem)
        except Exception:
            return None
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    # Return minimum (AlgoTune uses min_time as the benchmark)
    return min(times)


def validate_task(task_name, verbose=False):
    """
    Validate a single task's evolved solution with rigorous methodology.
    Returns dict with validation results.
    """
    from AlgoTuneTasks.factory import TaskFactory

    result = {
        "task": task_name,
        "valid": False,
        "speedup": 0.0,
        "error": None,
        "per_input": [],
        "num_inputs": 0,
        "num_runs": NUM_RUNS,
    }

    # Load task
    try:
        task = TaskFactory(task_name)
    except Exception as e:
        result["error"] = f"TaskFactory failed: {e}"
        return result

    # Load evolved solver
    solver_path = EVOLVED_DIR / task_name / "solver.py"
    if not solver_path.exists():
        result["error"] = f"No evolved solver at {solver_path}"
        return result

    try:
        solver = load_solver(str(solver_path))
    except Exception as e:
        result["error"] = f"Solver load failed: {e}"
        return result

    # Generate diverse test inputs
    inputs = generate_diverse_inputs(task)
    if not inputs:
        result["error"] = "Could not generate any test inputs"
        return result

    result["num_inputs"] = len(inputs)

    if verbose:
        print(f"  Generated {len(inputs)} test inputs")

    # Phase 1: Validate correctness on ALL inputs
    for i, (n, seed, problem) in enumerate(inputs):
        try:
            candidate_solution = solver.solve(problem)
        except Exception as e:
            result["error"] = f"Solver crashed on input {i} (n={n}, seed={seed}): {e}"
            return result

        try:
            valid = task.is_solution(problem, candidate_solution)
        except Exception as e:
            result["error"] = f"Validation crashed on input {i} (n={n}, seed={seed}): {e}"
            return result

        if not valid:
            result["error"] = f"Solution invalid on input {i} (n={n}, seed={seed})"
            return result

    if verbose:
        print(f"  Correctness: all {len(inputs)} inputs passed")

    # Phase 2: Rigorous timing
    speedups = []
    per_input = []

    for i, (n, seed, problem) in enumerate(inputs):
        ref_time = time_function(task.solve, problem, NUM_RUNS)
        if ref_time is None:
            if verbose:
                print(f"  Input {i} (n={n}): reference solver failed, skipping")
            continue

        cand_time = time_function(solver.solve, problem, NUM_RUNS)
        if cand_time is None:
            if verbose:
                print(f"  Input {i} (n={n}): candidate solver failed")
            result["error"] = f"Candidate failed timing on input {i} (n={n})"
            return result

        if cand_time <= 0:
            cand_time = 1  # Prevent division by zero

        speedup = ref_time / cand_time
        speedups.append(speedup)

        entry = {
            "n": n,
            "seed": seed,
            "ref_ns": ref_time,
            "cand_ns": cand_time,
            "ref_ms": round(ref_time / 1e6, 3),
            "cand_ms": round(cand_time / 1e6, 3),
            "speedup": round(speedup, 4),
        }
        per_input.append(entry)

        if verbose:
            print(
                f"  Input {i} (n={n:5d}): ref={entry['ref_ms']:10.3f}ms  "
                f"cand={entry['cand_ms']:10.3f}ms  speedup={speedup:.2f}x"
            )

    if not speedups:
        result["error"] = "No valid timing results"
        return result

    # Compute harmonic mean (matches AlgoTune methodology)
    n_spd = len(speedups)
    harmonic_mean = n_spd / sum(1.0 / s for s in speedups)

    # Also compute geometric mean for comparison
    log_mean = sum(np.log(s) for s in speedups) / n_spd
    geometric_mean = np.exp(log_mean)

    result["valid"] = True
    result["speedup"] = round(harmonic_mean, 4)
    result["geometric_mean"] = round(geometric_mean, 4)
    result["min_speedup"] = round(min(speedups), 4)
    result["max_speedup"] = round(max(speedups), 4)
    result["per_input"] = per_input

    if n_spd > 1:
        result["cv"] = round(statistics.stdev(speedups) / statistics.mean(speedups), 4)
    else:
        result["cv"] = 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="Rigorous validation of evolved solutions")
    parser.add_argument("--task", help="Validate a single task (default: all valid tasks)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-input details")
    parser.add_argument("--output", default=str(RESULTS_DIR / "validation.json"),
                        help="Output JSON file")
    args = parser.parse_args()

    tasks = [args.task] if args.task else VALID_TASKS

    print(f"Validating {len(tasks)} tasks with rigorous methodology")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}")
    print(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '?')}")
    print(f"  Timing: {NUM_RUNS} runs per input, min time used")
    print()

    all_results = {}
    valid_speedups = []

    for task_name in tasks:
        print(f"[{task_name}] ", end="", flush=True)

        result = validate_task(task_name, verbose=args.verbose)
        all_results[task_name] = result

        if result["valid"]:
            s = result["speedup"]
            valid_speedups.append(s)
            print(f"PASS  speedup={s:.2f}x  (range: {result['min_speedup']:.2f}x - {result['max_speedup']:.2f}x)")
        else:
            print(f"FAIL  {result['error']}")

    # Summary
    print(f"\n{'='*70}")
    if valid_speedups:
        n = len(valid_speedups)
        hm = n / sum(1.0 / s for s in valid_speedups)
        gm = np.exp(sum(np.log(s) for s in valid_speedups) / n)
        print(f"Valid: {n}/{len(tasks)}")
        print(f"Harmonic mean speedup: {hm:.2f}x")
        print(f"Geometric mean speedup: {gm:.2f}x")
        print(f"Range: {min(valid_speedups):.2f}x - {max(valid_speedups):.2f}x")

        # Load previous results for comparison
        prev_path = RESULTS_DIR / "comparison.json"
        if prev_path.exists():
            with open(prev_path) as f:
                prev = json.load(f)
            prev_hm = prev['summary'].get('harmonic_mean_speedup') or prev['summary'].get('harmonic_mean_speedup_evolution')
            if prev_hm:
                print(f"\nPrevious (evolution eval): {prev_hm:.2f}x")
                print(f"Validated (rigorous):      {hm:.2f}x")
    else:
        print("No valid results!")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    if valid_speedups:
        n = len(valid_speedups)
        summary = {
            "harmonic_mean_speedup": round(n / sum(1.0 / s for s in valid_speedups), 4),
            "geometric_mean_speedup": round(float(np.exp(sum(np.log(s) for s in valid_speedups) / n)), 4),
            "valid_tasks": n,
            "total_tasks": len(tasks),
            "num_runs": NUM_RUNS,
            "methodology": "single-threaded BLAS, min-of-10 timing, calibrated inputs",
        }

    output = {
        "summary": summary,
        "tasks": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
