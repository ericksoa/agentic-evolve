#!/usr/bin/env python3
"""
Validate evolved solutions using AlgoTune's OFFICIAL evaluation pipeline.

This script uses evaluate_results.py's AgentCompatibleEvaluator, which is
the exact same code path used by AlgoTune's real evaluation system:
  - AGENT_MODE=1 with ISOLATED_EVAL=1 (fresh subprocess per timing run)
  - BaselineManager for baseline time generation (isolated subprocess timing)
  - EvaluationOrchestrator with SolverExecutor (subprocess isolation)
  - 100% validity required for speedup to count
  - forkserver multiprocessing context
  - NUMBA_THREADING_LAYER=workqueue for fork safety

Usage:
    python scripts/validate_algotune_official.py --tasks pagerank cholesky_factorization
    python scripts/validate_algotune_official.py  # All valid tasks
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# CRITICAL: Initialize multiprocessing BEFORE any other imports
import multiprocessing
multiprocessing.freeze_support()
try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    pass

# Set environment BEFORE importing numpy/numba
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
EVOLVED_DIR = SHOWCASE_DIR / "evolved"
RESULTS_DIR = SHOWCASE_DIR / "results"

# Add AlgoTune to path
if str(ALGOTUNE_DIR) not in sys.path:
    sys.path.insert(0, str(ALGOTUNE_DIR))

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
    "dijkstra_from_indices",
]


def evaluate_task_official(task_name: str, data_dir: Path, verbose: bool = False) -> dict:
    """
    Evaluate a single task using AlgoTune's official AgentCompatibleEvaluator.

    This is the same code path used by evaluate_results.py, which is how
    AlgoTune officially evaluates submitted solutions.
    """
    from scripts.evaluate_results import AgentCompatibleEvaluator

    result = {
        "task": task_name,
        "valid": False,
        "speedup": None,
        "error": None,
        "methodology": "AlgoTune official (AgentCompatibleEvaluator)",
    }

    # Check that evolved solver exists
    solver_path = EVOLVED_DIR / task_name / "solver.py"
    if not solver_path.exists():
        result["error"] = f"No evolved solver at {solver_path}"
        return result

    # Create the evaluator with AlgoTune's data directory
    evaluator = AgentCompatibleEvaluator(data_dir)

    # evaluate_task expects: task_name, model_name, code_dir
    # code_dir is the directory containing solver.py
    code_dir = EVOLVED_DIR / task_name

    try:
        eval_result = evaluator.evaluate_task(task_name, "evolved", code_dir)

        if eval_result.success and eval_result.speedup is not None:
            result["valid"] = True
            result["speedup"] = round(eval_result.speedup, 4)
            result["baseline_time_ms"] = round(eval_result.baseline_time_ms, 3) if eval_result.baseline_time_ms else None
            result["optimized_time_ms"] = round(eval_result.optimized_time_ms, 3) if eval_result.optimized_time_ms else None
        else:
            result["error"] = eval_result.error_message or "Evaluation failed (no speedup)"

    except Exception as e:
        import traceback
        result["error"] = f"Exception: {e}"
        if verbose:
            traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate evolved solutions using AlgoTune's official evaluation pipeline"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to evaluate (default: all valid tasks)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--output", default=str(RESULTS_DIR / "validation_official.json"),
        help="Output JSON file"
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    tasks = args.tasks or VALID_TASKS

    # Data directory for AlgoTune datasets
    # Datasets are generated automatically by TaskFactory.load_dataset()
    data_dir = ALGOTUNE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Set DATA_DIR environment variable for AlgoTune
    os.environ["DATA_DIR"] = str(data_dir)

    print(f"Validating {len(tasks)} tasks using AlgoTune's official evaluation pipeline")
    print(f"  AGENT_MODE will be set to 1 (solver loaded from disk)")
    print(f"  ISOLATED_EVAL will be set to 1 (fresh subprocess per timing run)")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}")
    print(f"  NUMBA_THREADING_LAYER={os.environ.get('NUMBA_THREADING_LAYER', '?')}")
    print(f"  Data dir: {data_dir}")
    print()

    all_results = {}
    valid_speedups = []

    for task_name in tasks:
        print(f"[{task_name}] ", end="", flush=True)

        result = evaluate_task_official(task_name, data_dir, verbose=args.verbose)
        all_results[task_name] = result

        if result["valid"]:
            s = result["speedup"]
            valid_speedups.append(s)
            print(f"PASS  speedup={s:.2f}x  (baseline={result.get('baseline_time_ms', '?')}ms, optimized={result.get('optimized_time_ms', '?')}ms)")
        else:
            print(f"FAIL  {result['error']}")

    # Summary
    print(f"\n{'='*70}")
    if valid_speedups:
        n = len(valid_speedups)
        hm = n / sum(1.0 / s for s in valid_speedups)
        import numpy as np
        gm = float(np.exp(sum(np.log(s) for s in valid_speedups) / n))
        print(f"Valid: {n}/{len(tasks)}")
        print(f"Harmonic mean speedup: {hm:.2f}x")
        print(f"Geometric mean speedup: {gm:.2f}x")
        print(f"Range: {min(valid_speedups):.2f}x - {max(valid_speedups):.2f}x")

        # Compare with our custom validation
        our_validation_path = RESULTS_DIR / "validation.json"
        if our_validation_path.exists():
            with open(our_validation_path) as f:
                our = json.load(f)
            our_hm = our["summary"]["harmonic_mean_speedup"]
            print(f"\nOur custom validation: {our_hm:.2f}x")
            print(f"AlgoTune official:     {hm:.2f}x")
            print(f"Difference:            {((hm - our_hm) / our_hm * 100):+.1f}%")
    else:
        print("No valid results!")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    if valid_speedups:
        n = len(valid_speedups)
        import numpy as np
        summary = {
            "harmonic_mean_speedup": round(n / sum(1.0 / s for s in valid_speedups), 4),
            "geometric_mean_speedup": round(float(np.exp(sum(np.log(s) for s in valid_speedups) / n)), 4),
            "valid_tasks": n,
            "total_tasks": len(tasks),
            "methodology": "AlgoTune official AgentCompatibleEvaluator (AGENT_MODE=1, ISOLATED_EVAL=1, subprocess isolation)",
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
