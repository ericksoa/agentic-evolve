#!/usr/bin/env python3
"""
Select a subset of AlgoTune tasks with the most optimization headroom.

Strategy: Pick tasks where existing models (Opus 4.5, GPT-5.2) scored <1.5x,
meaning there's significant room for improvement with better optimization.

Output: configs/subset.json
"""

import json
import os
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
CONFIGS_DIR = SHOWCASE_DIR / "configs"

# Known Opus 4.5 results per task (from AlgoTune leaderboard).
# These represent the baseline we want to beat.
# Tasks with low speedups have the most headroom for improvement.
# This dict is populated from the AlgoTune results/ directory if available,
# otherwise falls back to curated defaults based on published results.
DEFAULT_LOW_HEADROOM_TASKS = [
    # Linear algebra tasks — often backed by optimized LAPACK, but algorithmic
    # shortcuts (blocked algorithms, Strassen, GPU offload) can help
    "cholesky_factorization",
    "eigenvalue_decomposition",
    "lu_factorization",
    "qr_factorization",
    "singular_value_decomposition",
    "matrix_exponential",
    "linear_system_solver",
    # Optimization tasks — solver choice, warm-starting, preconditioning matter
    "linear_programming",
    "group_lasso",
    "basis_pursuit",
    "lasso_regression",
    "quadratic_programming",
    # Signal processing — FFT backends, vectorization, chunking
    "fft_convolution",
    "discrete_cosine_transform",
    "signal_correlation",
    "spectral_estimation",
    # Cryptography — hard to beat C backends, but parallelism/batching possible
    "aes_gcm_encryption",
    "chacha20_encryption",
    # Differential equations — solver selection, adaptive stepping, JIT
    "ode_lorenz96",
    "ode_brusselator",
    "ode_fitzhugh_nagumo",
    # Graph algorithms — algorithmic improvements, C extensions
    "max_clique",
    "graph_coloring",
    "shortest_path",
    # Computational geometry
    "convex_hull",
    "voronoi_diagram",
]


def scan_algotune_results(results_dir: Path) -> dict:
    """Parse AlgoTune results directory for per-task speedups by model."""
    model_results = {}
    if not results_dir.exists():
        return model_results

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_results[model_name] = {}

        for result_file in model_dir.glob("*.json"):
            try:
                data = json.loads(result_file.read_text())
                if isinstance(data, dict):
                    # Try common result formats
                    task_name = result_file.stem
                    speedup = data.get("speedup", data.get("mean_speedup", None))
                    if speedup is not None:
                        model_results[model_name][task_name] = float(speedup)
            except (json.JSONDecodeError, ValueError):
                continue

    return model_results


def list_all_tasks(algotune_dir: Path) -> list:
    """List all available AlgoTune tasks."""
    tasks_dir = algotune_dir / "AlgoTuneTasks"
    if not tasks_dir.exists():
        print(f"Warning: {tasks_dir} not found. Using default task list.")
        return DEFAULT_LOW_HEADROOM_TASKS

    tasks = []
    for item in sorted(tasks_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("_"):
            # Check if it has a task implementation file
            task_file = item / f"{item.name}.py"
            if task_file.exists():
                tasks.append(item.name)
    return tasks


def select_subset(
    target_count: int = 25,
    max_existing_speedup: float = 1.5,
) -> list:
    """
    Select tasks with the most optimization headroom.

    Priority:
    1. Tasks where best existing model scored < max_existing_speedup
    2. Tasks in categories known to be amenable to optimization
    3. Diversity across task categories
    """
    all_tasks = list_all_tasks(ALGOTUNE_DIR)
    print(f"Found {len(all_tasks)} total tasks")

    # Try to get actual results
    results = scan_algotune_results(ALGOTUNE_DIR / "results")

    if results:
        # Find best speedup per task across all models
        best_per_task = {}
        for model, task_speedups in results.items():
            for task, speedup in task_speedups.items():
                if task not in best_per_task or speedup > best_per_task[task]:
                    best_per_task[task] = speedup

        # Select tasks with low existing speedups
        low_speedup_tasks = [
            (task, speedup)
            for task, speedup in sorted(best_per_task.items(), key=lambda x: x[1])
            if speedup < max_existing_speedup and task in all_tasks
        ]

        if len(low_speedup_tasks) >= target_count:
            selected = [t[0] for t in low_speedup_tasks[:target_count]]
            print(f"Selected {len(selected)} tasks with speedup < {max_existing_speedup}x")
            return selected

    # Fallback: use curated default list, filtered to available tasks
    available_defaults = [t for t in DEFAULT_LOW_HEADROOM_TASKS if t in all_tasks]

    if len(available_defaults) >= target_count:
        selected = available_defaults[:target_count]
    else:
        # Pad with remaining tasks
        remaining = [t for t in all_tasks if t not in available_defaults]
        selected = available_defaults + remaining[: target_count - len(available_defaults)]

    print(f"Selected {len(selected)} tasks (curated defaults + available)")
    return selected


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Select AlgoTune task subset")
    parser.add_argument(
        "--count", type=int, default=25, help="Number of tasks to select (default: 25)"
    )
    parser.add_argument(
        "--max-speedup",
        type=float,
        default=1.5,
        help="Max existing speedup threshold (default: 1.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CONFIGS_DIR / "subset.json"),
        help="Output file path",
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_subset(
        target_count=args.count,
        max_existing_speedup=args.max_speedup,
    )

    output = {
        "tasks": selected,
        "count": len(selected),
        "selection_criteria": {
            "max_existing_speedup": args.max_speedup,
            "target_count": args.count,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSubset saved to {output_path}")
    print(f"Tasks: {', '.join(selected)}")


if __name__ == "__main__":
    main()
