#!/usr/bin/env python3
"""
Phase 1: Run AlgoTune's built-in agent with Opus 4.6 per task.

This produces baseline solutions by letting the AlgoTune agent optimize
each task using Claude Opus 4.6 as the underlying model.

Results are saved incrementally so the run can be resumed.

Usage:
    python scripts/run_phase1.py                    # All tasks from subset.json
    python scripts/run_phase1.py --tasks 3           # First 3 tasks only (smoke test)
    python scripts/run_phase1.py --task linear_system_solver  # Single task
    python scripts/run_phase1.py --dry-run           # Show what would run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
SOLUTIONS_DIR = SHOWCASE_DIR / "solutions"
RESULTS_DIR = SHOWCASE_DIR / "results"
CONFIGS_DIR = SHOWCASE_DIR / "configs"

# AlgoTune agent entry point
ALGOTUNE_MAIN = ALGOTUNE_DIR / "algotune.py"
ALGOTUNE_SCRIPTS = ALGOTUNE_DIR / "scripts" / "algotune.py"


def load_results(results_path: Path) -> dict:
    """Load existing results for resume capability."""
    if results_path.exists():
        return json.loads(results_path.read_text())
    return {"tasks": {}, "model": "claude-opus-4-6"}


def save_results(results: dict, results_path: Path):
    """Save results incrementally."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


def get_task_list(args) -> list:
    """Get the list of tasks to run."""
    if args.task:
        return [args.task]

    subset_path = CONFIGS_DIR / "subset.json"
    if subset_path.exists():
        data = json.loads(subset_path.read_text())
        tasks = data["tasks"]
    else:
        print("No subset.json found. Run select_subset.py first.")
        sys.exit(1)

    if args.tasks:
        tasks = tasks[: args.tasks]

    return tasks


def run_algotune_agent(task_name: str, output_dir: Path) -> dict:
    """
    Run AlgoTune's agent on a single task.

    The agent writes its optimized solver.py to a working directory.
    We then evaluate it using our bridge for comparable measurements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the correct entry point
    if ALGOTUNE_SCRIPTS.exists():
        entry = str(ALGOTUNE_SCRIPTS)
    elif ALGOTUNE_MAIN.exists():
        entry = str(ALGOTUNE_MAIN)
    else:
        # Fallback: try running as module
        entry = None

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"

    # Build the command to run AlgoTune's agent
    if entry:
        cmd = [
            sys.executable,
            entry,
            "--task", task_name,
            "--model", "claude-opus-4-6",
            "--output_dir", str(output_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m", "AlgoTuner",
            "--task", task_name,
            "--model", "claude-opus-4-6",
            "--output_dir", str(output_dir),
        ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(ALGOTUNE_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per task
        )

        if result.returncode != 0:
            print(f"  Agent stderr: {result.stderr[:500]}")
            return {
                "status": "agent_error",
                "returncode": result.returncode,
                "stderr": result.stderr[:1000],
            }

    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except FileNotFoundError as e:
        return {"status": "not_found", "error": str(e)}

    # Check for output solver.py
    solver_path = output_dir / "solver.py"
    if not solver_path.exists():
        # Search for it in subdirectories
        for candidate in output_dir.rglob("solver.py"):
            solver_path = candidate
            break

    if not solver_path.exists():
        return {"status": "no_solver", "error": "Agent did not produce solver.py"}

    # Copy solver to canonical location
    canonical = SOLUTIONS_DIR / task_name / "solver.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solver_path, canonical)

    # Evaluate using our bridge
    eval_result = evaluate_solution(str(canonical), task_name)
    eval_result["status"] = "success"
    eval_result["solver_path"] = str(canonical)

    return eval_result


def evaluate_solution(solution_path: str, task_name: str) -> dict:
    """Evaluate a solution using our bridge script."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        str(SHOWCASE_DIR / "evaluate_task.py"),
        solution_path,
        "--task", task_name,
        "--json",
        "--runs", "3",
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        else:
            return {
                "fitness": 0.0,
                "valid": False,
                "error": f"Evaluation failed: {result.stderr[:500]}",
            }
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"fitness": 0.0, "valid": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Run AlgoTune agent baseline"
    )
    parser.add_argument("--task", type=str, help="Run a single specific task")
    parser.add_argument("--tasks", type=int, help="Run first N tasks (smoke test)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show tasks without running"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if result exists"
    )
    args = parser.parse_args()

    tasks = get_task_list(args)
    results_path = RESULTS_DIR / "phase1_baseline.json"
    results = load_results(results_path)

    print(f"=== Phase 1: AlgoTune Agent Baseline (Opus 4.6) ===")
    print(f"Tasks: {len(tasks)}")
    print()

    if args.dry_run:
        for i, task in enumerate(tasks, 1):
            status = "done" if task in results["tasks"] else "pending"
            print(f"  [{status}] {i}. {task}")
        return

    for i, task in enumerate(tasks, 1):
        if task in results["tasks"] and not args.force:
            existing = results["tasks"][task]
            print(
                f"[{i}/{len(tasks)}] {task}: "
                f"already done (speedup={existing.get('speedup', '?')}x) — skipping"
            )
            continue

        print(f"[{i}/{len(tasks)}] {task}: running AlgoTune agent...")
        output_dir = SHOWCASE_DIR / "workdirs" / f"phase1_{task}"

        task_result = run_algotune_agent(task, output_dir)
        results["tasks"][task] = task_result

        # Save incrementally
        save_results(results, results_path)

        speedup = task_result.get("speedup", 0)
        status = task_result.get("status", "unknown")
        print(f"  -> {status}: speedup={speedup}x")
        print()

    # Summary
    print("=== Phase 1 Summary ===")
    valid_speedups = [
        r["speedup"]
        for r in results["tasks"].values()
        if r.get("valid") and r.get("speedup", 0) > 0
    ]
    if valid_speedups:
        n = len(valid_speedups)
        harmonic = n / sum(1.0 / s for s in valid_speedups)
        print(f"Valid tasks: {n}/{len(tasks)}")
        print(f"Harmonic mean speedup: {harmonic:.2f}x")
    else:
        print("No valid results yet.")

    save_results(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
