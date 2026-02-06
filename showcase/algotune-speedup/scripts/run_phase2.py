#!/usr/bin/env python3
"""
Phase 2: Run evolve-sdk evolution on top of Phase 1 solutions.

For each task, the Phase 1 baseline solution becomes the starter solution,
and evolve-sdk applies mutation/crossover/selection to improve it further.

Usage:
    python scripts/run_phase2.py                    # All tasks
    python scripts/run_phase2.py --tasks 3           # First 3 tasks (smoke test)
    python scripts/run_phase2.py --task linear_system_solver  # Single task
    python scripts/run_phase2.py --max-generations 5  # Fewer generations
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
SOLUTIONS_DIR = SHOWCASE_DIR / "solutions"
EVOLVED_DIR = SHOWCASE_DIR / "evolved"
RESULTS_DIR = SHOWCASE_DIR / "results"
CONFIGS_DIR = SHOWCASE_DIR / "configs"

# Default evolution parameters (conservative for local machine)
DEFAULT_MAX_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 6
DEFAULT_PLATEAU = 3


def load_results(results_path: Path) -> dict:
    """Load existing Phase 2 results for resume."""
    if results_path.exists():
        return json.loads(results_path.read_text())
    return {"tasks": {}}


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


def check_phase1_solution(task_name: str) -> Path | None:
    """Check if Phase 1 produced a solution for this task."""
    solver = SOLUTIONS_DIR / task_name / "solver.py"
    if solver.exists():
        return solver
    return None


def run_evolution(
    task_name: str,
    max_generations: int,
    population_size: int,
    plateau: int,
) -> dict:
    """
    Run evolve-sdk evolution for a single task.

    Uses the per-task config from configs/{task_name}.json.
    Evolve-sdk handles checkpointing and resume automatically.
    """
    config_path = CONFIGS_DIR / f"{task_name}.json"
    if not config_path.exists():
        return {
            "status": "no_config",
            "error": f"Config not found: {config_path}. Run generate_evolve_configs.py first.",
        }

    # Verify starter solution exists
    starter = check_phase1_solution(task_name)
    if not starter:
        return {
            "status": "no_starter",
            "error": f"No Phase 1 solution for {task_name}. Run Phase 1 first.",
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"

    # Run evolve-sdk
    cmd = [
        sys.executable,
        "-m", "evolve_sdk",
        "--config", str(config_path),
        "--mode", "perf",
        "--max-generations", str(max_generations),
        "--population-size", str(population_size),
        "--plateau", str(plateau),
        "--no-parallel",
    ]

    print(f"  Running: {' '.join(cmd[:6])} ...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(SHOWCASE_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per task
        )

        if result.returncode != 0:
            print(f"  evolve-sdk stderr (last 500 chars): {result.stderr[-500:]}")
            # Still check if a champion was produced
    except subprocess.TimeoutExpired:
        print(f"  Timeout after 30 minutes")
        # Still check for partial results

    # Find the champion solution
    champion_path = find_champion(task_name)

    if champion_path:
        # Copy to evolved/ directory
        evolved_solver = EVOLVED_DIR / task_name / "solver.py"
        evolved_solver.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(champion_path, evolved_solver)

        # Evaluate the evolved solution
        eval_result = evaluate_solution(str(evolved_solver), task_name, env)
        eval_result["status"] = "success"
        eval_result["champion_path"] = str(champion_path)
        eval_result["evolved_path"] = str(evolved_solver)
        return eval_result
    else:
        return {
            "status": "no_champion",
            "error": "Evolution did not produce a champion",
        }


def find_champion(task_name: str) -> Path | None:
    """Find the champion solution from evolve-sdk artifacts."""
    # evolve-sdk stores champions in .evolve-sdk/{problem_id}/
    evolve_dir = SHOWCASE_DIR / ".evolve-sdk"
    if not evolve_dir.exists():
        return None

    # Search for champion.json or the best mutation file
    for candidate_dir in evolve_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        # Check if this dir is for our task (name matching)
        if task_name not in candidate_dir.name:
            continue

        champion_json = candidate_dir / "champion.json"
        if champion_json.exists():
            try:
                data = json.loads(champion_json.read_text())
                champ_file = data.get("file")
                if champ_file:
                    champ_path = Path(champ_file)
                    if champ_path.exists():
                        return champ_path
                    # Try relative to evolve dir
                    champ_path = candidate_dir / champ_file
                    if champ_path.exists():
                        return champ_path
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: find the latest .py file in mutations/
        mutations_dir = candidate_dir / "mutations"
        if mutations_dir.exists():
            py_files = sorted(mutations_dir.glob("*.py"), key=lambda p: p.stat().st_mtime)
            if py_files:
                return py_files[-1]

    return None


def evaluate_solution(solution_path: str, task_name: str, env: dict) -> dict:
    """Evaluate a solution using our bridge script."""
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
        description="Phase 2: Evolve-SDK evolution on Phase 1 solutions"
    )
    parser.add_argument("--task", type=str, help="Run a single specific task")
    parser.add_argument("--tasks", type=int, help="Run first N tasks (smoke test)")
    parser.add_argument(
        "--max-generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help=f"Max generations (default: {DEFAULT_MAX_GENERATIONS})",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"Population size (default: {DEFAULT_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--plateau",
        type=int,
        default=DEFAULT_PLATEAU,
        help=f"Plateau threshold (default: {DEFAULT_PLATEAU})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if result exists"
    )
    args = parser.parse_args()

    tasks = get_task_list(args)
    results_path = RESULTS_DIR / "phase2_evolved.json"
    results = load_results(results_path)

    # Load Phase 1 results for comparison
    phase1_path = RESULTS_DIR / "phase1_baseline.json"
    phase1 = json.loads(phase1_path.read_text()) if phase1_path.exists() else {"tasks": {}}

    print(f"=== Phase 2: Evolve-SDK Evolution ===")
    print(f"Tasks: {len(tasks)}")
    print(
        f"Config: {args.max_generations} gens, "
        f"pop={args.population_size}, plateau={args.plateau}"
    )
    print()

    for i, task in enumerate(tasks, 1):
        if task in results["tasks"] and not args.force:
            existing = results["tasks"][task]
            print(
                f"[{i}/{len(tasks)}] {task}: "
                f"already done (speedup={existing.get('speedup', '?')}x) — skipping"
            )
            continue

        # Check Phase 1 prerequisite
        if not check_phase1_solution(task):
            print(f"[{i}/{len(tasks)}] {task}: no Phase 1 solution — skipping")
            results["tasks"][task] = {
                "status": "no_starter",
                "error": "Phase 1 solution missing",
            }
            save_results(results, results_path)
            continue

        # Show Phase 1 baseline for context
        p1 = phase1.get("tasks", {}).get(task, {})
        p1_speedup = p1.get("speedup", "?")
        print(f"[{i}/{len(tasks)}] {task}: evolving (Phase 1 baseline: {p1_speedup}x)...")

        task_result = run_evolution(
            task_name=task,
            max_generations=args.max_generations,
            population_size=args.population_size,
            plateau=args.plateau,
        )
        results["tasks"][task] = task_result

        # Save incrementally
        save_results(results, results_path)

        speedup = task_result.get("speedup", 0)
        status = task_result.get("status", "unknown")
        improvement = ""
        if isinstance(p1_speedup, (int, float)) and speedup > 0:
            delta = speedup - p1_speedup
            improvement = f" (delta: {'+' if delta > 0 else ''}{delta:.2f}x)"
        print(f"  -> {status}: speedup={speedup}x{improvement}")
        print()

    # Summary
    print("=== Phase 2 Summary ===")
    valid_speedups = [
        r["speedup"]
        for r in results["tasks"].values()
        if r.get("valid") and r.get("speedup", 0) > 0
    ]
    if valid_speedups:
        n = len(valid_speedups)
        harmonic = n / sum(1.0 / s for s in valid_speedups)
        print(f"Valid evolved tasks: {n}/{len(tasks)}")
        print(f"Harmonic mean speedup: {harmonic:.2f}x")
    else:
        print("No valid evolved results yet.")

    save_results(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
