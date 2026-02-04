#!/usr/bin/env python3
"""
Evolution bridge for a single RefactorBench task.

Picks up where RALPH left off. Collects the terminal workdir states from
RALPH chains, scores them, deduplicates, seeds them into the agentic-evolve
SDK's EvolutionRunner with workspace_mode=True, and lets the SDK run
the full evolution loop with all its features (trust, memory, diversity,
plateau breaker, etc.).

Usage:
    # After RALPH has run on a task:
    python3 evolve_task.py --repo django_refactor --task add-log-parameter-get-resolver

    # Skip RALPH seeding, start from scratch:
    python3 evolve_task.py --repo django_refactor --task add-log-parameter-get-resolver --no-seed
"""

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from refactor_agent import get_task_info, run_test, setup_workdir, PYTHON_EXE

# SDK imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk"))
from evolve_sdk.runner import EvolutionRunner


# ---------------------------------------------------------------------------
# RALPH seed collection
# ---------------------------------------------------------------------------

def compute_granular_fitness(test_result: dict) -> dict:
    """Parse pytest -v output to count individual test passes/failures."""
    stdout = test_result.get("stdout", "")

    # Get authoritative total from pytest's "collected N items" line.
    # Format: "collected 3 items" or "collected 2 items / 1 error"
    collected_match = re.search(r'collected\s+(\d+)\s+items?', stdout)
    collection_err_match = re.search(r'collected\s+\d+\s+items?\s*/\s*(\d+)\s+errors?', stdout)
    collected_total = 0
    if collected_match:
        collected_total = int(collected_match.group(1))
    if collection_err_match:
        collected_total += int(collection_err_match.group(1))

    test_lines = re.findall(
        r'([\w/\.\-]+::[\w\[\]\-]+)\s+(PASSED|FAILED|ERROR)', stdout
    )

    if test_lines:
        tests_passed = sum(1 for _, status in test_lines if status == "PASSED")
        failing_tests = [name for name, status in test_lines if status != "PASSED"]
        tests_total = max(collected_total, len(test_lines))
    else:
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        failed_match = re.search(r'(\d+)\s+failed', stdout)
        error_match = re.search(r'(\d+)\s+error', stdout)

        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_errored = int(error_match.group(1)) if error_match else 0
        tests_total = max(collected_total, tests_passed + tests_failed + tests_errored)
        failing_tests = []

        if tests_total == 0 and not test_result.get("passed"):
            tests_total = 1

    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "failing_tests": failing_tests,
    }


def evaluate_workspace(workspace: Path, task_info: dict) -> dict:
    """Evaluate a workspace directory with run_test and return fitness dict."""
    test_result = run_test(task_info["test_file"], workspace, task_info["repo_name"])
    granular = compute_granular_fitness(test_result)

    tests_total = max(granular["tests_total"], 1)
    fitness = granular["tests_passed"] / tests_total

    return {
        "fitness": round(fitness, 4),
        "tests_passed": granular["tests_passed"],
        "tests_total": tests_total,
        "failing_tests": granular["failing_tests"],
        "test_output": test_result.get("stdout", "")[-2000:],
        "passed": test_result["passed"],
    }


def collect_ralph_seeds(
    repo: str, task: str, ralph_root: Path, task_info: dict,
) -> list[dict]:
    """Collect and score RALPH chain terminal states.

    Returns list of population dicts compatible with EvolutionRunner.population
    (each has 'file' and 'fitness' at minimum).
    """
    task_dir = ralph_root / f"{repo}__{task}"
    if not task_dir.exists():
        print(f"  No RALPH workdirs found at {task_dir}")
        return []

    candidates = []

    for chain_dir in sorted(task_dir.iterdir()):
        if not chain_dir.is_dir() or not chain_dir.name.startswith("chain_"):
            continue

        chain_id = chain_dir.name
        repo_dir = chain_dir / repo
        if not repo_dir.exists():
            continue

        print(f"  Scoring {chain_id}...", end=" ", flush=True)
        result = evaluate_workspace(repo_dir, task_info)
        print(f"{result['tests_passed']}/{result['tests_total']} "
              f"({result['fitness']:.2%})")

        candidates.append({
            "file": str(repo_dir),
            "fitness": result["fitness"],
            "tests_passed": result["tests_passed"],
            "tests_total": result["tests_total"],
            "failing_tests": result["failing_tests"],
            "test_output": result["test_output"],
            "origin": f"ralph_{chain_id}",
            "valid": True,
        })

    return candidates


def deduplicate(individuals: list[dict]) -> list[dict]:
    """Remove near-duplicate individuals by hashing .py file contents."""
    seen = set()
    unique = []

    for ind in individuals:
        workspace = Path(ind["file"])
        h = hashlib.md5()
        for py_file in sorted(workspace.rglob("*.py")):
            try:
                h.update(py_file.read_bytes())
            except OSError:
                pass
        digest = h.hexdigest()

        if digest not in seen:
            seen.add(digest)
            unique.append(ind)

    removed = len(individuals) - len(unique)
    if removed:
        print(f"  Deduplicated: {removed} duplicates removed, {len(unique)} unique")

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evolution bridge — seeds EvolutionRunner from RALPH for a single task"
    )
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--generations", type=int, default=20, help="Max generations (default: 20)")
    parser.add_argument("--population", type=int, default=6, help="Population size (default: 6)")
    parser.add_argument("--mutations", type=int, default=3, help="Mutations per gen (default: 3)")
    parser.add_argument("--plateau", type=int, default=5, help="Plateau threshold (default: 5)")
    parser.add_argument("--no-seed", action="store_true", help="Don't seed from RALPH")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.resolve()
    task_info = get_task_info(args.repo, args.task)
    task_id = f"{args.repo}/{args.task}"

    # Build workspace_test_command for the evaluator agent
    test_script = project_dir / "workspace_test.py"
    workspace_test_cmd = (
        f"{PYTHON_EXE} {test_script} {{workspace}} "
        f"--repo {args.repo} --task {args.task}"
    )

    # Build workspace_source (unmodified baseline)
    baseline_parent = project_dir / "evolve_workdirs" / f"{args.repo}__{args.task}" / "baseline"
    baseline_parent.mkdir(parents=True, exist_ok=True)
    baseline_dir = setup_workdir(args.repo, baseline_parent)

    # Create the SDK runner
    runner = EvolutionRunner(
        problem=f"Fix failing tests in {task_id} (RefactorBench multi-file refactoring)",
        mode="ml",
        cwd=str(project_dir),
        workspace_mode=True,
        workspace_source=str(baseline_dir),
        workspace_test_command=workspace_test_cmd,
        workspace_test_file=task_info.get("test_file"),
        max_generations=args.generations,
        population_size=args.population,
        elite_count=min(3, args.population),
        mutation_variants=args.mutations,
        plateau_threshold=args.plateau,
        test_command=workspace_test_cmd,
    )

    # Seed population from RALPH (or baseline)
    population = []

    if not args.no_seed:
        print(f"\nCollecting RALPH seeds for {task_id}...")
        ralph_root = project_dir / "ralph_workdirs"
        seeds = collect_ralph_seeds(args.repo, args.task, ralph_root, task_info)

        if seeds:
            # Check if already solved
            for s in seeds:
                if s["fitness"] >= 1.0:
                    print(f"\nTask already SOLVED by {s['origin']}!")
                    print(json.dumps({"status": "solved_by_ralph", "task_id": task_id}, indent=2))
                    return

            seeds = deduplicate(seeds)
            seeds.sort(key=lambda x: x["fitness"], reverse=True)
            population = seeds[:args.population]

    if not population:
        # Start from unmodified baseline
        print(f"\nNo RALPH seeds, evaluating baseline...")
        result = evaluate_workspace(baseline_dir, task_info)
        population = [{
            "file": str(baseline_dir),
            "fitness": result["fitness"],
            "tests_passed": result["tests_passed"],
            "tests_total": result["tests_total"],
            "failing_tests": result["failing_tests"],
            "test_output": result["test_output"],
            "origin": "baseline",
            "valid": True,
        }]

    print(f"\nSeeding EvolutionRunner with {len(population)} individuals")
    print(f"Best seed: {population[0]['fitness']:.2%}")

    # Write the state file so runner.run() picks it up via _load_state()
    runner.work_dir.mkdir(parents=True, exist_ok=True)
    runner.mutations_dir.mkdir(parents=True, exist_ok=True)

    champion = max(population, key=lambda x: x["fitness"])
    state = {
        "problem": runner.config.problem,
        "mode": runner.config.mode,
        "generation": 0,
        "population": population,
        "champion": champion,
        "history": [],
        "config": {
            "max_generations": runner.config.max_generations,
            "population_size": runner.config.population_size,
            "elite_count": runner.config.elite_count,
            "minimize": runner.config.minimize,
        },
        "updated_at": datetime.now().isoformat(),
    }
    state_file = runner.work_dir / "evolution.json"
    state_file.write_text(json.dumps(state, indent=2, default=str))

    # Run evolution — the SDK handles everything from here
    result = asyncio.run(runner.run())
    print(json.dumps(result, indent=2, default=str))

    # Update progress.json
    champion_fitness = 0
    if isinstance(result, dict):
        champion_fitness = result.get("fitness", 0)

    if champion_fitness >= 1.0:
        update_progress(
            args.repo, args.task, passed=True, method="evolution",
            details=f"Solved via evolution, {args.generations} max gens",
        )
    else:
        update_progress(
            args.repo, args.task, passed=False, method="evolution_failed",
            details=f"Failed after evolution, best fitness {champion_fitness:.4f}",
        )


def update_progress(repo: str, task: str, passed: bool, method: str, details: str) -> None:
    """Update results/progress.json with the outcome of a task run."""
    progress_file = Path(__file__).parent.resolve() / "results" / "progress.json"
    if not progress_file.exists():
        print(f"  WARNING: {progress_file} not found, skipping progress update")
        return

    progress = json.loads(progress_file.read_text())

    for t in progress["tasks"]:
        if t["repo"] == repo and t["task"] == task:
            t["current_passed"] = passed
            t["method"] = method
            t["details"] = details
            break

    progress["current_passed"] = sum(1 for t in progress["tasks"] if t["current_passed"])
    progress["updated_at"] = datetime.now().isoformat()
    progress_file.write_text(json.dumps(progress, indent=2))

    print(f"\n  Progress updated: {progress['current_passed']}/{progress['current_total']}")


if __name__ == "__main__":
    main()
