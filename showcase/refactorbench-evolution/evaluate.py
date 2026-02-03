#!/usr/bin/env python3
"""
Evaluation harness for RefactorBench evolution.

Usage:
    python3 evaluate.py <strategy_json> --json [--subset N] [--tasks repo:task,...]
    python3 evaluate.py <strategy_json> --json --subset 30

Outputs JSON: {"fitness": pass_rate, "valid": true, "passed": N, "total": M, "task_details": [...]}
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import from refactor_agent
from refactor_agent import (
    BENCH_ROOT,
    get_task_info,
    load_strategy,
    run_agent,
    setup_workdir,
)

# All 100 descriptive tasks indexed by repo
TASK_CATALOG = {}


def build_task_catalog() -> dict[str, list[str]]:
    """Build catalog of all descriptive tasks by repo."""
    catalog = {}
    problems_dir = BENCH_ROOT / "problems" / "descriptive_problems"
    for repo_dir in sorted(problems_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name
        tasks = []
        for task_file in sorted(repo_dir.iterdir()):
            if task_file.name.endswith("-task.txt"):
                task_name = task_file.name.replace("-task.txt", "")
                tasks.append(task_name)
        catalog[repo_name] = tasks
    return catalog


def get_all_tasks() -> list[tuple[str, str]]:
    """Return list of (repo_name, task_name) for all 100 tasks."""
    catalog = build_task_catalog()
    all_tasks = []
    for repo_name, tasks in sorted(catalog.items()):
        for task_name in tasks:
            all_tasks.append((repo_name, task_name))
    return all_tasks


def get_representative_subset(n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Get a representative subset balanced across repos."""
    catalog = build_task_catalog()
    rng = random.Random(seed)

    all_tasks = get_all_tasks()
    if n >= len(all_tasks):
        return all_tasks

    # Stratified sampling: pick proportionally from each repo
    total = len(all_tasks)
    selected = []
    for repo_name, tasks in sorted(catalog.items()):
        repo_count = max(1, round(n * len(tasks) / total))
        repo_sample = rng.sample(tasks, min(repo_count, len(tasks)))
        for task_name in repo_sample:
            selected.append((repo_name, task_name))

    # Trim or pad to exact n
    if len(selected) > n:
        selected = rng.sample(selected, n)
    elif len(selected) < n:
        remaining = [t for t in all_tasks if t not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


def evaluate_single_task(strategy_path: str, repo_name: str, task_name: str,
                         work_root: str, model: str = None, verbose: bool = False) -> dict:
    """Evaluate a single task. Designed to run in a subprocess."""
    try:
        strategy = load_strategy(strategy_path)
        task_info = get_task_info(repo_name, task_name)

        work_root_path = Path(work_root)
        # Use unique workdir per task to avoid conflicts
        task_work_root = work_root_path / f"{repo_name}__{task_name}"
        task_work_root.mkdir(parents=True, exist_ok=True)
        workdir = setup_workdir(repo_name, task_work_root)

        result = run_agent(strategy, task_info, workdir, model=model, verbose=verbose)

        # Clean up workdir
        try:
            shutil.rmtree(task_work_root)
        except Exception:
            pass

        return {
            "repo": repo_name,
            "task": task_name,
            "passed": result["passed"],
            "turns_used": result.get("turns_used", 0),
            "error": result.get("error"),
        }

    except Exception as e:
        return {
            "repo": repo_name,
            "task": task_name,
            "passed": False,
            "turns_used": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate refactoring strategy on RefactorBench")
    parser.add_argument("strategy", help="Path to strategy JSON file (the {solution})")
    parser.add_argument("--json", action="store_true", help="Output JSON for evolve-sdk")
    parser.add_argument("--subset", type=int, default=0, help="Use N-task representative subset (0 = all 100)")
    parser.add_argument("--tasks", help="Comma-separated repo:task pairs to run")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel workers (default: 4)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--workdir", default="workdirs", help="Working directory root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")
    args = parser.parse_args()

    # Validate strategy file
    if not os.path.exists(args.strategy):
        result = {"fitness": 0.0, "valid": False, "error": f"Strategy file not found: {args.strategy}"}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Load and validate strategy
    try:
        strategy = load_strategy(args.strategy)
    except Exception as e:
        result = {"fitness": 0.0, "valid": False, "error": f"Invalid strategy JSON: {e}"}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Select tasks
    if args.tasks:
        tasks = []
        for pair in args.tasks.split(","):
            repo, task = pair.strip().split(":")
            tasks.append((repo, task))
    elif args.subset > 0:
        tasks = get_representative_subset(args.subset, seed=args.seed)
    else:
        tasks = get_all_tasks()

    total = len(tasks)
    if not args.json:
        print(f"Evaluating strategy '{strategy.get('name', 'unknown')}' on {total} tasks...")
        print(f"Using up to {args.parallel} parallel workers")

    # Run evaluations
    work_root = Path(args.workdir)
    work_root.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()

    if args.parallel <= 1:
        # Sequential execution
        for i, (repo_name, task_name) in enumerate(tasks):
            if not args.json:
                print(f"  [{i + 1}/{total}] {repo_name}/{task_name}...", end=" ", flush=True)
            r = evaluate_single_task(args.strategy, repo_name, task_name,
                                     str(work_root), model=args.model, verbose=args.verbose)
            results.append(r)
            if not args.json:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"{status} ({r['turns_used']} turns)")
    else:
        # Parallel execution
        futures = {}
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            for repo_name, task_name in tasks:
                f = executor.submit(
                    evaluate_single_task,
                    args.strategy, repo_name, task_name,
                    str(work_root), args.model, args.verbose,
                )
                futures[f] = (repo_name, task_name)

            completed = 0
            for f in as_completed(futures):
                completed += 1
                repo_name, task_name = futures[f]
                try:
                    r = f.result(timeout=600)
                except Exception as e:
                    r = {
                        "repo": repo_name,
                        "task": task_name,
                        "passed": False,
                        "turns_used": 0,
                        "error": str(e),
                    }
                results.append(r)
                if not args.json:
                    status = "PASS" if r["passed"] else "FAIL"
                    print(f"  [{completed}/{total}] {repo_name}/{task_name}: {status}")

    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / total if total > 0 else 0.0

    output = {
        "fitness": round(pass_rate, 4),
        "valid": True,
        "passed": passed,
        "total": total,
        "pass_rate_pct": round(pass_rate * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
        "task_details": results,
    }

    if args.json:
        print(json.dumps(output))
    else:
        print(f"\n{'=' * 60}")
        print(f"Results: {passed}/{total} passed ({pass_rate * 100:.1f}%)")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'=' * 60}")

        # Per-repo breakdown
        repo_stats = {}
        for r in results:
            repo = r["repo"]
            if repo not in repo_stats:
                repo_stats[repo] = {"passed": 0, "total": 0}
            repo_stats[repo]["total"] += 1
            if r["passed"]:
                repo_stats[repo]["passed"] += 1

        print("\nPer-repo breakdown:")
        for repo, stats in sorted(repo_stats.items()):
            pct = stats["passed"] / stats["total"] * 100
            print(f"  {repo}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")

        # Failed tasks
        failures = [r for r in results if not r["passed"]]
        if failures:
            print(f"\nFailed tasks ({len(failures)}):")
            for r in failures[:20]:
                err = f" - {r['error']}" if r.get("error") else ""
                print(f"  {r['repo']}/{r['task']}{err}")
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
