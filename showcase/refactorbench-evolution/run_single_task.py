#!/usr/bin/env python3
"""
Run a single RefactorBench task for debugging and development.

Usage:
    python3 run_single_task.py --repo flask_refactor --task rename-send-from-directory
    python3 run_single_task.py --repo flask_refactor --task rename-send-from-directory --strategy baseline_strategy.json
    python3 run_single_task.py --list  # List all available tasks
"""

import argparse
import json
import sys
from pathlib import Path

from refactor_agent import (
    BENCH_ROOT,
    get_task_info,
    load_strategy,
    run_agent,
    run_test,
    setup_workdir,
)


def list_all_tasks():
    """List all available descriptive tasks."""
    problems_dir = BENCH_ROOT / "problems" / "descriptive_problems"
    total = 0
    for repo_dir in sorted(problems_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        tasks = sorted(f.stem.replace("-task", "") for f in repo_dir.iterdir() if f.name.endswith("-task.txt"))
        print(f"\n{repo_dir.name} ({len(tasks)} tasks):")
        for t in tasks:
            print(f"  {t}")
        total += len(tasks)
    print(f"\nTotal: {total} tasks")


def main():
    parser = argparse.ArgumentParser(description="Run a single RefactorBench task")
    parser.add_argument("--repo", help="Repository name (e.g., flask_refactor)")
    parser.add_argument("--task", help="Task name (e.g., rename-send-from-directory)")
    parser.add_argument("--strategy", default="baseline_strategy.json", help="Strategy JSON path")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--workdir", default="workdirs", help="Working directory root")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--test-only", action="store_true", help="Only run the test (no agent)")
    args = parser.parse_args()

    if args.list:
        list_all_tasks()
        return

    if not args.repo or not args.task:
        parser.error("--repo and --task are required (or use --list)")

    # Load task info
    task_info = get_task_info(args.repo, args.task)
    print(f"Task: {args.repo}/{args.task}")
    print(f"Description: {task_info['description'][:200]}...")
    print(f"Test file: {task_info['test_file']}")
    print()

    # Setup workdir
    work_root = Path(args.workdir)
    work_root.mkdir(parents=True, exist_ok=True)
    workdir = setup_workdir(args.repo, work_root)
    print(f"Working directory: {workdir}")

    if args.test_only:
        # Just run the test to see if it works on unmodified repo
        print("\nRunning test on unmodified repo...")
        result = run_test(task_info["test_file"], workdir, args.repo)
        print(f"Passed: {result['passed']}")
        if result["stdout"]:
            print(f"stdout:\n{result['stdout']}")
        if result["stderr"]:
            print(f"stderr:\n{result['stderr']}")
        return

    # Load strategy
    strategy = load_strategy(args.strategy)
    print(f"Strategy: {strategy.get('name', 'unknown')}")
    print()

    # Run agent
    print("Running refactoring agent...")
    print("=" * 60)
    result = run_agent(strategy, task_info, workdir, model=args.model, verbose=args.verbose)
    print("=" * 60)
    print()

    print(f"Result: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Turns used: {result.get('turns_used', 0)}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    if result.get("test_results"):
        print("Test history:")
        for tr in result["test_results"]:
            status = "PASS" if tr["passed"] else "FAIL"
            print(f"  Turn {tr['turn']}: {status}")

    print(f"\nFull result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
