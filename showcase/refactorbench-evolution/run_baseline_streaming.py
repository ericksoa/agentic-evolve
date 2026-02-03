#!/usr/bin/env python3
"""Run baseline evaluation sequentially, printing running average after each task."""

import json
import sys
import time
from pathlib import Path

from refactor_agent import get_task_info, load_strategy, run_agent, setup_workdir
from evaluate import get_all_tasks

import shutil


def main():
    strategy_path = "baseline_strategy.json"
    strategy = load_strategy(strategy_path)
    tasks = get_all_tasks()
    total = len(tasks)

    work_root = Path("workdirs")
    work_root.mkdir(parents=True, exist_ok=True)

    passed = 0
    completed = 0
    results = []

    print(f"Running baseline on all {total} tasks sequentially...")
    print(f"{'#':<4} {'Repo/Task':<60} {'Result':<8} {'Turns':<6} {'Running Avg':<12} {'Pass/Done'}")
    print("-" * 110)

    start = time.time()

    for repo_name, task_name in tasks:
        completed += 1
        task_label = f"{repo_name}/{task_name}"

        try:
            task_info = get_task_info(repo_name, task_name)
            task_work_root = work_root / f"{repo_name}__{task_name}"
            task_work_root.mkdir(parents=True, exist_ok=True)
            workdir = setup_workdir(repo_name, task_work_root)

            result = run_agent(strategy, task_info, workdir, verbose=False)

            task_passed = result["passed"]
            turns = result.get("turns_used", 0)

            # Clean up
            try:
                shutil.rmtree(task_work_root)
            except Exception:
                pass

        except Exception as e:
            task_passed = False
            turns = 0

        if task_passed:
            passed += 1

        running_avg = passed / completed * 100
        status = "PASS" if task_passed else "FAIL"

        print(f"{completed:<4} {task_label:<60} {status:<8} {turns:<6} {running_avg:>6.1f}%      {passed}/{completed}")
        sys.stdout.flush()

        results.append({
            "repo": repo_name,
            "task": task_name,
            "passed": task_passed,
            "turns_used": turns,
        })

    elapsed = time.time() - start
    print("-" * 110)
    print(f"FINAL: {passed}/{total} passed ({passed/total*100:.1f}%) in {elapsed:.0f}s")

    # Save results
    output = {
        "fitness": round(passed / total, 4),
        "valid": True,
        "passed": passed,
        "total": total,
        "pass_rate_pct": round(passed / total * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
        "task_details": results,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "baseline_full.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/baseline_full.json")


if __name__ == "__main__":
    main()
