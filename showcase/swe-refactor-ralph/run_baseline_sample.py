#!/usr/bin/env python3
"""
Run a baseline sample of SWE-Refactor tasks to establish initial metrics.

Runs single-shot baseline on a stratified sample:
- 2 tasks per refactoring type
- From different projects
"""

import json
import sys
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import swe_refactor_agent as agent

RESULTS_FILE = Path(__file__).parent / "results" / "baseline_sample.json"


def select_sample_tasks(n_per_type: int = 2) -> list[dict]:
    """Select a stratified sample of tasks."""
    tasks = agent.load_tasks()

    # Group by type
    by_type = defaultdict(list)
    for t in tasks:
        by_type[t["type"]].append(t)

    # Sample from each type
    sample = []
    for refactor_type, type_tasks in by_type.items():
        # Shuffle to get variety of projects
        random.shuffle(type_tasks)
        sample.extend(type_tasks[:n_per_type])

    return sample


def run_baseline_sample(n_per_type: int = 2, model: str = "opus"):
    """Run baseline on a sample of tasks."""
    sample = select_sample_tasks(n_per_type)

    print(f"Running baseline on {len(sample)} tasks")
    print(f"Model: {model}")
    print("=" * 60)

    results = {
        "model": model,
        "n_per_type": n_per_type,
        "started_at": datetime.now().isoformat(),
        "tasks": [],
    }

    passed = 0
    for i, task in enumerate(sample):
        task_id = task["uniqueId"]
        project = task["projectName"]
        refactor_type = task["type"]

        print(f"\n[{i+1}/{len(sample)}] {project} - {refactor_type}")
        print(f"  Task: {task_id[:50]}...")

        try:
            result = agent.run_single_task(task_id, model=model, verbose=True)

            status = "PASSED" if result.get("passed") else "FAILED"
            print(f"  Result: {status}")

            if result.get("passed"):
                passed += 1

            results["tasks"].append({
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "passed": result.get("passed", False),
                "turns_used": result.get("turns_used", 0),
                "error": result.get("error"),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results["tasks"].append({
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "passed": False,
                "error": str(e),
            })

        # Save after each task
        results["passed"] = passed
        results["total"] = len(results["tasks"])
        results["updated_at"] = datetime.now().isoformat()
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Final summary
    print("\n" + "=" * 60)
    print(f"Baseline Sample Complete")
    print(f"Passed: {passed}/{len(sample)} ({passed/len(sample)*100:.1f}%)")
    print("=" * 60)

    # By type
    print("\nBy Type:")
    by_type = defaultdict(lambda: {"passed": 0, "total": 0})
    for t in results["tasks"]:
        by_type[t["type"]]["total"] += 1
        if t["passed"]:
            by_type[t["type"]]["passed"] += 1

    for refactor_type, stats in sorted(by_type.items()):
        print(f"  {refactor_type}: {stats['passed']}/{stats['total']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-type", type=int, default=2, help="Tasks per refactoring type")
    parser.add_argument("--model", default="opus", help="Model to use")
    args = parser.parse_args()

    run_baseline_sample(n_per_type=args.n_per_type, model=args.model)
