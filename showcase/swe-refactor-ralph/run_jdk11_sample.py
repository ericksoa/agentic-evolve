#!/usr/bin/env python3
"""
Run a sample of SWE-Refactor tasks using only JDK 11+ compatible projects.

Avoids JDK 8 tasks which can't run on ARM Macs.
"""

import json
import sys
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import swe_refactor_agent as agent

RESULTS_FILE = Path(__file__).parent / "results" / "jdk11_sample.json"


def select_jdk11_sample(n_per_type: int = 1) -> list[dict]:
    """Select tasks that use JDK 11, 17, or 21."""
    tasks = agent.load_tasks()

    # Filter to JDK 11+
    compatible = [t for t in tasks if str(t.get("compileJDK", "")) in ["11", "17", "21"]]

    # Group by type
    by_type = defaultdict(list)
    for t in compatible:
        by_type[t["type"]].append(t)

    # Sample from each type, preferring smaller projects (faster builds)
    small_projects = {"gson", "mockito", "shiro", "junit4"}
    sample = []

    for refactor_type, type_tasks in by_type.items():
        # Sort by project size preference
        type_tasks.sort(key=lambda t: (t["projectName"] not in small_projects, t["projectName"]))
        sample.append(type_tasks[0])  # Take first (smallest project)

        if n_per_type > 1:
            # Add more from different projects
            seen_projects = {type_tasks[0]["projectName"]}
            for t in type_tasks[1:]:
                if t["projectName"] not in seen_projects:
                    sample.append(t)
                    seen_projects.add(t["projectName"])
                    if len([s for s in sample if s["type"] == refactor_type]) >= n_per_type:
                        break

    return sample


def run_sample(n_per_type: int = 1, model: str = "opus"):
    """Run sample evaluation."""
    sample = select_jdk11_sample(n_per_type)

    print(f"Running JDK 11+ sample: {len(sample)} tasks")
    print(f"Model: {model}")
    print("=" * 60)

    results = {
        "model": model,
        "n_per_type": n_per_type,
        "jdk_filter": "11+",
        "started_at": datetime.now().isoformat(),
        "tasks": [],
    }

    passed = 0
    for i, task in enumerate(sample):
        task_id = task["uniqueId"]
        project = task["projectName"]
        refactor_type = task["type"]
        jdk = task.get("compileJDK", "?")

        print(f"\n[{i+1}/{len(sample)}] {project} (JDK {jdk}) - {refactor_type}")
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
                "jdk": jdk,
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
                "jdk": jdk,
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
    print(f"JDK 11+ Sample Complete")
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
    parser.add_argument("--n-per-type", type=int, default=1, help="Tasks per refactoring type")
    parser.add_argument("--model", default="opus", help="Model to use")
    args = parser.parse_args()

    run_sample(n_per_type=args.n_per_type, model=args.model)
