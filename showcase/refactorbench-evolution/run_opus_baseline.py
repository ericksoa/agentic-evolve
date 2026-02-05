"""
Run the 53 baseline tasks with Opus 4.5 to validate results.

This re-runs all tasks that passed in the original Sonnet baseline,
using Opus 4.5 instead, to address the model confound and establish
an Opus single-shot baseline.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import refactor_agent

RESULTS_FILE = Path(__file__).parent / "results" / "opus_baseline.json"
PROGRESS_FILE = Path(__file__).parent / "results" / "progress.json"
STRATEGY_FILE = Path(__file__).parent / "baseline_strategy.json"
WORKDIR_ROOT = Path(__file__).parent / "opus_baseline_workdirs"

MODEL = "opus"


def load_baseline_tasks() -> list[dict]:
    """Extract the 53 baseline-passed tasks from progress.json."""
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    tasks = []
    for t in progress["tasks"]:
        if t["method"] == "baseline" and t["baseline_passed"]:
            tasks.append({"repo": t["repo"], "task": t["task"]})
    return tasks


def load_existing_results() -> dict:
    """Load any existing results to allow resuming."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"model": MODEL, "tasks": [], "started_at": datetime.now(timezone.utc).isoformat()}


def save_results(results: dict):
    """Save results to disk."""
    results["updated_at"] = datetime.now(timezone.utc).isoformat()
    passed = sum(1 for t in results["tasks"] if t["passed"])
    total = len(results["tasks"])
    results["passed"] = passed
    results["total"] = total
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def already_completed(results: dict, repo: str, task: str) -> bool:
    """Check if a task was already completed in a previous run."""
    for t in results["tasks"]:
        if t["repo"] == repo and t["task"] == task:
            return True
    return False


def main():
    tasks = load_baseline_tasks()
    print(f"Found {len(tasks)} baseline tasks to validate with Opus 4.5")

    strategy = refactor_agent.load_strategy(str(STRATEGY_FILE))
    results = load_existing_results()

    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)

    completed = sum(1 for t in results["tasks"] if already_completed(results, t["repo"], t["task"]))
    skipped = 0

    for i, task_entry in enumerate(tasks):
        repo = task_entry["repo"]
        task = task_entry["task"]

        if already_completed(results, repo, task):
            skipped += 1
            continue

        idx = i + 1
        passed_so_far = sum(1 for t in results["tasks"] if t["passed"])
        done_so_far = len(results["tasks"])
        print(f"\n[{idx}/{len(tasks)}] {repo}/{task}  (done: {done_so_far}, passed: {passed_so_far})")

        try:
            task_info = refactor_agent.get_task_info(repo, task)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            results["tasks"].append({
                "repo": repo,
                "task": task,
                "passed": False,
                "error": str(e),
                "turns_used": 0,
            })
            save_results(results)
            continue

        # Setup fresh working directory
        workdir = refactor_agent.setup_workdir(repo, WORKDIR_ROOT)

        # Run agent with Opus
        result = refactor_agent.run_agent(
            strategy=strategy,
            task_info=task_info,
            workdir=workdir,
            model=MODEL,
            verbose=True,
        )

        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  Result: {status} (turns: {result.get('turns_used', '?')})")

        results["tasks"].append({
            "repo": repo,
            "task": task,
            "passed": result["passed"],
            "turns_used": result.get("turns_used", 0),
            "error": result.get("error"),
        })

        save_results(results)

    # Final summary
    passed = sum(1 for t in results["tasks"] if t["passed"])
    total = len(results["tasks"])
    print(f"\n{'='*60}")
    print(f"Opus Baseline Validation Complete")
    print(f"Passed: {passed}/{total}")
    print(f"{'='*60}")

    # Show failures if any
    failures = [t for t in results["tasks"] if not t["passed"]]
    if failures:
        print(f"\nFailed tasks ({len(failures)}):")
        for f in failures:
            print(f"  - {f['repo']}/{f['task']}")
    else:
        print("\nAll tasks passed!")


if __name__ == "__main__":
    main()
