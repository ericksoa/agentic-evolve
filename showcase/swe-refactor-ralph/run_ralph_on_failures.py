#!/usr/bin/env python3
"""
Run RALPH on failed tasks from the JDK 11+ sample.

Tracks iterations required and updates results accordingly.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ralph_runner import RALPHRunner, RALPHConfig

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
JDK11_RESULTS = RESULTS_DIR / "jdk11_sample.json"
RALPH_SUMMARY = RESULTS_DIR / "ralph_summary.json"


async def run_ralph_on_failures(
    chains: int = 5,
    iterations: int = 10,
    model: str = "opus",
):
    """Run RALPH on all failed tasks from the JDK 11+ sample."""

    # Load JDK 11+ sample results
    if not JDK11_RESULTS.exists():
        print(f"ERROR: {JDK11_RESULTS} not found")
        print("Run run_jdk11_sample.py first to generate baseline results")
        return

    sample_results = json.loads(JDK11_RESULTS.read_text())
    tasks = sample_results.get("tasks", [])

    # Find failed tasks
    failed_tasks = [t for t in tasks if not t.get("passed")]

    if not failed_tasks:
        print("All tasks in JDK 11+ sample passed! Nothing to run RALPH on.")
        return

    print(f"\n{'=' * 60}")
    print(f"Running RALPH on {len(failed_tasks)} failed task(s)")
    print(f"Config: {chains} chains x {iterations} iterations, model={model}")
    print(f"{'=' * 60}")

    config = RALPHConfig(
        max_chains=chains,
        max_iterations=iterations,
        model=model,
        verbose=True,
    )

    ralph_results = {
        "started_at": datetime.now().isoformat(),
        "config": {
            "chains": chains,
            "iterations": iterations,
            "model": model,
        },
        "tasks": [],
    }

    solved_count = 0

    for task_info in failed_tasks:
        task_id = task_info["task_id"]
        project = task_info["project"]
        refactor_type = task_info["type"]

        print(f"\n[RALPH] {project} - {refactor_type}")
        print(f"  Task ID: {task_id[:40]}...")

        try:
            runner = RALPHRunner(config, task_id=task_id)
            result = await runner.run()

            if result.get("status") == "solved":
                solved_count += 1
                print(f"\n  ✅ SOLVED in iteration {result.get('solved_iteration')}")

                ralph_results["tasks"].append({
                    "task_id": task_id,
                    "project": project,
                    "type": refactor_type,
                    "status": "solved",
                    "chain_id": result.get("chain_id"),
                    "solved_iteration": result.get("solved_iteration"),
                })
            else:
                print(f"\n  ❌ FAILED after {chains} chains")

                ralph_results["tasks"].append({
                    "task_id": task_id,
                    "project": project,
                    "type": refactor_type,
                    "status": "failed",
                    "chains_tried": chains,
                })

        except Exception as e:
            print(f"\n  ERROR: {e}")
            ralph_results["tasks"].append({
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "status": "error",
                "error": str(e),
            })

    # Final summary
    ralph_results["completed_at"] = datetime.now().isoformat()
    ralph_results["solved"] = solved_count
    ralph_results["total"] = len(failed_tasks)

    print(f"\n{'=' * 60}")
    print(f"RALPH Summary: {solved_count}/{len(failed_tasks)} previously failed tasks now solved")
    print(f"{'=' * 60}")

    # Calculate overall results including baseline passes
    baseline_passed = sum(1 for t in tasks if t.get("passed"))
    total_tasks = len(tasks)
    total_passed = baseline_passed + solved_count

    print(f"\nOverall Results (baseline + RALPH):")
    print(f"  Baseline passed: {baseline_passed}/{total_tasks}")
    print(f"  RALPH solved: {solved_count}/{len(failed_tasks)}")
    print(f"  Total: {total_passed}/{total_tasks} ({total_passed/total_tasks*100:.1f}%)")

    # Save RALPH results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RALPH_SUMMARY.write_text(json.dumps(ralph_results, indent=2))
    print(f"\nResults saved to: {RALPH_SUMMARY}")

    # Update overall progress tracking
    update_combined_results(tasks, ralph_results)


def update_combined_results(baseline_tasks: list, ralph_results: dict):
    """Create combined results file with iteration tracking."""
    combined = {
        "benchmark": "SWE-Refactor (JDK 11+ sample)",
        "updated_at": datetime.now().isoformat(),
        "tasks": [],
    }

    # Map RALPH results by task_id
    ralph_by_id = {t["task_id"]: t for t in ralph_results.get("tasks", [])}

    for task in baseline_tasks:
        task_id = task["task_id"]

        if task.get("passed"):
            # Baseline pass
            combined["tasks"].append({
                "task_id": task_id,
                "project": task["project"],
                "type": task["type"],
                "passed": True,
                "method": "baseline",
                "iterations": 1,  # Single-shot
            })
        elif task_id in ralph_by_id:
            # RALPH attempted
            ralph = ralph_by_id[task_id]
            if ralph.get("status") == "solved":
                combined["tasks"].append({
                    "task_id": task_id,
                    "project": task["project"],
                    "type": task["type"],
                    "passed": True,
                    "method": "ralph",
                    "iterations": ralph.get("solved_iteration", 0),
                    "chain_id": ralph.get("chain_id", 0),
                })
            else:
                combined["tasks"].append({
                    "task_id": task_id,
                    "project": task["project"],
                    "type": task["type"],
                    "passed": False,
                    "method": "ralph_failed",
                    "chains_tried": ralph.get("chains_tried", 5),
                })
        else:
            # Baseline fail, no RALPH
            combined["tasks"].append({
                "task_id": task_id,
                "project": task["project"],
                "type": task["type"],
                "passed": False,
                "method": "baseline_failed",
            })

    # Summary stats
    passed = sum(1 for t in combined["tasks"] if t.get("passed"))
    total = len(combined["tasks"])
    combined["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": f"{passed/total*100:.1f}%",
    }

    # By method
    baseline_pass = sum(1 for t in combined["tasks"]
                        if t.get("method") == "baseline" and t.get("passed"))
    ralph_pass = sum(1 for t in combined["tasks"]
                     if t.get("method") == "ralph" and t.get("passed"))
    combined["summary"]["by_method"] = {
        "baseline_passed": baseline_pass,
        "ralph_passed": ralph_pass,
    }

    # Iteration distribution for RALPH solves
    ralph_iterations = [t.get("iterations", 0) for t in combined["tasks"]
                        if t.get("method") == "ralph" and t.get("passed")]
    if ralph_iterations:
        combined["summary"]["ralph_iteration_distribution"] = {
            "min": min(ralph_iterations),
            "max": max(ralph_iterations),
            "mean": sum(ralph_iterations) / len(ralph_iterations),
        }

    combined_file = RESULTS_DIR / "combined_results.json"
    combined_file.write_text(json.dumps(combined, indent=2))
    print(f"Combined results saved to: {combined_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RALPH on failed JDK 11+ tasks")
    parser.add_argument("--chains", type=int, default=5, help="Number of chains")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations per chain")
    parser.add_argument("--model", default="opus", help="Model to use")
    args = parser.parse_args()

    asyncio.run(run_ralph_on_failures(
        chains=args.chains,
        iterations=args.iterations,
        model=args.model,
    ))
