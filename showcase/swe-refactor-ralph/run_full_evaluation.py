#!/usr/bin/env python3
"""
Run full SWE-Refactor evaluation on JDK 11+ compatible tasks.

Runs baseline evaluation first, then RALPH on failures.
Results are saved incrementally after each task.

Usage:
    python3 run_full_evaluation.py                    # Run all JDK 11+ tasks
    python3 run_full_evaluation.py --n-per-type 3    # Run 3 tasks per type (18 total)
    python3 run_full_evaluation.py --resume          # Resume from last checkpoint
"""

import asyncio
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import swe_refactor_agent as agent
from ralph_runner import RALPHRunner, RALPHConfig

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_FILE = RESULTS_DIR / "full_evaluation_checkpoint.json"


def select_jdk11_tasks(n_per_type: int = 0) -> list[dict]:
    """Select JDK 11+ compatible tasks.

    Args:
        n_per_type: If > 0, limit to N tasks per refactoring type.
                   If 0, return all JDK 11+ tasks.
    """
    tasks = agent.load_tasks()

    # Filter to JDK 11+
    compatible = [t for t in tasks if str(t.get("compileJDK", "")) in ["11", "17", "21"]]

    if n_per_type == 0:
        return compatible

    # Group by type and sample
    by_type = defaultdict(list)
    for t in compatible:
        by_type[t["type"]].append(t)

    # Prefer smaller projects for faster builds
    small_projects = {"gson", "mockito", "shiro", "junit4", "zxing", "jadx"}

    sample = []
    for refactor_type, type_tasks in by_type.items():
        # Sort by project size preference, then shuffle within size groups
        type_tasks.sort(key=lambda t: (t["projectName"] not in small_projects, t["projectName"]))

        # Take up to n_per_type, preferring different projects
        seen_projects = set()
        for t in type_tasks:
            if len([s for s in sample if s["type"] == refactor_type]) >= n_per_type:
                break
            if t["projectName"] not in seen_projects or len(seen_projects) >= n_per_type:
                sample.append(t)
                seen_projects.add(t["projectName"])

    return sample


def load_checkpoint() -> dict:
    """Load checkpoint from previous run."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "completed_tasks": {},
        "baseline_results": [],
        "ralph_results": [],
        "started_at": datetime.now().isoformat(),
    }


def save_checkpoint(checkpoint: dict):
    """Save checkpoint for resumability."""
    checkpoint["updated_at"] = datetime.now().isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


async def run_evaluation(
    n_per_type: int = 0,
    model: str = "opus",
    ralph_chains: int = 5,
    ralph_iterations: int = 10,
    resume: bool = False,
):
    """Run full evaluation: baseline + RALPH on failures."""

    # Load or initialize checkpoint
    if resume:
        checkpoint = load_checkpoint()
        print(f"Resuming from checkpoint: {len(checkpoint['completed_tasks'])} tasks already done")
    else:
        checkpoint = load_checkpoint()
        if checkpoint["completed_tasks"]:
            print(f"Warning: Overwriting previous checkpoint with {len(checkpoint['completed_tasks'])} tasks")
        checkpoint = {
            "completed_tasks": {},
            "baseline_results": [],
            "ralph_results": [],
            "started_at": datetime.now().isoformat(),
        }

    # Select tasks
    tasks = select_jdk11_tasks(n_per_type)
    print(f"\n{'=' * 60}")
    print(f"SWE-Refactor Full Evaluation")
    print(f"Tasks: {len(tasks)} (JDK 11+ only)")
    print(f"Model: {model}")
    print(f"RALPH config: {ralph_chains} chains x {ralph_iterations} iterations")
    print(f"{'=' * 60}\n")

    # Phase 1: Baseline evaluation
    print("=" * 60)
    print("PHASE 1: Baseline Evaluation")
    print("=" * 60)

    baseline_passed = 0
    baseline_failed = []
    env_issues = []

    for i, task in enumerate(tasks):
        task_id = task["uniqueId"]
        project = task["projectName"]
        refactor_type = task["type"]
        jdk = task.get("compileJDK", "?")

        # Skip if already completed
        if task_id in checkpoint["completed_tasks"]:
            result = checkpoint["completed_tasks"][task_id]
            if result.get("env_issue"):
                env_issues.append(task)
                print(f"[{i+1}/{len(tasks)}] {project} - {refactor_type} (JDK {jdk}): "
                      f"ENV_ISSUE (cached)")
            elif result.get("baseline_passed"):
                baseline_passed += 1
                print(f"[{i+1}/{len(tasks)}] {project} - {refactor_type} (JDK {jdk}): "
                      f"PASS (cached)")
            else:
                baseline_failed.append(task)
                print(f"[{i+1}/{len(tasks)}] {project} - {refactor_type} (JDK {jdk}): "
                      f"FAIL (cached)")
            continue

        print(f"\n[{i+1}/{len(tasks)}] {project} - {refactor_type} (JDK {jdk})")
        print(f"  Task: {task_id[:40]}...")

        try:
            result = agent.run_single_task(task_id, model=model, verbose=True, check_env=True)

            # Check for environment issues
            if result.get("env_issue"):
                print(f"  Result: ENV_ISSUE - {result.get('error', 'unknown')}")
                checkpoint["completed_tasks"][task_id] = {
                    "task_id": task_id,
                    "project": project,
                    "type": refactor_type,
                    "jdk": jdk,
                    "env_issue": True,
                    "baseline_passed": False,
                    "baseline_error": result.get("error"),
                }
                env_issues.append(task)
                save_checkpoint(checkpoint)
                continue

            passed = result.get("passed", False)

            checkpoint["completed_tasks"][task_id] = {
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "jdk": jdk,
                "env_issue": False,
                "baseline_passed": passed,
                "baseline_turns": result.get("turns_used", 0),
                "baseline_error": result.get("error"),
            }

            checkpoint["baseline_results"].append({
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "jdk": jdk,
                "passed": passed,
                "turns_used": result.get("turns_used", 0),
                "error": result.get("error"),
            })

            if passed:
                baseline_passed += 1
                print(f"  Result: PASSED")
            else:
                baseline_failed.append(task)
                print(f"  Result: FAILED")

        except Exception as e:
            print(f"  ERROR: {e}")
            checkpoint["completed_tasks"][task_id] = {
                "task_id": task_id,
                "project": project,
                "type": refactor_type,
                "jdk": jdk,
                "env_issue": False,
                "baseline_passed": False,
                "baseline_error": str(e),
            }
            baseline_failed.append(task)

        # Save checkpoint after each task
        save_checkpoint(checkpoint)

    valid_tasks = len(tasks) - len(env_issues)
    print(f"\n{'=' * 60}")
    print(f"Baseline Complete:")
    print(f"  Passed: {baseline_passed}/{valid_tasks} ({baseline_passed/valid_tasks*100:.1f}% of valid tasks)")
    print(f"  Failed: {len(baseline_failed)}/{valid_tasks}")
    print(f"  Env Issues (skipped): {len(env_issues)}")
    print(f"{'=' * 60}")

    # Phase 2: RALPH on failures
    ralph_solved = 0
    if baseline_failed:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: RALPH on {len(baseline_failed)} Failed Tasks")
        print(f"{'=' * 60}")

        ralph_config = RALPHConfig(
            max_chains=ralph_chains,
            max_iterations=ralph_iterations,
            model=model,
            verbose=True,
        )

        ralph_solved = 0

        for i, task in enumerate(baseline_failed):
            task_id = task["uniqueId"]
            project = task["projectName"]
            refactor_type = task["type"]

            # Skip if RALPH already attempted
            existing = checkpoint["completed_tasks"].get(task_id, {})
            if existing.get("ralph_attempted"):
                if existing.get("ralph_solved"):
                    ralph_solved += 1
                print(f"[{i+1}/{len(baseline_failed)}] {project} - {refactor_type}: "
                      f"{'SOLVED' if existing.get('ralph_solved') else 'FAILED'} (cached)")
                continue

            print(f"\n[{i+1}/{len(baseline_failed)}] RALPH: {project} - {refactor_type}")

            try:
                runner = RALPHRunner(ralph_config, task_id=task_id)
                result = await runner.run()

                solved = result.get("status") == "solved"

                # Update checkpoint
                checkpoint["completed_tasks"][task_id]["ralph_attempted"] = True
                checkpoint["completed_tasks"][task_id]["ralph_solved"] = solved
                if solved:
                    checkpoint["completed_tasks"][task_id]["ralph_chain"] = result.get("chain_id")
                    checkpoint["completed_tasks"][task_id]["ralph_iteration"] = result.get("solved_iteration")
                    ralph_solved += 1

                checkpoint["ralph_results"].append({
                    "task_id": task_id,
                    "project": project,
                    "type": refactor_type,
                    "status": result.get("status"),
                    "chain_id": result.get("chain_id"),
                    "solved_iteration": result.get("solved_iteration"),
                })

                print(f"  Result: {'SOLVED' if solved else 'FAILED'}"
                      + (f" (iteration {result.get('solved_iteration')})" if solved else ""))

            except Exception as e:
                print(f"  ERROR: {e}")
                checkpoint["completed_tasks"][task_id]["ralph_attempted"] = True
                checkpoint["completed_tasks"][task_id]["ralph_error"] = str(e)

            # Save checkpoint after each task
            save_checkpoint(checkpoint)

        print(f"\n{'=' * 60}")
        print(f"RALPH Complete: {ralph_solved}/{len(baseline_failed)} failed tasks solved")
        print(f"{'=' * 60}")

    # Final summary
    ralph_solved = ralph_solved if baseline_failed else 0
    total_passed = baseline_passed + ralph_solved
    valid_tasks = len(tasks) - len(env_issues)

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Valid Tasks: {valid_tasks}/{len(tasks)} ({len(env_issues)} had env issues)")
    print(f"Total Passed: {total_passed}/{valid_tasks} ({total_passed/valid_tasks*100:.1f}% of valid tasks)")
    print(f"  Baseline: {baseline_passed}")
    if baseline_failed:
        print(f"  RALPH solved: {ralph_solved}/{len(baseline_failed)}")
    print(f"\nPublished SOTA: 41.58% (DeepSeek-V3)")
    print(f"{'=' * 60}")

    # Save final results
    final_results = {
        "benchmark": "SWE-Refactor (JDK 11+)",
        "completed_at": datetime.now().isoformat(),
        "config": {
            "n_per_type": n_per_type,
            "model": model,
            "ralph_chains": ralph_chains,
            "ralph_iterations": ralph_iterations,
        },
        "summary": {
            "total_tasks": len(tasks),
            "valid_tasks": valid_tasks,
            "env_issues": len(env_issues),
            "baseline_passed": baseline_passed,
            "ralph_solved": ralph_solved,
            "final_passed": total_passed,
            "pass_rate_valid": f"{total_passed/valid_tasks*100:.1f}%",
            "pass_rate_total": f"{total_passed/len(tasks)*100:.1f}%",
        },
        "tasks": list(checkpoint["completed_tasks"].values()),
    }

    final_file = RESULTS_DIR / "full_evaluation_results.json"
    final_file.write_text(json.dumps(final_results, indent=2))
    print(f"\nResults saved to: {final_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full SWE-Refactor evaluation")
    parser.add_argument("--n-per-type", type=int, default=0,
                        help="Tasks per refactoring type (0 = all JDK 11+ tasks)")
    parser.add_argument("--model", default="opus", help="Model to use")
    parser.add_argument("--ralph-chains", type=int, default=5, help="RALPH chains")
    parser.add_argument("--ralph-iterations", type=int, default=10, help="RALPH iterations per chain")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        n_per_type=args.n_per_type,
        model=args.model,
        ralph_chains=args.ralph_chains,
        ralph_iterations=args.ralph_iterations,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
