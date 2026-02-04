#!/usr/bin/env python3
"""
Coordinator for RefactorBench task solving.

Runs RALPH first, then falls back to evolution if RALPH fails.
Both scripts stay independent — this script just orchestrates them.

Usage:
    python3 run_task.py --repo scrapy_refactor --task new-downloadermiddlewares-utils
    python3 run_task.py --repo scrapy_refactor --task new-downloadermiddlewares-utils --verbose
    python3 run_task.py --repo scrapy_refactor --task new-downloadermiddlewares-utils --skip-ralph
    python3 run_task.py --repo scrapy_refactor --task new-downloadermiddlewares-utils --skip-evolution
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.resolve()
PROGRESS_FILE = PROJECT_DIR / "results" / "progress.json"


def run_ralph(repo: str, task: str, verbose: bool) -> dict:
    """Run ralph_runner.py and return parsed result."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "ralph_runner.py"),
        "--repo", repo,
        "--task", task,
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"\n{'='*60}")
    print(f"[PHASE 1] RALPH: {repo}/{task}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=3600)

    # RALPH prints JSON as its last output line and updates progress.json.
    # Read progress.json to check outcome.
    return read_task_status(repo, task)


def run_evolution(repo: str, task: str) -> dict:
    """Run evolve_task.py and return parsed result."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "evolve_task.py"),
        "--repo", repo,
        "--task", task,
    ]

    print(f"\n{'='*60}")
    print(f"[PHASE 2] Evolution: {repo}/{task}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=7200)

    return read_task_status(repo, task)


def read_task_status(repo: str, task: str) -> dict:
    """Read the current status of a task from progress.json."""
    if not PROGRESS_FILE.exists():
        return {"current_passed": False, "method": None}

    progress = json.loads(PROGRESS_FILE.read_text())
    for t in progress["tasks"]:
        if t["repo"] == repo and t["task"] == task:
            return t

    return {"current_passed": False, "method": None}


def read_score() -> tuple[int, int]:
    """Read current score from progress.json."""
    if not PROGRESS_FILE.exists():
        return 0, 0
    progress = json.loads(PROGRESS_FILE.read_text())
    return progress.get("current_passed", 0), progress.get("current_total", 0)


def main():
    parser = argparse.ArgumentParser(
        description="Run RALPH then evolution on a RefactorBench task"
    )
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-ralph", action="store_true",
                        help="Skip RALPH, go straight to evolution")
    parser.add_argument("--skip-evolution", action="store_true",
                        help="Skip evolution if RALPH fails")
    args = parser.parse_args()

    task_id = f"{args.repo}/{args.task}"
    solved = False

    # Phase 1: RALPH
    if not args.skip_ralph:
        try:
            status = run_ralph(args.repo, args.task, args.verbose)
            if status.get("current_passed"):
                solved = True
                print(f"\n[SOLVED by RALPH] {task_id}")
        except subprocess.TimeoutExpired:
            print(f"\n[RALPH TIMED OUT] {task_id}")
        except Exception as e:
            print(f"\n[RALPH ERROR] {task_id}: {e}")

    # Phase 2: Evolution (only if RALPH didn't solve it)
    if not solved and not args.skip_evolution:
        try:
            status = run_evolution(args.repo, args.task)
            if status.get("current_passed"):
                solved = True
                print(f"\n[SOLVED by Evolution] {task_id}")
        except subprocess.TimeoutExpired:
            print(f"\n[EVOLUTION TIMED OUT] {task_id}")
        except Exception as e:
            print(f"\n[EVOLUTION ERROR] {task_id}: {e}")

    # Final report
    passed, total = read_score()
    status_str = "SOLVED" if solved else "FAILED"

    print(f"\n{'='*60}")
    print(f"[{status_str}] {task_id}")
    print(f"Current score: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
