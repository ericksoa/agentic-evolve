#!/usr/bin/env python3
"""
RALPH Loop Runner for RefactorBench.

Runs iterative agent retry loops with filesystem-based memory on a
SINGLE task at a time.

RALPH pattern:
- Agent runs, does work, exits
- Work persists on the filesystem (working directory retains all changes)
- Fresh agent starts, reads progress.json from the working directory, continues
- Repeats until tests pass or max iterations

Usage:
    python3 ralph_runner.py --repo django_refactor --task add-log-parameter-get-resolver
    python3 ralph_runner.py --repo django_refactor --task add-log-parameter-get-resolver --chains 2 --iterations 3
    python3 ralph_runner.py --repo django_refactor --task add-log-parameter-get-resolver --verbose
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from refactor_agent import get_task_info, run_test, setup_workdir

import ralph_prompt_builder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RALPHConfig:
    """Configuration for a RALPH loop run on a single task."""

    max_chains: int = 5          # Independent chains for this task
    max_iterations: int = 10     # Iterations per chain
    model: str = "opus"          # Model for claude -p
    max_turns: int = 15          # Max turns per agent invocation
    timeout: int = 600           # Timeout per agent invocation (seconds)
    workdir_root: str = "ralph_workdirs"
    results_dir: str = "ralph_results"
    verbose: bool = False


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(workdir: Path) -> dict | None:
    """Load progress.json from a working directory."""
    progress_file = workdir / "progress.json"
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_progress(workdir: Path, progress: dict) -> None:
    """Save progress.json to a working directory."""
    progress_file = workdir / "progress.json"
    progress_file.write_text(json.dumps(progress, indent=2))


def init_progress(task_id: str, chain_id: int) -> dict:
    """Create initial progress.json structure."""
    return {
        "task_id": task_id,
        "chain_id": chain_id,
        "iteration": 0,
        "status": "in_progress",
        "test_results": [],
        "best_score": 0,
        "failed_approaches": [],
    }


# ---------------------------------------------------------------------------
# Test result parsing
# ---------------------------------------------------------------------------

def compute_granular_fitness(test_result: dict) -> dict:
    """Parse pytest -v output to count individual test passes/failures.

    Returns dict with:
        tests_passed: int
        tests_total: int
        failing_tests: list[str]
        error_summary: str
    """
    stdout = test_result.get("stdout", "")

    # Parse individual test results from pytest -v output
    # Pattern: scripts/test_file.py::test_name PASSED/FAILED
    test_lines = re.findall(
        r'([\w/\.\-]+::[\w\[\]\-]+)\s+(PASSED|FAILED|ERROR)', stdout
    )

    if test_lines:
        tests_total = len(test_lines)
        tests_passed = sum(1 for _, status in test_lines if status == "PASSED")
        failing_tests = [name for name, status in test_lines if status != "PASSED"]
    else:
        # Fall back to summary line: "X passed, Y failed"
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        failed_match = re.search(r'(\d+)\s+failed', stdout)
        error_match = re.search(r'(\d+)\s+error', stdout)

        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_errored = int(error_match.group(1)) if error_match else 0
        tests_total = tests_passed + tests_failed + tests_errored
        failing_tests = []

        if tests_total == 0 and not test_result.get("passed"):
            tests_total = 1  # at least 1 test existed

    # Extract error summary
    error_summary = ""
    stderr = test_result.get("stderr", "")
    if "ERRORS" in stdout or "ERROR" in stdout:
        error_block = re.search(r'(?:ERRORS?|ERROR)\s*[-=]+\s*\n(.+?)(?:\n[-=]|\Z)',
                                stdout, re.DOTALL)
        if error_block:
            error_summary = error_block.group(1).strip()[:300]
    elif "AssertionError" in stdout or "AssertionError" in stderr:
        error_summary = "Assertion failure in tests"
    elif "ModuleNotFoundError" in stdout or "ModuleNotFoundError" in stderr:
        mod_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'",
                              stdout + stderr)
        error_summary = f"Missing module: {mod_match.group(1)}" if mod_match else "Missing module"
    elif "ImportError" in stdout or "ImportError" in stderr:
        error_summary = "Import error"
    elif not test_result.get("passed"):
        error_summary = "Tests did not pass"

    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "failing_tests": failing_tests,
        "error_summary": error_summary,
    }


# ---------------------------------------------------------------------------
# Core RALPH execution — single task only
# ---------------------------------------------------------------------------

class RALPHRunner:
    """Runs RALPH loops on a single RefactorBench task."""

    def __init__(self, config: RALPHConfig, repo: str, task: str):
        self.config = config
        self.repo = repo
        self.task = task
        self.task_id = f"{repo}/{task}"
        self.project_dir = Path(__file__).parent.resolve()
        self.workdir_root = self.project_dir / config.workdir_root
        self.results_dir = self.project_dir / config.results_dir
        self.workdir_root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict:
        """Run RALPH chains on this task. Exits early on success."""
        print(f"\n[RALPH] {self.task_id}")
        print(f"  Config: {self.config.max_chains} chains x "
              f"{self.config.max_iterations} iterations")
        print(f"  Model: {self.config.model}")
        print("-" * 60)

        start_time = time.time()
        best_result = None
        best_score = 0

        for chain_id in range(self.config.max_chains):
            # Check if a previous chain already solved it
            task_results_dir = self.results_dir / f"{self.repo}__{self.task}"
            chain_result_file = task_results_dir / f"chain_{chain_id}.json"
            if chain_result_file.exists():
                try:
                    existing = json.loads(chain_result_file.read_text())
                    if existing.get("status") == "solved":
                        print(f"  Chain {chain_id}: already solved (from previous run)")
                        elapsed = time.time() - start_time
                        self._print_result("SOLVED", chain_id, elapsed,
                                           existing.get("best_score", 0))
                        return {"task_id": self.task_id, "status": "solved",
                                "chain_id": chain_id}
                except (json.JSONDecodeError, OSError):
                    pass

            print(f"\n  Chain {chain_id}/{self.config.max_chains - 1}:")
            result = await self._run_chain(chain_id)

            # Save chain result
            task_results_dir.mkdir(parents=True, exist_ok=True)
            chain_result_file.write_text(json.dumps(result, indent=2))

            if result.get("status") == "solved":
                elapsed = time.time() - start_time
                self._print_result("SOLVED", chain_id, elapsed,
                                   result.get("best_score", 0),
                                   result.get("solved_iteration"))
                return {"task_id": self.task_id, "status": "solved",
                        "chain_id": chain_id}

            # Track best score across chains
            chain_best = result.get("best_score", 0)
            if chain_best > best_score:
                best_score = chain_best
                best_result = result

        # All chains exhausted
        elapsed = time.time() - start_time
        self._print_result("FAILED", -1, elapsed, best_score)
        return {"task_id": self.task_id, "status": "failed",
                "best_score": best_score}

    async def _run_chain(self, chain_id: int) -> dict:
        """Run a single chain of RALPH iterations."""
        chain_workdir = (self.workdir_root / f"{self.repo}__{self.task}"
                         / f"chain_{chain_id}")

        # Determine starting iteration (for resumability)
        start_iteration = 1
        progress = None

        if chain_workdir.exists():
            repo_dir = chain_workdir / self.repo
            if repo_dir.exists():
                progress = load_progress(repo_dir)
                if progress:
                    if progress.get("status") == "solved":
                        return {"status": "solved", "chain_id": chain_id,
                                "best_score": progress.get("best_score", 0),
                                "solved_iteration": progress.get("iteration")}
                    start_iteration = progress.get("iteration", 0) + 1
                    if start_iteration > self.config.max_iterations:
                        return {"status": "exhausted", "chain_id": chain_id,
                                "best_score": progress.get("best_score", 0)}

        # Setup workdir for iteration 1 (fresh copy of repo)
        if start_iteration == 1:
            if chain_workdir.exists():
                shutil.rmtree(chain_workdir)
            chain_workdir.mkdir(parents=True, exist_ok=True)
            workdir = setup_workdir(self.repo, chain_workdir)
            progress = init_progress(self.task_id, chain_id)
        else:
            workdir = chain_workdir / self.repo

        task_info = get_task_info(self.repo, self.task)

        for iteration in range(start_iteration, self.config.max_iterations + 1):
            print(f"    Iteration {iteration}/{self.config.max_iterations}...",
                  end=" ", flush=True)

            # Build iteration-aware prompt
            prompt = ralph_prompt_builder.build_prompt(
                task_info=task_info,
                workdir=workdir,
                progress=progress if iteration > 1 else None,
                iteration=iteration,
                max_iterations=self.config.max_iterations,
            )

            # Run the agent
            await self._run_agent(prompt, workdir)

            # Run tests
            test_result = run_test(task_info["test_file"], workdir,
                                   task_info["repo_name"])

            # Parse granular fitness
            granular = compute_granular_fitness(test_result)

            # Update progress
            progress["iteration"] = iteration
            progress["test_results"].append({
                "iteration": iteration,
                "passed": test_result["passed"],
                "tests_passed": granular["tests_passed"],
                "tests_total": granular["tests_total"],
                "failing_tests": granular["failing_tests"],
                "error_summary": granular["error_summary"],
                "test_stdout": test_result.get("stdout", "")[-2000:],
            })

            if granular["tests_passed"] > progress["best_score"]:
                progress["best_score"] = granular["tests_passed"]

            # Record what was tried (for "DO NOT REPEAT")
            approach_desc = (
                f"Iteration {iteration}: "
                f"{granular['tests_passed']}/{granular['tests_total']} tests passed. "
                f"Failing: {', '.join(granular['failing_tests'][:5])}"
            )
            if granular["error_summary"]:
                approach_desc += f". Error: {granular['error_summary']}"
            progress["failed_approaches"].append(approach_desc)

            # Save progress to workdir
            save_progress(workdir, progress)

            # Check if solved
            if test_result["passed"]:
                progress["status"] = "solved"
                save_progress(workdir, progress)
                print(f"PASSED ({granular['tests_passed']}/{granular['tests_total']})")
                return {
                    "status": "solved",
                    "chain_id": chain_id,
                    "solved_iteration": iteration,
                    "best_score": granular["tests_passed"],
                }

            print(f"{granular['tests_passed']}/{granular['tests_total']} tests")

        # Chain exhausted
        return {
            "status": "exhausted",
            "chain_id": chain_id,
            "best_score": progress.get("best_score", 0),
        }

    async def _run_agent(self, prompt: str, workdir: Path) -> None:
        """Run claude -p as a subprocess."""
        cmd = [
            "claude",
            "-p",
            "--model", self.config.model,
            "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "--output-format", "json",
            "--max-turns", str(self.config.max_turns),
            prompt,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir.resolve()),
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"},
            )
            await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        except Exception as e:
            if self.config.verbose:
                print(f"  Agent error: {e}", file=sys.stderr)

    def _print_result(self, status: str, chain_id: int, elapsed: float,
                      best_score: int, solved_iteration: int | None = None):
        """Print final result for this task."""
        print(f"\n{'=' * 60}")
        print(f"[{status}] {self.task_id}")
        if status == "SOLVED":
            print(f"  Chain: {chain_id}, Iteration: {solved_iteration}")
        else:
            print(f"  Best score: {best_score} tests passing")
        print(f"  Elapsed: {elapsed:.0f}s")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Progress tracking (source of truth: results/progress.json)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RALPH Loop Runner — runs on a SINGLE task"
    )
    parser.add_argument(
        "--repo", required=True,
        help="Repository name (e.g. django_refactor)",
    )
    parser.add_argument(
        "--task", required=True,
        help="Task name (e.g. add-log-parameter-get-resolver)",
    )
    parser.add_argument(
        "--chains", type=int, default=5,
        help="Independent chains (default: 5)",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Max iterations per chain (default: 10)",
    )
    parser.add_argument(
        "--model", default="opus",
        help="Model for claude -p (default: opus)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=15,
        help="Max turns per agent invocation (default: 15)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    config = RALPHConfig(
        max_chains=args.chains,
        max_iterations=args.iterations,
        model=args.model,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    runner = RALPHRunner(config, repo=args.repo, task=args.task)
    result = asyncio.run(runner.run())

    # Update progress.json
    if result.get("status") == "solved":
        chain_id = result.get("chain_id", "?")
        solved_iter = result.get("solved_iteration", "?")
        update_progress(
            args.repo, args.task, passed=True, method="ralph",
            details=f"Solved chain {chain_id}, iteration {solved_iter}",
        )
    else:
        best = result.get("best_score", 0)
        update_progress(
            args.repo, args.task, passed=False, method="ralph_failed",
            details=f"Failed after {args.chains} chains, best score {best}",
        )

    # Print machine-readable result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
