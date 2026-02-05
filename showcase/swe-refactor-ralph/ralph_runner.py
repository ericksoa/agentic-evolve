#!/usr/bin/env python3
"""
RALPH Loop Runner for SWE-Refactor.

Runs iterative agent retry loops with filesystem-based memory on a
SINGLE task at a time.

RALPH pattern:
- Agent runs, does work, exits
- Work persists on the filesystem (working directory retains all changes)
- Fresh agent starts, reads progress.json from the working directory, continues
- Repeats until compilation passes or max iterations

Engineering notebook records all iteration details for analysis.

Usage:
    python3 ralph_runner.py --task-id <unique_id>
    python3 ralph_runner.py --task-id <unique_id> --chains 2 --iterations 5
    python3 ralph_runner.py --task-id <unique_id> --verbose
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import swe_refactor_agent as agent
import ralph_prompt_builder
from notebook import (
    FileSnapshot, NotebookWriter, parse_stream_json,
    compute_solution_diff, compute_diff_stats,
)

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "ralph_results"
WORKDIRS_ROOT = PROJECT_ROOT / "ralph_workdirs"


@dataclass
class RALPHConfig:
    """Configuration for a RALPH loop run on a single task."""
    max_chains: int = 5
    max_iterations: int = 10
    model: str = "opus"
    max_turns: int = 20
    timeout: int = 900
    verbose: bool = False


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
    """Create initial progress structure."""
    return {
        "task_id": task_id,
        "chain_id": chain_id,
        "iteration": 0,
        "status": "in_progress",
        "compile_results": [],
        "failed_approaches": [],
    }


class RALPHRunner:
    """Runs RALPH loops on a single SWE-Refactor task."""

    def __init__(self, config: RALPHConfig, task_id: str):
        self.config = config
        self.task_id = task_id
        self.task = agent.get_task_by_id(task_id)
        if not self.task:
            raise ValueError(f"Task not found: {task_id}")

        self.project = self.task["projectName"]
        self.refactor_type = self.task["type"]
        self.workdir_root = WORKDIRS_ROOT
        self.results_dir = RESULTS_DIR
        self.workdir_root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict:
        """Run RALPH chains on this task. Exits early on success."""
        print(f"\n[RALPH] {self.project}/{self.task_id[:30]}...")
        print(f"  Type: {self.refactor_type}")
        print(f"  Config: {self.config.max_chains} chains x "
              f"{self.config.max_iterations} iterations")
        print(f"  Model: {self.config.model}")
        print("-" * 60)

        start_time = time.time()
        best_result = None

        for chain_id in range(self.config.max_chains):
            # Check if a previous chain already solved it
            task_results_dir = self.results_dir / f"{self.project}__{self.task_id[:40]}"
            chain_result_file = task_results_dir / f"chain_{chain_id}.json"
            if chain_result_file.exists():
                try:
                    existing = json.loads(chain_result_file.read_text())
                    if existing.get("status") == "solved":
                        print(f"  Chain {chain_id}: already solved (from previous run)")
                        elapsed = time.time() - start_time
                        self._print_result("SOLVED", chain_id, elapsed,
                                           existing.get("solved_iteration"))
                        return {"task_id": self.task_id, "status": "solved",
                                "chain_id": chain_id,
                                "solved_iteration": existing.get("solved_iteration")}
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
                                   result.get("solved_iteration"))
                return {"task_id": self.task_id, "status": "solved",
                        "chain_id": chain_id,
                        "solved_iteration": result.get("solved_iteration")}

            best_result = result

        # All chains exhausted
        elapsed = time.time() - start_time
        self._print_result("FAILED", -1, elapsed, None)
        return {"task_id": self.task_id, "status": "failed",
                "chains_tried": self.config.max_chains}

    async def _run_chain(self, chain_id: int) -> dict:
        """Run a single chain of RALPH iterations."""
        chain_workdir = (self.workdir_root / f"{self.project}__{self.task_id[:30]}"
                         / f"chain_{chain_id}")

        # Determine starting iteration (for resumability)
        start_iteration = 1
        progress = None
        workdir = None

        if chain_workdir.exists():
            # Find the repo subdirectory
            for subdir in chain_workdir.iterdir():
                if subdir.is_dir() and subdir.name == self.project:
                    workdir = subdir
                    break

            if workdir and workdir.exists():
                progress = load_progress(workdir)
                if progress:
                    if progress.get("status") == "solved":
                        return {"status": "solved", "chain_id": chain_id,
                                "solved_iteration": progress.get("iteration")}
                    start_iteration = progress.get("iteration", 0) + 1
                    if start_iteration > self.config.max_iterations:
                        return {"status": "exhausted", "chain_id": chain_id}

        # Setup workdir for iteration 1 (fresh copy of repo)
        if start_iteration == 1:
            if chain_workdir.exists():
                shutil.rmtree(chain_workdir)
            chain_workdir.mkdir(parents=True, exist_ok=True)
            workdir = agent.setup_workdir(self.task, chain_workdir)
            progress = init_progress(self.task_id, chain_id)

        if workdir is None:
            workdir = chain_workdir / self.project

        # --- Engineering notebook setup ---
        notebook_dir = chain_workdir / "notebook"
        notebook = NotebookWriter(
            notebook_dir,
            task_id=self.task_id,
            chain_id=chain_id,
            model=self.config.model,
            task_description=self.task.get("description", ""),
        )
        task_results_dir = self.results_dir / f"{self.project}__{self.task_id[:40]}"

        for iteration in range(start_iteration, self.config.max_iterations + 1):
            print(f"    Iteration {iteration}/{self.config.max_iterations}...",
                  end=" ", flush=True)

            iter_start = time.time()

            # Snapshot file state before agent runs
            pre_snapshot = FileSnapshot(workdir)

            # Build iteration-aware prompt
            prompt = ralph_prompt_builder.build_prompt(
                task=self.task,
                workdir=workdir,
                progress=progress if iteration > 1 else None,
                iteration=iteration,
                max_iterations=self.config.max_iterations,
            )

            # Run the agent (returns stream-json output)
            agent_output = await self._run_agent(prompt, workdir)

            # Parse agent output for notebook
            agent_data = parse_stream_json(agent_output)

            # Snapshot file state after agent runs, compute diffs
            post_snapshot = FileSnapshot(workdir)
            file_diffs = pre_snapshot.diff_against(post_snapshot, workdir, workdir)
            diff_stats = compute_diff_stats(file_diffs)

            # Compile to check result
            compile_result = agent.run_compile(self.task, workdir)

            # Update progress
            progress["iteration"] = iteration
            progress["compile_results"].append({
                "iteration": iteration,
                "passed": compile_result["passed"],
                "stderr": compile_result.get("stderr", "")[:2000],
            })

            # Record what was tried (for "DO NOT REPEAT")
            approach_desc = (
                f"Iteration {iteration}: "
                f"{'compiled successfully' if compile_result['passed'] else 'compilation failed'}"
            )
            if not compile_result["passed"]:
                # Extract key error from stderr
                stderr = compile_result.get("stderr", "")
                if "cannot find symbol" in stderr:
                    approach_desc += " (cannot find symbol error)"
                elif "package does not exist" in stderr:
                    approach_desc += " (package not found error)"
                elif "incompatible types" in stderr:
                    approach_desc += " (type mismatch error)"
                progress["failed_approaches"].append(approach_desc)

            # Save progress to workdir
            save_progress(workdir, progress)

            # --- Notebook entry ---
            notebook.add_entry({
                "iteration": iteration,
                "timestamp_start": datetime.fromtimestamp(iter_start).isoformat(),
                "timestamp_end": datetime.now().isoformat(),
                "wall_clock_seconds": round(time.time() - iter_start, 1),
                "num_turns": agent_data.get("num_turns", 0),
                "cost_usd": agent_data.get("cost_usd", 0),
                "input_tokens": agent_data.get("input_tokens", 0),
                "output_tokens": agent_data.get("output_tokens", 0),
                "agent_reasoning": agent_data.get("reasoning", []),
                "tool_calls": agent_data.get("tool_calls", []),
                "files_read": agent_data.get("files_read", []),
                "files_modified": list(file_diffs.keys()),
                "files_created": agent_data.get("files_created", []),
                "grep_patterns": agent_data.get("grep_patterns", []),
                "compiled": compile_result["passed"],
                "error_summary": compile_result.get("stderr", "")[:500] if not compile_result["passed"] else "",
                "file_diffs": file_diffs,
                "diff_stats": diff_stats,
            })

            # Check if solved
            if compile_result["passed"]:
                progress["status"] = "solved"
                save_progress(workdir, progress)
                print("COMPILED!")

                # Finalize notebook with solution diff
                # Get original repo path
                original_repo = agent.REPOS_DIR / self.project
                solution_diff = ""
                if original_repo.exists():
                    solution_diff = compute_solution_diff(original_repo, workdir)
                notebook.finalize(
                    status="solved",
                    solution_diff=solution_diff,
                    results_dir=task_results_dir,
                )

                return {
                    "status": "solved",
                    "chain_id": chain_id,
                    "solved_iteration": iteration,
                }

            print("FAILED")

        # Chain exhausted — finalize notebook
        original_repo = agent.REPOS_DIR / self.project
        solution_diff = ""
        if original_repo.exists():
            solution_diff = compute_solution_diff(original_repo, workdir)
        notebook.finalize(
            status="exhausted",
            solution_diff=solution_diff,
            results_dir=task_results_dir,
        )

        return {
            "status": "exhausted",
            "chain_id": chain_id,
            "iterations_tried": self.config.max_iterations,
        }

    async def _run_agent(self, prompt: str, workdir: Path) -> str:
        """Run claude -p as a subprocess, return raw stream-json output."""
        cmd = [
            "claude",
            "-p",
            "--model", self.config.model,
            "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "--output-format", "stream-json",
            "--verbose",
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
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout,
            )
            return stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return ""
        except Exception as e:
            if self.config.verbose:
                print(f"  Agent error: {e}", file=sys.stderr)
            return ""

    def _print_result(self, status: str, chain_id: int, elapsed: float,
                      solved_iteration: int | None = None):
        """Print final result for this task."""
        print(f"\n{'=' * 60}")
        print(f"[{status}] {self.project}/{self.task_id[:30]}...")
        if status == "SOLVED":
            print(f"  Chain: {chain_id}, Iteration: {solved_iteration}")
        else:
            print(f"  All {self.config.max_chains} chains exhausted")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="RALPH Loop Runner — runs on a SINGLE task"
    )
    parser.add_argument(
        "--task-id", required=True,
        help="Unique task ID from SWE-Refactor dataset",
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
        "--max-turns", type=int, default=20,
        help="Max turns per agent invocation (default: 20)",
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

    runner = RALPHRunner(config, task_id=args.task_id)
    result = asyncio.run(runner.run())

    # Print machine-readable result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
