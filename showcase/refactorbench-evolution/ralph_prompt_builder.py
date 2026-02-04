"""
Prompt builder for RALPH loop iterations on RefactorBench tasks.

Constructs iteration-aware prompts:
- Iteration 1 (cold start): Task description + instructions
- Iteration 2+ (warm start): Previous test output, failing tests,
  failed approaches from progress.json
"""

import json
from pathlib import Path

from refactor_agent import PYTHON_EXE


def build_prompt(
    task_info: dict,
    workdir: Path,
    progress: dict | None = None,
    iteration: int = 1,
    max_iterations: int = 10,
) -> str:
    """Build a prompt for a RALPH loop iteration.

    Args:
        task_info: Dict with repo_name, task_name, description, test_file
        workdir: Working directory (already contains the repo)
        progress: progress.json contents (None for iteration 1)
        iteration: Current iteration number (1-based)
        max_iterations: Maximum iterations in this chain

    Returns:
        Complete prompt string for claude -p
    """
    parts = []

    # System instructions (always included)
    parts.append(_system_instructions())

    # Task description (always included)
    parts.append(_task_section(task_info, workdir))

    # Iteration context (iteration 2+)
    if iteration > 1 and progress:
        parts.append(_iteration_context(progress, iteration, max_iterations))

    # Validation instructions (always included)
    parts.append(_validation_section(task_info, workdir))

    return "\n\n".join(parts)


def _system_instructions() -> str:
    """Core system prompt for the refactoring agent."""
    return """You are an expert Python developer performing code refactoring tasks.
You must make precise, targeted changes to the codebase.

## Core Principles

1. **Read before writing.** Always read a file before modifying it. Understand the full context.
2. **Search exhaustively.** Use Grep and Glob to find ALL references to functions, classes, imports, and variables you are modifying. Missing a reference is the #1 cause of test failures.
3. **Modify ALL call sites.** When you rename, move, or change a function's signature, update EVERY file that imports or calls it.
4. **Preserve semantics.** Refactoring changes structure, not behavior. The code must do the same thing after your changes.
5. **Think about imports.** When moving code between files, update both the source file (exports) and all importing files.

## Strategy

1. Read the task description carefully and identify what needs to change
2. Use Grep to find ALL files that reference the things being changed
3. Plan your changes to cover every reference
4. Make all changes
5. Run the validation test to check your work
6. If tests fail, read the error output carefully and fix the issues"""


def _task_section(task_info: dict, workdir: Path) -> str:
    """Task description section."""
    return f"""## Refactoring Task

You are working in the directory: {workdir.resolve()}

{task_info['description']}"""


def _iteration_context(progress: dict, iteration: int, max_iterations: int) -> str:
    """Context from previous iterations (warm start)."""
    parts = []

    parts.append(f"## Iteration {iteration}/{max_iterations} - Previous Attempts")
    parts.append(
        "This is NOT your first attempt. Previous iterations have already modified "
        "files in this directory. Read the current state of files before making changes."
    )

    # Best score so far
    best = progress.get("best_score", 0)
    total = None
    test_results = progress.get("test_results", [])
    if test_results:
        last = test_results[-1]
        total = last.get("tests_total")
        parts.append(
            f"\n**Best score so far:** {best}/{total} tests passing"
            if total
            else f"\n**Best score so far:** {best} tests passing"
        )

    # Last test result
    if test_results:
        last = test_results[-1]
        last_passed = last.get("tests_passed", 0)
        last_total = last.get("tests_total", "?")
        parts.append(
            f"\n**Last iteration result:** {last_passed}/{last_total} tests passing"
        )

        # Failing test names
        failing = last.get("failing_tests", [])
        if failing:
            parts.append("\n**Currently failing tests:**")
            for t in failing[:20]:
                parts.append(f"- `{t}`")

        # Error summary
        err = last.get("error_summary", "")
        if err:
            parts.append(f"\n**Error summary:** {err}")

        # Last pytest output (truncated)
        stdout = last.get("test_stdout", "")
        if stdout:
            # Show last 1500 chars of pytest output
            truncated = stdout[-1500:]
            if len(stdout) > 1500:
                truncated = "... (truncated)\n" + truncated
            parts.append(f"\n**Last test output:**\n```\n{truncated}\n```")

    # Failed approaches (DO NOT REPEAT)
    failed = progress.get("failed_approaches", [])
    if failed:
        parts.append("\n## DO NOT REPEAT These Failed Approaches")
        parts.append(
            "The following approaches were already tried and FAILED. "
            "Do NOT try them again. Find a different approach."
        )
        for i, approach in enumerate(failed[-10:], 1):  # Last 10
            parts.append(f"{i}. {approach}")

    return "\n".join(parts)


def _validation_section(task_info: dict, workdir: Path) -> str:
    """Validation/test instructions."""
    python_exe = PYTHON_EXE

    test_file = task_info.get("test_file", "")
    test_filename = Path(test_file).name if test_file else "test.py"

    return f"""## Validation

After making all changes, validate by running the AST-based test. The test checks
that your refactoring produced the correct code structure (imports, function names, etc.).

To run the test:
1. Create a scripts/ subdirectory if it doesn't exist: mkdir -p scripts
2. Copy the test file: cp {test_file} scripts/{test_filename}
3. If a conftest.py exists at the repo root, temporarily rename it: mv conftest.py conftest.py.bak (if it exists)
4. Write a minimal scripts/conftest.py with: import os\\nimport pytest\\n\\n@pytest.fixture\\ndef chdir(tmp_path, monkeypatch):\\n    monkeypatch.chdir(tmp_path)\\n    return tmp_path
5. Run: cd scripts && {python_exe} -m pytest {test_filename} -v --tb=short --rootdir=. --override-ini=pythonpath=
6. Restore conftest if moved: mv conftest.py.bak conftest.py (if it was moved)

The tests use relative paths like ../src/flask/app.py (relative to scripts/), so ../ points to the repo root.

IMPORTANT: Make ALL required code changes FIRST, then run the test to validate. If tests fail, read the error output carefully and fix the issues."""
