"""
Refactoring agent that uses Claude Code CLI to perform
multi-file refactoring tasks from RefactorBench.

Uses `claude -p` (print mode) as the agent backend, which handles
tool use, file editing, and multi-turn interaction internally.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).parent / ".refactorbench"
DEFAULT_MODEL = "sonnet"
MAX_TURNS = 15

# Use Python 3.12 for running AST-based tests.
# RefactorBench repos target Python 3.8-3.12; Python 3.13+ has AST changes
# that cause false failures. This MUST stay within the supported version envelope.
_PROJECT_DIR = Path(__file__).parent.resolve()
PYTHON_EXE = str(_PROJECT_DIR / ".venv-3.12" / "bin" / "python3")


def load_strategy(strategy_path: str) -> dict:
    with open(strategy_path) as f:
        return json.load(f)


def get_task_info(repo_name: str, task_name: str) -> dict:
    """Load task description and identify the test file."""
    problems_dir = BENCH_ROOT / "problems" / "descriptive_problems" / repo_name
    tests_dir = BENCH_ROOT / "tests" / repo_name

    task_file = problems_dir / f"{task_name}-task.txt"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    task_description = task_file.read_text().strip()

    # Find matching test file
    test_file = None
    for tf in tests_dir.iterdir():
        stem = tf.stem
        if task_name in stem or stem.replace("-test", "") == task_name.replace("-task", ""):
            test_file = tf
            break

    # Fall back: use the mapping
    if test_file is None:
        test_file = _find_test_from_mapping(repo_name, task_name)

    return {
        "repo_name": repo_name,
        "task_name": task_name,
        "description": task_description,
        "test_file": str(test_file) if test_file else None,
    }


def _find_test_from_mapping(repo_name: str, task_name: str) -> Path | None:
    """Use the descriptive_mapping.py to find the test file for a task."""
    mapping_file = BENCH_ROOT / "scripts" / "descriptive_mapping.py"
    if not mapping_file.exists():
        return None

    mapping_text = mapping_file.read_text()
    local_ns = {}
    exec(mapping_text, {}, local_ns)
    file_mapping = local_ns.get("file_mapping", {})

    for test_path, task_path in file_mapping.items():
        if task_name in task_path:
            resolved = (BENCH_ROOT / "scripts" / test_path).resolve()
            if resolved.exists():
                return resolved

    return None


def setup_workdir(repo_name: str, work_root: Path) -> Path:
    """Copy the repository to a working directory."""
    src = BENCH_ROOT / "repositories" / repo_name
    dst = work_root / repo_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run_test(test_file: str, workdir: Path, repo_name: str) -> dict:
    """Run the AST-based test against the working directory."""
    scripts_dir = workdir.resolve() / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Copy the test file into scripts/
    test_path = Path(test_file).resolve()
    local_test = scripts_dir / test_path.name
    shutil.copy2(test_file, local_test)

    # Write a minimal conftest.py with common fixtures
    conftest = scripts_dir / "conftest.py"
    conftest.write_text(
        "import os\nimport pytest\n\n"
        "@pytest.fixture\n"
        "def chdir(tmp_path, monkeypatch):\n"
        "    monkeypatch.chdir(tmp_path)\n"
        "    return tmp_path\n"
    )

    # Temporarily disable any repo-level conftest.py that might import
    # unavailable dependencies (e.g. twisted, django)
    repo_conftest = workdir / "conftest.py"
    repo_conftest_bak = workdir / "conftest.py.bak"
    conftest_moved = False
    if repo_conftest.exists():
        repo_conftest.rename(repo_conftest_bak)
        conftest_moved = True

    python_exe = PYTHON_EXE

    try:
        result = subprocess.run(
            [python_exe, "-m", "pytest", str(local_test.resolve()),
             "-v", "--tb=short", f"--rootdir={scripts_dir}",
             "--override-ini=pythonpath="],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir),
            timeout=60,
        )
    finally:
        if conftest_moved and repo_conftest_bak.exists():
            repo_conftest_bak.rename(repo_conftest)

    return {
        "passed": result.returncode == 0,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "returncode": result.returncode,
    }


def build_agent_prompt(strategy: dict, task_info: dict, workdir: Path) -> str:
    """Build the full prompt for the Claude Code agent."""
    parts = []

    # Strategy-informed system instructions
    base_prompt = strategy.get("system_prompt", "You are an expert Python developer.")
    parts.append(base_prompt)

    # Add approach steps if defined
    approach = strategy.get("approach", {})
    if approach.get("steps"):
        parts.append("\n## Approach\nFollow these steps:")
        for i, step in enumerate(approach["steps"], 1):
            parts.append(f"{i}. {step}")

    # Add file discovery instructions
    fd = strategy.get("file_discovery", {})
    if fd.get("enabled"):
        strat = fd.get("strategy", "")
        parts.append(f"\n## File Discovery\n{strat}")

    # Add state tracking instructions
    st = strategy.get("state_tracking", {})
    if st.get("enabled"):
        instructions = st.get("instructions", "Keep track of your progress and what files you've modified.")
        parts.append(f"\n## State Tracking\n{instructions}")

    # Add incremental validation
    iv = strategy.get("incremental_validation", {})
    if iv.get("enabled"):
        instructions = iv.get("instructions", "Run tests after each significant change.")
        parts.append(f"\n## Incremental Validation\n{instructions}")

    # Add error recovery
    er = strategy.get("error_recovery", {})
    if er.get("enabled"):
        instructions = er.get("instructions", "If tests fail, analyze the error and retry.")
        max_retries = er.get("max_retries", 3)
        parts.append(f"\n## Error Recovery\n{instructions}\nMax retries: {max_retries}")

    # Add context management
    cm = strategy.get("context_management", {})
    if cm.get("strategy"):
        parts.append(f"\n## Context Management\n{cm['strategy']}")

    # Add any custom sections
    for key, value in strategy.items():
        if key.startswith("custom_") and isinstance(value, str):
            parts.append(f"\n## {key.replace('custom_', '').replace('_', ' ').title()}\n{value}")

    strategy_text = "\n".join(parts)

    # Build the task-specific prompt
    python_exe = PYTHON_EXE

    test_file = task_info.get("test_file", "")
    test_filename = Path(test_file).name if test_file else "test.py"

    prompt = f"""{strategy_text}

## Refactoring Task

You are working in the directory: {workdir.resolve()}

{task_info['description']}

## Validation

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

    return prompt


def run_agent(strategy: dict, task_info: dict, workdir: Path,
              model: str = None, verbose: bool = False) -> dict:
    """Run the refactoring agent using Claude Code CLI."""
    model = model or strategy.get("model", DEFAULT_MODEL)
    max_turns = strategy.get("max_turns", MAX_TURNS)

    prompt = build_agent_prompt(strategy, task_info, workdir)

    # Build claude command
    cmd = [
        "claude",
        "-p",  # print mode (non-interactive)
        "--model", model,
        "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        "--output-format", "json",
        "--max-turns", str(max_turns),
        prompt,
    ]

    if verbose:
        print(f"  Running claude agent in {workdir}...", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(workdir.resolve()),
            timeout=600,  # 10 min max per task
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "cli"},
        )
    except subprocess.TimeoutExpired:
        if verbose:
            print("  Agent timed out", file=sys.stderr)
        return {
            "passed": False,
            "turns_used": max_turns,
            "error": "Agent timed out (600s)",
            "test_results": [],
        }
    except Exception as e:
        return {
            "passed": False,
            "turns_used": 0,
            "error": f"Failed to run claude: {e}",
            "test_results": [],
        }

    if verbose:
        if result.returncode != 0:
            print(f"  Claude exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)

    # Parse the JSON output to determine how many turns were used
    turns_used = 0
    try:
        output = json.loads(result.stdout)
        if isinstance(output, dict):
            turns_used = output.get("num_turns", 0)
    except (json.JSONDecodeError, TypeError):
        pass

    # Now run the AST test ourselves to determine pass/fail definitively
    test_result = run_test(task_info["test_file"], workdir, task_info["repo_name"])

    if verbose:
        status = "PASSED" if test_result["passed"] else "FAILED"
        print(f"  Test result: {status}", file=sys.stderr)

    return {
        "passed": test_result["passed"],
        "turns_used": turns_used,
        "test_results": [{"turn": turns_used, "passed": test_result["passed"]}],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run refactoring agent on a single task")
    parser.add_argument("strategy", help="Path to strategy JSON")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--task", required=True, help="Task name (without -task.txt suffix)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--workdir", help="Working directory root", default="workdirs")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    task_info = get_task_info(args.repo, args.task)

    work_root = Path(args.workdir)
    work_root.mkdir(parents=True, exist_ok=True)
    workdir = setup_workdir(args.repo, work_root)

    result = run_agent(strategy, task_info, workdir, model=args.model, verbose=args.verbose)
    print(json.dumps(result, indent=2))
