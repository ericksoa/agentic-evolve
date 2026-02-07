#!/usr/bin/env python3
"""Auto-generate reference-equivalent baseline solvers for all tasks in subset."""

import inspect
import json
import logging
import re
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
sys.path.insert(0, str(ALGOTUNE_DIR))

from AlgoTuneTasks.factory import TaskFactory


def extract_solve_source(task_name: str) -> str:
    """Extract the solve() method and build a standalone Solver class."""
    task = TaskFactory(task_name)
    task_module = inspect.getmodule(task)
    module_src = inspect.getsource(task_module)

    # Collect imports (skip AlgoTune-internal ones)
    imports = []
    for line in module_src.split("\n"):
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) and \
           "AlgoTune" not in stripped and \
           "logging" not in stripped and \
           stripped not in imports:
            imports.append(stripped)

    # Get solve method source
    solve_src = inspect.getsource(task.solve)
    # Dedent: remove the class-level indentation
    lines = solve_src.split("\n")
    # Find minimum indentation (excluding empty lines)
    min_indent = 999
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    dedented = []
    for line in lines:
        if line.strip():
            dedented.append(line[min_indent:])
        else:
            dedented.append("")

    # Remove docstring
    clean_lines = []
    in_docstring = False
    for line in dedented:
        if '"""' in line or "'''" in line:
            # Count quotes
            if line.count('"""') == 2 or line.count("'''") == 2:
                continue  # Single-line docstring
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            clean_lines.append(line)

    # Build the solver file
    solver = []
    solver.extend(sorted(set(imports)))
    solver.append("")
    solver.append("")
    solver.append("class Solver:")
    for line in clean_lines:
        if line.strip():
            solver.append(f"    {line}")
        else:
            solver.append("")

    return "\n".join(solver) + "\n"


def main():
    configs_dir = SHOWCASE_DIR / "configs"
    subset = json.loads((configs_dir / "subset.json").read_text())
    tasks = subset["tasks"]

    solutions_dir = SHOWCASE_DIR / "solutions"
    created = 0
    skipped = 0
    failed = 0

    for task_name in tasks:
        solver_dir = solutions_dir / task_name
        solver_path = solver_dir / "solver.py"

        if solver_path.exists():
            print(f"  exists:  {task_name}")
            skipped += 1
            continue

        try:
            code = extract_solve_source(task_name)
            solver_dir.mkdir(parents=True, exist_ok=True)
            solver_path.write_text(code)
            created += 1
            print(f"  created: {task_name}")
        except Exception as e:
            failed += 1
            print(f"  FAILED:  {task_name} [{e}]")

    print(f"\nCreated: {created}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
