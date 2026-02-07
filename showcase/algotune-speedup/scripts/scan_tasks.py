#!/usr/bin/env python3
"""Scan all AlgoTune tasks for viability and generate baseline solvers."""

import inspect
import json
import logging
import sys
import time
from pathlib import Path

# Suppress all logging
logging.disable(logging.CRITICAL)

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
sys.path.insert(0, str(ALGOTUNE_DIR))

from AlgoTuneTasks.factory import TaskFactory


def scan_all_tasks():
    """Test every task: generate, solve, validate."""
    tasks_dir = ALGOTUNE_DIR / "AlgoTuneTasks"
    all_tasks = sorted([
        d.name for d in tasks_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / f"{d.name}.py").exists()
    ])

    print(f"Total task dirs: {len(all_tasks)}", flush=True)
    print(flush=True)

    ok_tasks = []
    err_tasks = []

    for task_name in all_tasks:
        try:
            task = TaskFactory(task_name)
            p = task.generate_problem(n=10, random_seed=42)
            s = task.solve(p)
            v = task.is_solution(p, s)

            start = time.perf_counter()
            for _ in range(3):
                task.solve(p)
            ref_us = (time.perf_counter() - start) / 3 * 1e6

            if v:
                ok_tasks.append((task_name, ref_us))
                print(f"OK   {ref_us:10.0f}us  {task_name}", flush=True)
            else:
                err_tasks.append(task_name)
                print(f"BAD  {ref_us:10.0f}us  {task_name}", flush=True)
        except Exception as e:
            err_tasks.append(task_name)
            print(f"ERR  {0:10.0f}us  {task_name}  [{str(e)[:60]}]", flush=True)

    print(f"\nWorking: {len(ok_tasks)}/{len(all_tasks)}", flush=True)
    print(f"Failed:  {len(err_tasks)}", flush=True)

    return ok_tasks, err_tasks


def create_baseline_solver(task_name: str) -> str:
    """Create a reference-equivalent baseline solver for a task."""
    task = TaskFactory(task_name)
    src = inspect.getsource(task.solve)

    # Extract imports from the task module
    task_module = inspect.getmodule(task)
    module_src = inspect.getsource(task_module)

    imports = set()
    for line in module_src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Skip AlgoTuneTasks imports
            if "AlgoTune" not in stripped:
                imports.add(stripped)

    # Build solver file
    solver_lines = sorted(imports)
    solver_lines.append("")
    solver_lines.append("class Solver:")
    # Adapt the solve method
    for line in src.split("\n"):
        # Skip docstrings for brevity
        if '"""' in line or "'''" in line:
            continue
        solver_lines.append(f"    {line.strip()}" if line.strip() else "")

    return "\n".join(solver_lines)


def generate_baselines(tasks: list):
    """Generate baseline solvers for all working tasks."""
    solutions_dir = SHOWCASE_DIR / "solutions"
    generated = 0

    for task_name, _ in tasks:
        solver_dir = solutions_dir / task_name
        solver_path = solver_dir / "solver.py"

        if solver_path.exists():
            print(f"  exists: {task_name}", flush=True)
            generated += 1
            continue

        try:
            solver_code = create_baseline_solver(task_name)
            solver_dir.mkdir(parents=True, exist_ok=True)
            solver_path.write_text(solver_code)
            generated += 1
            print(f"  created: {task_name}", flush=True)
        except Exception as e:
            print(f"  FAILED: {task_name} [{e}]", flush=True)

    print(f"\nGenerated {generated} baseline solvers", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Also generate baseline solvers")
    parser.add_argument("--count", type=int, default=25, help="Number of tasks to select")
    args = parser.parse_args()

    print("=== Scanning AlgoTune tasks ===", flush=True)
    ok_tasks, err_tasks = scan_all_tasks()

    if args.generate and ok_tasks:
        # Select top N by diversity (pick from different categories)
        selected = ok_tasks[:args.count]
        print(f"\n=== Generating baselines for {len(selected)} tasks ===", flush=True)
        generate_baselines(selected)

        # Save subset
        configs_dir = SHOWCASE_DIR / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        subset = {
            "tasks": [t[0] for t in selected],
            "count": len(selected),
            "selection_criteria": {"method": "scan_viable_tasks"}
        }
        (configs_dir / "subset.json").write_text(json.dumps(subset, indent=2))
        print(f"\nSubset saved to configs/subset.json", flush=True)
