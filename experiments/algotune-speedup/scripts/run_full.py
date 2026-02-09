#!/usr/bin/env python3
"""
Full evolution run: process all tasks sequentially.
Skips tasks that already have results. Saves incrementally.

Usage:
    python scripts/run_full.py                # All 25 tasks
    python scripts/run_full.py --start 4      # Start from task #4 (0-indexed)
    python scripts/run_full.py --task svd     # Single task
    python scripts/run_full.py --force        # Re-run completed tasks
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
EVOLVED_DIR = SHOWCASE_DIR / "evolved"
RESULTS_DIR = SHOWCASE_DIR / "results"
CONFIGS_DIR = SHOWCASE_DIR / "configs"

# Evolution parameters
MAX_GENERATIONS = 10
POPULATION_SIZE = 6
PLATEAU = 3


def load_results() -> dict:
    path = RESULTS_DIR / "phase2_evolved.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"tasks": {}}


def save_results(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase2_evolved.json").write_text(json.dumps(results, indent=2))


def find_champion(task_name: str) -> Path | None:
    evolve_dir = SHOWCASE_DIR / ".evolve-sdk"
    if not evolve_dir.exists():
        return None

    for d in evolve_dir.iterdir():
        if not d.is_dir() or task_name not in d.name:
            continue

        champion_json = d / "champion.json"
        if champion_json.exists():
            try:
                data = json.loads(champion_json.read_text())
                f = data.get("file")
                if f:
                    p = Path(f)
                    if p.exists():
                        return p
                    p = d / f
                    if p.exists():
                        return p
            except Exception:
                pass

        mutations = d / "mutations"
        if mutations.exists():
            py_files = sorted(mutations.glob("*.py"), key=lambda x: x.stat().st_mtime)
            if py_files:
                return py_files[-1]

    return None


def evaluate(solution_path: str, task_name: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(SHOWCASE_DIR / "evaluate_task.py"),
        solution_path, "--task", task_name, "--json", "--runs", "3",
    ]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout.strip())
    except Exception:
        pass
    return {"fitness": 0.0, "valid": False, "speedup": 0.0}


def run_task(task_name: str) -> dict:
    config_path = CONFIGS_DIR / f"{task_name}.json"
    if not config_path.exists():
        return {"status": "no_config", "error": f"Missing {config_path}"}

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "evolve_sdk",
        "--config", str(config_path),
        "--mode", "perf",
        "--max-generations", str(MAX_GENERATIONS),
        "--population-size", str(POPULATION_SIZE),
        "--plateau", str(PLATEAU),
        "--no-parallel",
    ]

    print(f"  evolve-sdk: {MAX_GENERATIONS} gens, pop {POPULATION_SIZE}, plateau {PLATEAU}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(SHOWCASE_DIR),
            capture_output=True, text=True, timeout=2400,  # 40 min timeout
        )
        # Print last few meaningful lines
        for line in result.stdout.strip().split("\n")[-10:]:
            if line.strip() and not line.startswith("  "):
                print(f"  {line.strip()}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 40 min")

    elapsed = time.time() - start

    champion = find_champion(task_name)
    if champion:
        evolved_path = EVOLVED_DIR / task_name / "solver.py"
        evolved_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(champion, evolved_path)

        eval_result = evaluate(str(evolved_path), task_name)
        eval_result["status"] = "success"
        eval_result["champion"] = str(champion)
        return eval_result

    return {"status": "no_champion", "fitness": 0.0, "valid": False, "speedup": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Single task")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subset = json.loads((CONFIGS_DIR / "subset.json").read_text())
    tasks = subset["tasks"]

    if args.task:
        tasks = [args.task]
    else:
        tasks = tasks[args.start:]

    results = load_results()

    total = len(tasks)
    print(f"=== Full Evolution Run: {total} tasks ===")
    print(f"Config: {MAX_GENERATIONS} gens, pop {POPULATION_SIZE}, plateau {PLATEAU}")
    print()

    for i, task in enumerate(tasks, 1):
        existing = results["tasks"].get(task, {})
        if existing.get("valid") and existing.get("speedup", 0) > 0 and not args.force:
            print(f"[{i}/{total}] {task}: already done ({existing['speedup']}x) -- skip")
            continue

        print(f"[{i}/{total}] {task}: starting evolution...")
        task_result = run_task(task)
        results["tasks"][task] = task_result
        save_results(results)

        s = task_result.get("speedup", 0)
        st = task_result.get("status", "?")
        print(f"  => {st}: {s}x")
        print()

    # Final summary
    valid = [(t, r["speedup"]) for t, r in results["tasks"].items() if r.get("valid") and r.get("speedup", 0) > 0]
    if valid:
        n = len(valid)
        hm = n / sum(1.0 / s for _, s in valid)
        print(f"=== Done: {n} valid tasks, harmonic mean {hm:.2f}x ===")
        for t, s in sorted(valid, key=lambda x: -x[1]):
            print(f"  {s:6.2f}x  {t}")
    else:
        print("No valid results yet.")


if __name__ == "__main__":
    main()
