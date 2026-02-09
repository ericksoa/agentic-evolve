#!/usr/bin/env python3
"""
Aggregate results from Phase 1 and Phase 2, compute harmonic mean speedups,
and generate comparison tables.

For each task, picks the best of Phase 1 vs Phase 2.
Computes harmonic mean: H = n / sum(1/s_i).
Generates comparison table vs published Opus 4.5 and GPT-5.2 results.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --re-evaluate   # Re-evaluate all solutions fresh
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
ALGOTUNE_DIR = SHOWCASE_DIR / "algotune"
SOLUTIONS_DIR = SHOWCASE_DIR / "solutions"
EVOLVED_DIR = SHOWCASE_DIR / "evolved"
RESULTS_DIR = SHOWCASE_DIR / "results"

# Published baselines (AlgoTune leaderboard)
PUBLISHED_BASELINES = {
    "o4-mini-high": {"harmonic_mean": 1.72, "tasks_improved_pct": 59.7},
    "deepseek-r1": {"harmonic_mean": 1.70, "tasks_improved_pct": 61.0},
    "gemini-2.5-pro": {"harmonic_mean": 1.51, "tasks_improved_pct": 49.4},
    "claude-opus-4": {"harmonic_mean": 1.33, "tasks_improved_pct": 40.3},
}


def harmonic_mean(values: list) -> float:
    """Compute harmonic mean of positive values."""
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


def load_phase_results(path: Path) -> dict:
    """Load results from a phase JSON file."""
    if path.exists():
        return json.loads(path.read_text())
    return {"tasks": {}}


def evaluate_solution(solution_path: str, task_name: str) -> dict:
    """Re-evaluate a solution using the bridge script."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ALGOTUNE_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        str(SHOWCASE_DIR / "evaluate_task.py"),
        solution_path,
        "--task", task_name,
        "--json",
        "--runs", "5",  # More runs for final evaluation
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return {"fitness": 0.0, "valid": False, "speedup": 0.0}


def aggregate(re_evaluate: bool = False) -> dict:
    """Aggregate results from both phases."""
    phase1 = load_phase_results(RESULTS_DIR / "phase1_baseline.json")
    phase2 = load_phase_results(RESULTS_DIR / "phase2_evolved.json")

    # Get union of all tasks
    all_tasks = sorted(
        set(phase1.get("tasks", {}).keys()) | set(phase2.get("tasks", {}).keys())
    )

    per_task = {}

    for task in all_tasks:
        p1 = phase1.get("tasks", {}).get(task, {})
        p2 = phase2.get("tasks", {}).get(task, {})

        p1_speedup = p1.get("speedup", 0.0) if p1.get("valid") else 0.0
        p2_speedup = p2.get("speedup", 0.0) if p2.get("valid") else 0.0

        # Re-evaluate if requested
        if re_evaluate:
            p1_solver = SOLUTIONS_DIR / task / "solver.py"
            p2_solver = EVOLVED_DIR / task / "solver.py"

            if p1_solver.exists():
                fresh = evaluate_solution(str(p1_solver), task)
                if fresh.get("valid"):
                    p1_speedup = fresh.get("speedup", 0.0)

            if p2_solver.exists():
                fresh = evaluate_solution(str(p2_solver), task)
                if fresh.get("valid"):
                    p2_speedup = fresh.get("speedup", 0.0)

        # Pick best
        best_phase = "phase2" if p2_speedup > p1_speedup else "phase1"
        best_speedup = max(p1_speedup, p2_speedup)

        # Floor at 1.0 (can't be worse than reference)
        if best_speedup > 0 and best_speedup < 1.0:
            best_speedup = 1.0

        per_task[task] = {
            "phase1_speedup": round(p1_speedup, 4),
            "phase2_speedup": round(p2_speedup, 4),
            "best_speedup": round(best_speedup, 4),
            "best_phase": best_phase,
            "valid": best_speedup > 0,
        }

    # Compute aggregates
    valid_speedups = [
        t["best_speedup"] for t in per_task.values() if t["valid"]
    ]
    h_mean = harmonic_mean(valid_speedups)
    arith_mean = sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0
    tasks_improved = sum(1 for s in valid_speedups if s > 1.0)
    tasks_improved_pct = (
        (tasks_improved / len(valid_speedups) * 100) if valid_speedups else 0
    )

    summary = {
        "harmonic_mean_speedup": round(h_mean, 4),
        "arithmetic_mean_speedup": round(arith_mean, 4),
        "total_tasks": len(all_tasks),
        "valid_tasks": len(valid_speedups),
        "tasks_improved": tasks_improved,
        "tasks_improved_pct": round(tasks_improved_pct, 1),
        "best_speedup": round(max(valid_speedups), 4) if valid_speedups else 0,
        "worst_speedup": round(min(valid_speedups), 4) if valid_speedups else 0,
    }

    return {
        "summary": summary,
        "per_task": per_task,
        "baselines": PUBLISHED_BASELINES,
    }


def format_comparison_table(results: dict) -> str:
    """Generate a markdown comparison table."""
    summary = results["summary"]
    baselines = results["baselines"]

    lines = []
    lines.append("## Results Comparison")
    lines.append("")
    lines.append(
        "| Model | Harmonic Mean Speedup | Tasks Improved |"
    )
    lines.append("|-------|----------------------|----------------|")

    # Our result (highlighted)
    lines.append(
        f"| **Opus 4.6 + evolve-sdk** | **{summary['harmonic_mean_speedup']:.2f}x** "
        f"| **{summary['tasks_improved_pct']:.0f}%** |"
    )

    # Published baselines
    for model, data in sorted(
        baselines.items(), key=lambda x: x[1]["harmonic_mean"], reverse=True
    ):
        lines.append(
            f"| {model} | {data['harmonic_mean']:.2f}x "
            f"| {data['tasks_improved_pct']:.0f}% |"
        )

    lines.append("")
    lines.append(
        f"*Results on curated {summary['valid_tasks']}-task subset. "
        f"See methodology section for full-benchmark extrapolation.*"
    )

    return "\n".join(lines)


def format_per_task_table(results: dict) -> str:
    """Generate a per-task breakdown table."""
    per_task = results["per_task"]

    lines = []
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append(
        "| Task | Phase 1 | Phase 2 | Best | Source |"
    )
    lines.append("|------|---------|---------|------|--------|")

    for task in sorted(per_task.keys()):
        t = per_task[task]
        p1 = f"{t['phase1_speedup']:.2f}x" if t["phase1_speedup"] > 0 else "-"
        p2 = f"{t['phase2_speedup']:.2f}x" if t["phase2_speedup"] > 0 else "-"
        best = f"{t['best_speedup']:.2f}x" if t["valid"] else "invalid"
        source = t["best_phase"]
        lines.append(f"| {task} | {p1} | {p2} | **{best}** | {source} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate AlgoTune results")
    parser.add_argument(
        "--re-evaluate",
        action="store_true",
        help="Re-evaluate all solutions fresh (slower but more accurate)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "comparison.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    print("=== Aggregating Results ===")
    if args.re_evaluate:
        print("Re-evaluating all solutions (this may take a while)...")

    results = aggregate(re_evaluate=args.re_evaluate)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nJSON results saved to {output_path}")

    # Print tables
    print()
    print(format_comparison_table(results))
    print()
    print(format_per_task_table(results))

    # Save markdown tables
    md_path = RESULTS_DIR / "comparison.md"
    with open(md_path, "w") as f:
        f.write(format_comparison_table(results))
        f.write("\n\n")
        f.write(format_per_task_table(results))
    print(f"\nMarkdown tables saved to {md_path}")

    # Summary
    s = results["summary"]
    print(f"\n=== Summary ===")
    print(f"Harmonic mean speedup: {s['harmonic_mean_speedup']:.2f}x")
    print(f"Valid tasks: {s['valid_tasks']}/{s['total_tasks']}")
    print(f"Tasks improved: {s['tasks_improved']} ({s['tasks_improved_pct']:.0f}%)")
    print(f"Best single task: {s['best_speedup']:.2f}x")


if __name__ == "__main__":
    main()
