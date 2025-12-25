#!/usr/bin/env python3
"""Evaluator for Fibonacci evolution."""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime

os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")

RUST_DIR = Path(__file__).parent / "rust"
EVOLUTION_FILE = Path(__file__).parent / "evolution.json"

def load_evolution_state():
    """Load or create evolution state."""
    if EVOLUTION_FILE.exists():
        with open(EVOLUTION_FILE) as f:
            return json.load(f)
    return None

def save_evolution_state(state):
    """Save evolution state."""
    state["updated"] = datetime.now().isoformat()
    with open(EVOLUTION_FILE, "w") as f:
        json.dump(state, f, indent=2)

def evaluate(evolved_code_path: str = None) -> dict:
    """Evaluate the evolved Fibonacci implementation."""

    if evolved_code_path:
        shutil.copy(evolved_code_path, RUST_DIR / "src" / "evolved.rs")

    # Build
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Build failed: {result.stderr}",
            "correctness": False
        }

    # Run benchmark
    try:
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "benchmark"],
            cwd=RUST_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        return {
            "fitness": 0.0,
            "error": "Benchmark timeout (likely naive recursion too slow)",
            "correctness": False
        }

    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Benchmark failed: {result.stderr}",
            "correctness": False
        }

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "fitness": 0.0,
            "error": f"Invalid JSON: {result.stdout}",
            "correctness": False
        }

    if not data.get("correctness", False):
        return {
            "fitness": 0.0,
            "error": "Correctness check failed",
            "correctness": False,
            "all_results": data
        }

    results = {r["name"]: r["ops_per_second"] for r in data["results"]}
    evolved_ops = results.get("evolved", 0)
    naive_ops = results.get("naive", 1)
    iterative_ops = results.get("iterative", 1)
    matrix_ops = results.get("matrix", 1)
    lookup_ops = results.get("lookup", 1)

    best_baseline = max(iterative_ops, matrix_ops, lookup_ops)

    # Calculate improvement over naive (the "bad" algorithm)
    vs_naive = ((evolved_ops / naive_ops) - 1) * 100 if naive_ops > 0 else 0
    vs_best = ((evolved_ops / best_baseline) - 1) * 100 if best_baseline > 0 else 0

    # Fitness based on improvement over naive
    # Naive is very slow, so even iterative is 1000x+ faster
    speed_ratio = evolved_ops / naive_ops if naive_ops > 0 else 0

    # Log scale for fitness since improvements can be massive
    import math
    if speed_ratio > 1:
        fitness = min(math.log10(speed_ratio) / 6, 1.0)  # Cap at 10^6 improvement
    else:
        fitness = 0.0

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": evolved_ops,
        "vs_naive": round(vs_naive, 1),
        "vs_best_baseline": round(vs_best, 1),
        "correctness": True,
        "all_results": results
    }

if __name__ == "__main__":
    evolved_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = evaluate(evolved_path)
    print(json.dumps(result, indent=2))
