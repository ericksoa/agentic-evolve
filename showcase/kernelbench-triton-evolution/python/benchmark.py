#!/usr/bin/env python3
"""
Benchmark harness for evaluating Triton kernels.

Usage:
    python benchmark.py <kernel_file.py> [--task=softmax] [--iterations=100]

The kernel file must define a function with the signature:
    def kernel(x: torch.Tensor) -> torch.Tensor
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# Add tasks to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "tasks" / "softmax"))


def load_kernel(kernel_path: str) -> Callable:
    """
    Dynamically load a kernel function from a Python file.

    The file must define either:
    - A function named 'kernel'
    - A function named 'softmax_triton'
    - A function named 'forward'
    """
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try different function names
    for name in ["kernel", "softmax_triton", "softmax", "forward"]:
        if hasattr(module, name):
            return getattr(module, name)

    raise ValueError(f"No kernel function found in {kernel_path}. "
                     f"Expected one of: kernel, softmax_triton, softmax, forward")


def load_baseline(task: str) -> dict:
    """Load baseline timings for a task."""
    baseline_file = SCRIPT_DIR / "results" / "baselines.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baselines = json.load(f)
            return baselines.get(task, {})
    return {}


def benchmark_kernel(
    kernel_fn: Callable,
    batch_size: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Benchmark a kernel with given input shape.

    Returns:
        Dict with timing statistics
    """
    # Generate input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)

    # Get reference output
    expected = F.softmax(x, dim=-1)

    # Warmup
    for _ in range(warmup):
        try:
            _ = kernel_fn(x)
        except Exception as e:
            return {"error": str(e), "valid": False}

    torch.cuda.synchronize()

    # Check correctness
    try:
        result = kernel_fn(x)
        torch.cuda.synchronize()

        if not torch.allclose(result, expected, atol=1e-5, rtol=1e-5):
            max_diff = (result - expected).abs().max().item()
            return {
                "error": f"Incorrect output, max_diff={max_diff:.2e}",
                "valid": False,
                "max_diff": max_diff,
            }

        if torch.isnan(result).any() or torch.isinf(result).any():
            return {"error": "Output contains NaN/Inf", "valid": False}

    except Exception as e:
        return {"error": str(e), "valid": False}

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = kernel_fn(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = sorted(times)
    median_time = times[len(times) // 2]
    mean_time = sum(times) / len(times)
    min_time = times[0]
    max_time = times[-1]

    return {
        "valid": True,
        "median_ms": median_time,
        "mean_ms": mean_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "iterations": iterations,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def compute_fitness(kernel_results: dict, baseline_results: dict) -> float:
    """
    Compute fitness score.

    Fitness = speedup ratio if correct, else 0
    Speedup = baseline_time / kernel_time
    """
    if not kernel_results.get("valid", False):
        return 0.0

    if not baseline_results:
        # No baseline, return raw speed (higher is better)
        return 1.0 / kernel_results["median_ms"]

    baseline_time = baseline_results.get("median_ms", float("inf"))
    kernel_time = kernel_results["median_ms"]

    if kernel_time <= 0:
        return 0.0

    return baseline_time / kernel_time


def run_full_benchmark(kernel_path: str, task: str = "softmax") -> dict:
    """
    Run full benchmark suite on a kernel.

    Returns:
        Complete benchmark results with fitness score
    """
    print(f"Loading kernel from: {kernel_path}")

    try:
        kernel_fn = load_kernel(kernel_path)
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to load kernel: {e}",
            "fitness": 0.0,
        }

    # Load task config
    task_file = SCRIPT_DIR / "tasks" / task / "task.json"
    if task_file.exists():
        with open(task_file) as f:
            task_config = json.load(f)
        benchmark_shapes = task_config.get("benchmark_shapes", [])
    else:
        benchmark_shapes = [
            {"batch_size": 64, "seq_len": 2048},
            {"batch_size": 128, "seq_len": 4096},
        ]

    # Load baselines
    baselines = load_baseline(task)

    # Run benchmarks for each shape
    results = {
        "kernel_path": kernel_path,
        "task": task,
        "benchmarks": [],
        "valid": True,
    }

    total_speedup = 0.0
    num_valid = 0

    for shape in benchmark_shapes:
        bs, sl = shape["batch_size"], shape["seq_len"]
        shape_key = f"{bs}x{sl}"

        print(f"  Benchmarking {shape_key}...", end=" ", flush=True)

        kernel_result = benchmark_kernel(kernel_fn, bs, sl)

        if kernel_result.get("valid"):
            baseline = baselines.get(shape_key, {})
            speedup = compute_fitness(kernel_result, baseline)

            kernel_result["speedup"] = speedup
            kernel_result["baseline_ms"] = baseline.get("median_ms", "N/A")

            print(f"{kernel_result['median_ms']:.3f}ms (speedup: {speedup:.2f}x)")

            total_speedup += speedup
            num_valid += 1
        else:
            print(f"FAILED: {kernel_result.get('error', 'unknown')}")
            results["valid"] = False
            kernel_result["speedup"] = 0.0

        results["benchmarks"].append({
            "shape": shape_key,
            **kernel_result
        })

    # Overall fitness is average speedup (0 if any invalid)
    if results["valid"] and num_valid > 0:
        results["fitness"] = total_speedup / num_valid
    else:
        results["fitness"] = 0.0

    # Determine fast_p levels
    fitness = results["fitness"]
    results["fast_0"] = results["valid"]  # Correctness
    results["fast_1"] = fitness >= 1.0    # Faster than baseline
    results["fast_2"] = fitness >= 2.0    # 2x faster than baseline

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels")
    parser.add_argument("kernel", help="Path to kernel Python file")
    parser.add_argument("--task", default="softmax", help="Task name")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    results = run_full_benchmark(args.kernel, args.task)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 50)
        print(f"RESULTS: {args.kernel}")
        print("=" * 50)
        print(f"Valid: {results['valid']}")
        print(f"Fitness (avg speedup): {results['fitness']:.3f}x")
        print(f"fast_0 (correct): {results['fast_0']}")
        print(f"fast_1 (>1x): {results['fast_1']}")
        print(f"fast_2 (>2x): {results['fast_2']}")

        if not results["valid"]:
            print(f"\nError: {results.get('error', 'See benchmark details')}")

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
