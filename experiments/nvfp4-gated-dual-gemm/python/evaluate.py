"""
Evaluation Script for NVFP4 Gated Dual GEMM Evolution

This script evaluates kernel mutations for:
1. Correctness (must match reference)
2. Performance (speedup vs baseline)

Usage:
    python evaluate.py <submission_file.py>
"""
import sys
import time
import json
import importlib.util
import torch
from pathlib import Path

from task import (
    input_t, output_t,
    TRAIN_SPECS, VALID_SPECS, TEST_SPECS,
    generate_test_data, check_correctness
)
from reference import ref_kernel


# Benchmark configuration
WARMUP_ITERS = 5
BENCHMARK_ITERS = 20
TIMEOUT_SECONDS = 300  # 5 minutes max


def load_submission(filepath: str):
    """Dynamically load a submission module"""
    spec = importlib.util.spec_from_file_location("submission", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_kernel(kernel_fn, data: input_t, warmup: int = WARMUP_ITERS, iters: int = BENCHMARK_ITERS) -> float:
    """
    Benchmark a kernel and return median time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        kernel_fn(data)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        kernel_fn(data)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = sorted(times)
    return times[len(times) // 2]  # Median


def evaluate_submission(submission_path: str) -> dict:
    """
    Evaluate a submission on all test specs.

    Returns:
        {
            "valid": bool,
            "fitness": float,  # speedup vs reference
            "train": {...},
            "valid": {...},
            "test": {...},
            "error": str or None
        }
    """
    results = {
        "valid": False,
        "fitness": 0.0,
        "train": {"specs": [], "avg_speedup": 0.0},
        "valid": {"specs": [], "avg_speedup": 0.0},
        "test": {"specs": [], "avg_speedup": 0.0},
        "error": None
    }

    try:
        # Load submission
        submission = load_submission(submission_path)
        kernel_fn = submission.custom_kernel

        # Evaluate on each dataset
        all_correct = True
        all_speedups = []

        for dataset_name, specs in [
            ("train", TRAIN_SPECS),
            ("valid", VALID_SPECS),
            ("test", TEST_SPECS)
        ]:
            dataset_speedups = []

            for spec in specs:
                # Generate test data
                data = generate_test_data(spec)

                # Get reference result
                ref_output = ref_kernel(data)
                ref_time = benchmark_kernel(ref_kernel, generate_test_data(spec))

                # Get submission result
                data = generate_test_data(spec)  # Fresh data
                try:
                    sub_output = kernel_fn(data)
                    sub_time = benchmark_kernel(kernel_fn, generate_test_data(spec))
                except Exception as e:
                    results["error"] = f"Kernel error on {spec}: {str(e)}"
                    return results

                # Check correctness
                is_correct, max_error = check_correctness(sub_output, ref_output)
                if not is_correct:
                    all_correct = False
                    results["error"] = f"Correctness failed on {spec}: max_error={max_error}"

                # Calculate speedup
                speedup = ref_time / sub_time if sub_time > 0 else 0

                spec_result = {
                    "m": spec["m"],
                    "n": spec["n"],
                    "k": spec["k"],
                    "l": spec["l"],
                    "correct": is_correct,
                    "max_error": max_error,
                    "ref_time_ms": ref_time,
                    "sub_time_ms": sub_time,
                    "speedup": speedup
                }

                results[dataset_name]["specs"].append(spec_result)
                dataset_speedups.append(speedup)

            # Dataset average
            if dataset_speedups:
                results[dataset_name]["avg_speedup"] = sum(dataset_speedups) / len(dataset_speedups)
                all_speedups.extend(dataset_speedups)

        # Overall fitness = geometric mean of speedups (only if correct)
        if all_correct and all_speedups:
            results["valid"] = True
            from math import prod
            results["fitness"] = prod(all_speedups) ** (1.0 / len(all_speedups))
        else:
            results["fitness"] = 0.0

    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <submission.py>")
        sys.exit(1)

    submission_path = sys.argv[1]

    if not Path(submission_path).exists():
        print(f"Error: {submission_path} not found")
        sys.exit(1)

    print(f"Evaluating: {submission_path}")
    print("=" * 60)

    results = evaluate_submission(submission_path)

    # Print results
    print(f"\nValid: {results['valid']}")
    print(f"Fitness (geometric mean speedup): {results['fitness']:.4f}")

    if results["error"]:
        print(f"\nError: {results['error']}")

    for dataset in ["train", "valid", "test"]:
        print(f"\n{dataset.upper()} Results:")
        print(f"  Average Speedup: {results[dataset]['avg_speedup']:.4f}x")
        for spec in results[dataset]["specs"]:
            status = "OK" if spec["correct"] else "FAIL"
            print(f"  [{status}] {spec['m']}x{spec['n']}x{spec['k']}: "
                  f"{spec['speedup']:.2f}x ({spec['sub_time_ms']:.2f}ms vs {spec['ref_time_ms']:.2f}ms)")

    # Output JSON for evolution
    print("\n" + "=" * 60)
    print("JSON Output:")
    print(json.dumps(results, indent=2))

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
