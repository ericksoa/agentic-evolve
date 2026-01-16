#!/usr/bin/env python3
"""
run_benchmark.py - Benchmark cuOpt configurations on real LP problems.

This script provides a reproducible benchmark comparing:
1. Baseline cuOpt configuration (defaults)
2. Evolved configuration (from autotuner)

Uses real Netlib LP benchmarks - the standard test suite for LP solvers.

Usage:
    # Download benchmarks first
    python3 -m showcase.cuopt_lp_autotuner.download_benchmarks --minimal

    # Run benchmark comparison
    python3 -m showcase.cuopt_lp_autotuner.run_benchmark --benchmark-dir ./benchmarks

    # With evolved config
    python3 -m showcase.cuopt_lp_autotuner.run_benchmark \
        --benchmark-dir ./benchmarks \
        --evolved-config .evolve-sdk/cuopt_evolution_20gen/results.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from .cuopt_config import CuOptConfig
from .cuopt_solver import CuOptSolver, SolveStatus, is_cuopt_available
from .mps_loader import load_problem, LPData
from .create_benchmarks import load_benchmark


@dataclass
class BenchmarkResult:
    """Result for a single problem."""
    name: str
    n_vars: int
    n_cons: int

    # Baseline results
    baseline_status: str = ""
    baseline_time_ms: float = 0.0
    baseline_objective: Optional[float] = None

    # Evolved results
    evolved_status: str = ""
    evolved_time_ms: float = 0.0
    evolved_objective: Optional[float] = None

    # Comparison
    speedup: float = 1.0
    objectives_match: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "n_cons": self.n_cons,
            "baseline_status": self.baseline_status,
            "baseline_time_ms": self.baseline_time_ms,
            "baseline_objective": self.baseline_objective,
            "evolved_status": self.evolved_status,
            "evolved_time_ms": self.evolved_time_ms,
            "evolved_objective": self.evolved_objective,
            "speedup": self.speedup,
            "objectives_match": self.objectives_match,
        }


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""
    n_problems: int = 0
    n_baseline_optimal: int = 0
    n_evolved_optimal: int = 0

    baseline_total_time_ms: float = 0.0
    evolved_total_time_ms: float = 0.0

    mean_speedup: float = 1.0
    median_speedup: float = 1.0
    geomean_speedup: float = 1.0

    problems_faster: int = 0
    problems_slower: int = 0
    problems_same: int = 0

    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_problems": self.n_problems,
            "n_baseline_optimal": self.n_baseline_optimal,
            "n_evolved_optimal": self.n_evolved_optimal,
            "baseline_total_time_ms": self.baseline_total_time_ms,
            "evolved_total_time_ms": self.evolved_total_time_ms,
            "overall_speedup": self.baseline_total_time_ms / self.evolved_total_time_ms if self.evolved_total_time_ms > 0 else 1.0,
            "mean_speedup": self.mean_speedup,
            "median_speedup": self.median_speedup,
            "geomean_speedup": self.geomean_speedup,
            "problems_faster": self.problems_faster,
            "problems_slower": self.problems_slower,
            "problems_same": self.problems_same,
            "results": [r.to_dict() for r in self.results],
        }


def load_evolved_config(path: str) -> CuOptConfig:
    """Load evolved configuration from results JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    if "best_config" in data:
        return CuOptConfig.from_dict(data["best_config"])
    else:
        return CuOptConfig.from_dict(data)


def run_benchmark(
    benchmark_dir: str,
    evolved_config: Optional[CuOptConfig] = None,
    n_runs: int = 3,
    timeout_ms: float = 30000,
    verbose: bool = True,
) -> BenchmarkSummary:
    """
    Run benchmark comparison on LP problems.

    Args:
        benchmark_dir: Directory containing .mps files
        evolved_config: Evolved configuration (None = skip evolved)
        n_runs: Number of runs per problem (for timing stability)
        timeout_ms: Timeout per solve
        verbose: Print progress

    Returns:
        BenchmarkSummary with all results
    """
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        print(f"Error: Benchmark directory '{benchmark_dir}' not found")
        print("Run: python3 -m showcase.cuopt_lp_autotuner.download_benchmarks --minimal")
        sys.exit(1)

    # Find all benchmark files (JSON or MPS)
    json_files = sorted(benchmark_path.glob("*.json"))
    mps_files = sorted(benchmark_path.glob("*.mps"))
    benchmark_files = json_files + mps_files

    if not benchmark_files:
        print(f"Error: No benchmark files found in '{benchmark_dir}'")
        print("Run: python3 -m showcase.cuopt_lp_autotuner.create_benchmarks --output ./benchmarks")
        sys.exit(1)

    if verbose:
        print("=" * 70)
        print("cuOpt LP AutoTuner - Benchmark Comparison")
        print("=" * 70)
        print()
        print(f"Backend: {'NVIDIA cuOpt (GPU)' if is_cuopt_available() else 'scipy HiGHS (CPU fallback)'}")
        print(f"Problems: {len(benchmark_files)}")
        print(f"Runs per problem: {n_runs}")
        print(f"Timeout: {timeout_ms/1000:.1f}s")
        print()

    # Initialize solver and configs
    solver = CuOptSolver(use_fallback=True, verbose=False)
    baseline_config = CuOptConfig.baseline()

    if evolved_config is None:
        evolved_config = baseline_config  # Compare baseline to itself if no evolved

    if verbose:
        print("Baseline config: cuOpt defaults")
        print(f"Evolved config: method={evolved_config.method.name}, "
              f"pdlp_mode={evolved_config.pdlp_solver_mode.value}")
        print()
        print("-" * 70)
        print(f"{'Problem':<12} {'Size':>10} {'Baseline':>12} {'Evolved':>12} {'Speedup':>10}")
        print("-" * 70)

    results = []

    for bench_file in benchmark_files:
        try:
            # Load problem (JSON or MPS)
            if str(bench_file).endswith('.json'):
                lp = load_benchmark(str(bench_file))
            else:
                lp = load_problem(str(bench_file))
            name = bench_file.stem

            result = BenchmarkResult(
                name=name,
                n_vars=lp.n_vars,
                n_cons=lp.n_cons,
            )

            # Run baseline
            baseline_times = []
            baseline_obj = None
            baseline_status = None

            for _ in range(n_runs):
                res = solver.solve(lp, baseline_config, timeout_ms)
                baseline_times.append(res.solve_time_ms)
                if res.status == SolveStatus.OPTIMAL:
                    baseline_obj = res.objective
                baseline_status = res.status

            result.baseline_time_ms = np.median(baseline_times)
            result.baseline_objective = baseline_obj
            result.baseline_status = baseline_status.value if baseline_status else "unknown"

            # Run evolved
            evolved_times = []
            evolved_obj = None
            evolved_status = None

            for _ in range(n_runs):
                res = solver.solve(lp, evolved_config, timeout_ms)
                evolved_times.append(res.solve_time_ms)
                if res.status == SolveStatus.OPTIMAL:
                    evolved_obj = res.objective
                evolved_status = res.status

            result.evolved_time_ms = np.median(evolved_times)
            result.evolved_objective = evolved_obj
            result.evolved_status = evolved_status.value if evolved_status else "unknown"

            # Compute speedup
            if result.evolved_time_ms > 0:
                result.speedup = result.baseline_time_ms / result.evolved_time_ms
            else:
                result.speedup = 1.0

            # Check objectives match
            if baseline_obj is not None and evolved_obj is not None:
                rel_diff = abs(baseline_obj - evolved_obj) / (abs(baseline_obj) + 1e-10)
                result.objectives_match = rel_diff < 0.01  # 1% tolerance

            results.append(result)

            if verbose:
                size_str = f"{lp.n_vars}x{lp.n_cons}"
                speedup_str = f"{result.speedup:.2f}x"
                if result.speedup > 1.05:
                    speedup_str = f"\033[92m{speedup_str}\033[0m"  # Green
                elif result.speedup < 0.95:
                    speedup_str = f"\033[91m{speedup_str}\033[0m"  # Red

                print(f"{name:<12} {size_str:>10} {result.baseline_time_ms:>10.2f}ms "
                      f"{result.evolved_time_ms:>10.2f}ms {speedup_str:>10}")

        except Exception as e:
            if verbose:
                print(f"{bench_file.stem:<12} {'ERROR':>10} {str(e)[:40]}")
            continue

    # Compute summary
    summary = BenchmarkSummary(results=results)
    summary.n_problems = len(results)
    summary.n_baseline_optimal = sum(1 for r in results if r.baseline_status == "optimal")
    summary.n_evolved_optimal = sum(1 for r in results if r.evolved_status == "optimal")
    summary.baseline_total_time_ms = sum(r.baseline_time_ms for r in results)
    summary.evolved_total_time_ms = sum(r.evolved_time_ms for r in results)

    speedups = [r.speedup for r in results if r.speedup > 0]
    if speedups:
        summary.mean_speedup = np.mean(speedups)
        summary.median_speedup = np.median(speedups)
        summary.geomean_speedup = np.exp(np.mean(np.log(speedups)))

    summary.problems_faster = sum(1 for r in results if r.speedup > 1.05)
    summary.problems_slower = sum(1 for r in results if r.speedup < 0.95)
    summary.problems_same = summary.n_problems - summary.problems_faster - summary.problems_slower

    if verbose:
        print("-" * 70)
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print(f"Problems solved:     {summary.n_problems}")
        print(f"Baseline optimal:    {summary.n_baseline_optimal}/{summary.n_problems}")
        print(f"Evolved optimal:     {summary.n_evolved_optimal}/{summary.n_problems}")
        print()
        print(f"Total baseline time: {summary.baseline_total_time_ms:.1f}ms")
        print(f"Total evolved time:  {summary.evolved_total_time_ms:.1f}ms")
        overall = summary.baseline_total_time_ms / summary.evolved_total_time_ms if summary.evolved_total_time_ms > 0 else 1.0
        print(f"Overall speedup:     {overall:.2f}x")
        print()
        print(f"Mean speedup:        {summary.mean_speedup:.2f}x")
        print(f"Median speedup:      {summary.median_speedup:.2f}x")
        print(f"Geometric mean:      {summary.geomean_speedup:.2f}x")
        print()
        print(f"Problems faster:     {summary.problems_faster} ({100*summary.problems_faster/summary.n_problems:.0f}%)")
        print(f"Problems slower:     {summary.problems_slower} ({100*summary.problems_slower/summary.n_problems:.0f}%)")
        print(f"Problems same:       {summary.problems_same} ({100*summary.problems_same/summary.n_problems:.0f}%)")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuOpt configurations on real LP problems"
    )
    parser.add_argument(
        "--benchmark-dir", "-b",
        type=str,
        default="./benchmarks",
        help="Directory containing .mps benchmark files"
    )
    parser.add_argument(
        "--evolved-config", "-e",
        type=str,
        help="Path to evolved config JSON (from evolution results)"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of runs per problem (default: 3)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30000,
        help="Timeout per solve in ms (default: 30000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Load evolved config if provided
    evolved_config = None
    if args.evolved_config:
        evolved_config = load_evolved_config(args.evolved_config)

    # Run benchmark
    summary = run_benchmark(
        benchmark_dir=args.benchmark_dir,
        evolved_config=evolved_config,
        n_runs=args.runs,
        timeout_ms=args.timeout,
        verbose=not args.quiet,
    )

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
