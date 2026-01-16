#!/usr/bin/env python3
"""
demo.py - Complete demonstration of the cuOpt LP AutoTuner.

This script demonstrates the full pipeline:
1. Create reproducible LP benchmark problems
2. Run evolution to find optimal cuOpt configuration
3. Benchmark baseline vs evolved on real problems
4. Show detailed comparison results

Usage:
    python3 -m showcase.cuopt_lp_autotuner.demo

For a quick demo:
    python3 -m showcase.cuopt_lp_autotuner.demo --quick
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from showcase.cuopt_lp_autotuner.create_benchmarks import create_all_benchmarks, BENCHMARK_SUITE
from showcase.cuopt_lp_autotuner.evolution import CuOptEvolution, EvolutionConfig, LPCorpus
from showcase.cuopt_lp_autotuner.run_benchmark import run_benchmark, load_evolved_config
from showcase.cuopt_lp_autotuner.cuopt_solver import is_cuopt_available


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def run_demo(
    quick: bool = False,
    output_dir: str = ".evolve-sdk/cuopt_demo",
    verbose: bool = True,
):
    """
    Run the complete AutoTuner demonstration.

    Args:
        quick: Run a faster demo with fewer problems/generations
        output_dir: Directory for outputs
        verbose: Print detailed progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_path / "benchmarks"

    print_header("cuOpt LP AutoTuner - Complete Demonstration")

    print(f"Backend: {'NVIDIA cuOpt (GPU)' if is_cuopt_available() else 'scipy HiGHS (CPU fallback)'}")
    print(f"Mode: {'Quick demo' if quick else 'Full demo'}")
    print(f"Output: {output_path}")
    print()

    # Step 1: Create benchmarks
    print_header("Step 1: Create LP Benchmark Problems")

    print("Creating reproducible LP problems from real-world domains:")
    print("  - Transportation and logistics")
    print("  - Production planning")
    print("  - Network flow optimization")
    print("  - Portfolio optimization")
    print("  - Resource allocation")
    print()

    create_all_benchmarks(str(benchmark_dir), verbose=True)

    # Step 2: Run evolution
    print_header("Step 2: Evolve Optimal cuOpt Configuration")

    # Configure evolution
    if quick:
        n_synthetic = 50
        population = 10
        generations = 5
    else:
        n_synthetic = 100
        population = 16
        generations = 15

    print(f"Evolving with:")
    print(f"  - {n_synthetic} synthetic LP problems (train/val/test split)")
    print(f"  - Population size: {population}")
    print(f"  - Max generations: {generations}")
    print()

    # Create corpus
    corpus = LPCorpus(seed=42)
    corpus.generate_synthetic(
        n_train=int(n_synthetic * 0.6),
        n_val=int(n_synthetic * 0.2),
        n_test=int(n_synthetic * 0.2),
    )

    # Run evolution
    config = EvolutionConfig(
        population_size=population,
        max_generations=generations,
        verbose=True,
    )

    evolution = CuOptEvolution(corpus, config, seed=42)
    best = evolution.run()

    # Save results
    results_path = output_path / "evolution_results.json"
    evolution.save_results(str(results_path))

    # Show evolved configuration
    print()
    print("Evolved Configuration:")
    print(f"  Method: {best.config.method.name}")
    print(f"  PDLP Mode: {best.config.pdlp_solver_mode.value}")
    print(f"  Presolve: {best.config.presolve}")
    print(f"  Crossover: {best.config.crossover}")
    print(f"  Primal tolerance: {best.config.absolute_primal_tolerance:.2e}")
    print(f"  Dual tolerance: {best.config.absolute_dual_tolerance:.2e}")
    print(f"  Gap tolerance: {best.config.absolute_gap_tolerance:.2e}")
    print()

    if best.train_fitness and best.test_fitness:
        print("Evolution Results:")
        print(f"  Train fitness: {best.train_fitness.fitness:.2f}")
        print(f"  Test fitness: {best.test_fitness.fitness:.2f}")
        print(f"  Test throughput: {best.test_fitness.throughput_score:+.1f}% vs baseline")
        print(f"  Test p95 latency: {best.test_fitness.p95_regression_pct:+.1f}% vs baseline")
        if not best.test_fitness.violates_p95_constraint:
            print("  ✓ Passes p95 constraint")
        else:
            print("  ✗ Violates p95 constraint")

    # Step 3: Benchmark comparison
    print_header("Step 3: Benchmark on Real LP Problems")

    print(f"Comparing baseline vs evolved on {len(list(benchmark_dir.glob('*.json')))} benchmark problems...")
    print()

    summary = run_benchmark(
        benchmark_dir=str(benchmark_dir),
        evolved_config=best.config,
        n_runs=3 if quick else 5,
        verbose=True,
    )

    # Save benchmark results
    benchmark_results_path = output_path / "benchmark_results.json"
    with open(benchmark_results_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Final summary
    print_header("Final Summary")

    print("The cuOpt LP AutoTuner:")
    print()
    print("1. EVOLVED optimal solver configuration using genetic algorithm")
    print(f"   - Explored {population}×{generations} = {population*generations}+ configurations")
    print(f"   - Used train/val/test splits for anti-overfitting")
    print(f"   - Hard constraint: no >2% p95 regression on test set")
    print()
    print("2. BENCHMARKED on real LP problems from:")
    print("   - Transportation and logistics optimization")
    print("   - Multi-period production planning")
    print("   - Network flow problems")
    print("   - Portfolio optimization")
    print("   - Resource allocation")
    print()
    print("3. RESULTS:")
    print(f"   - Overall speedup: {summary.baseline_total_time_ms / summary.evolved_total_time_ms:.2f}x")
    print(f"   - Geometric mean: {summary.geomean_speedup:.2f}x")
    print(f"   - Problems faster: {summary.problems_faster}/{summary.n_problems}")
    print(f"   - Problems slower: {summary.problems_slower}/{summary.n_problems}")
    print()
    print(f"Results saved to: {output_path}")
    print()

    if is_cuopt_available():
        print("NOTE: Running with NVIDIA cuOpt GPU acceleration.")
        print("      Results show real cuOpt performance improvements.")
    else:
        print("NOTE: Running with scipy HiGHS (CPU fallback).")
        print("      With real cuOpt on GPU, expect larger speedups from")
        print("      optimized PDLP mode, tolerance, and method selection.")


def main():
    parser = argparse.ArgumentParser(
        description="Complete demonstration of the cuOpt LP AutoTuner"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run a quick demo (fewer problems, generations)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=".evolve-sdk/cuopt_demo",
        help="Output directory"
    )

    args = parser.parse_args()

    run_demo(
        quick=args.quick,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
