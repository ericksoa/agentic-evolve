#!/usr/bin/env python3
"""
lightning_run.py - Run cuOpt LP AutoTuner on Lightning.ai GPU instances.

This script is designed to run on Lightning.ai Studios or Lightning AI cloud.
It handles:
1. Environment setup and dependency installation
2. cuOpt installation (if available)
3. Running evolution with GPU-optimized settings
4. Saving and downloading results

Usage (on Lightning.ai):
    # Quick test
    python lightning_run.py --quick

    # Full GPU evolution
    python lightning_run.py --full

    # With specific settings
    python lightning_run.py --generations 50 --population 32

Local testing (will use CPU fallback):
    python showcase/cuopt_lp_autotuner/lightning_run.py --quick --local
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, check: bool = True, capture: bool = False):
    """Run a shell command."""
    print(f"$ {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Command failed: {cmd}")
        return result.stdout
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
        return None


def check_gpu():
    """Check if GPU is available."""
    try:
        output = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture=True)
        gpus = output.strip().split('\n')
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        return len(gpus)
    except Exception as e:
        print(f"No GPU detected: {e}")
        return 0


def check_cuopt():
    """Check if cuOpt is available."""
    try:
        result = run_command("python3 -c \"import cuopt; print(f'cuOpt {cuopt.__version__}')\"", capture=True)
        print(f"cuOpt available: {result.strip()}")
        return True
    except Exception:
        print("cuOpt not installed")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n=== Installing Dependencies ===\n")

    # Core dependencies
    run_command("pip install -q scipy numpy", check=False)

    # Try to install cuOpt (may not be available in all environments)
    print("\nAttempting to install cuOpt...")

    # Method 1: pip (if NVIDIA makes it available)
    try:
        run_command("pip install -q nvidia-cuopt", check=True)
        print("Installed cuOpt via pip")
        return True
    except Exception:
        pass

    # Method 2: conda (Rapids AI channel)
    try:
        run_command("conda install -y -c rapidsai -c nvidia cuopt cuda-version=12.0", check=True)
        print("Installed cuOpt via conda")
        return True
    except Exception:
        pass

    # Method 3: Check if it's pre-installed (NGC containers)
    if check_cuopt():
        return True

    print("\nWARNING: Could not install cuOpt. Will use scipy HiGHS fallback.")
    print("For real GPU acceleration, use an NVIDIA NGC container or DGX Cloud.")
    return False


def setup_environment():
    """Set up the Python environment."""
    # Add the repo root to path
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Set environment variables for GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["CUOPT_LOG_LEVEL"] = "WARNING"  # Reduce cuOpt verbosity


def run_evolution(
    generations: int = 30,
    population: int = 24,
    synthetic_problems: int = 150,
    output_dir: str = None,
    verbose: bool = True,
):
    """Run the cuOpt evolution."""
    from showcase.cuopt_lp_autotuner.evolution import (
        CuOptEvolution, EvolutionConfig, LPCorpus
    )
    from showcase.cuopt_lp_autotuner.cuopt_solver import is_cuopt_available
    from showcase.cuopt_lp_autotuner.create_benchmarks import create_all_benchmarks

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f".evolve-sdk/cuopt_lightning_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  cuOpt LP AutoTuner - Lightning.ai GPU Evolution")
    print("=" * 70)
    print()
    print(f"Backend: {'NVIDIA cuOpt (GPU)' if is_cuopt_available() else 'scipy HiGHS (CPU fallback)'}")
    print(f"Output: {output_path}")
    print()
    print(f"Evolution settings:")
    print(f"  - Synthetic problems: {synthetic_problems}")
    print(f"  - Population size: {population}")
    print(f"  - Max generations: {generations}")
    print()

    # Create benchmark problems
    print("Creating benchmark problems...")
    benchmark_dir = output_path / "benchmarks"
    create_all_benchmarks(str(benchmark_dir), verbose=False)
    print(f"Created 15 benchmark problems in {benchmark_dir}")
    print()

    # Create corpus for evolution
    print("Generating synthetic LP corpus...")
    corpus = LPCorpus(seed=42)
    corpus.generate_synthetic(
        n_train=int(synthetic_problems * 0.6),
        n_val=int(synthetic_problems * 0.2),
        n_test=int(synthetic_problems * 0.2),
    )
    print(f"Generated {len(corpus.train)} train, {len(corpus.val)} val, {len(corpus.test)} test problems")
    print()

    # Configure evolution
    config = EvolutionConfig(
        population_size=population,
        max_generations=generations,
        elite_count=max(2, population // 8),
        tournament_size=max(3, population // 6),
        mutation_rate=0.25,
        crossover_rate=0.35,
        plateau_limit=10,
        verbose=verbose,
    )

    # Run evolution
    print("Starting evolution...")
    start_time = time.time()

    evolution = CuOptEvolution(corpus, config, seed=42)
    best = evolution.run()

    elapsed = time.time() - start_time
    print(f"\nEvolution completed in {elapsed/60:.1f} minutes")

    # Save results
    results_path = output_path / "evolution_results.json"
    evolution.save_results(str(results_path))

    # Save best config separately
    best_config_path = output_path / "best_config.json"
    best.config.save(str(best_config_path))

    # Run benchmark comparison
    print("\n" + "=" * 70)
    print("  Benchmark Comparison: Baseline vs Evolved")
    print("=" * 70 + "\n")

    from showcase.cuopt_lp_autotuner.run_benchmark import run_benchmark

    summary = run_benchmark(
        benchmark_dir=str(benchmark_dir),
        evolved_config=best.config,
        n_runs=5,
        verbose=True,
    )

    # Save benchmark results
    benchmark_results_path = output_path / "benchmark_results.json"
    with open(benchmark_results_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"Best configuration: {best.id}")
    print(f"  Method: {best.config.method.name}")
    print(f"  PDLP Mode: {best.config.pdlp_solver_mode.value}")
    print(f"  Presolve: {best.config.presolve}")
    print(f"  Crossover: {best.config.crossover}")
    print()

    if best.test_fitness:
        print(f"Evolution test fitness: {best.test_fitness.fitness:.2f}")
        print(f"  Throughput: {best.test_fitness.throughput_score:+.1f}% vs baseline")
        print(f"  p95 latency: {best.test_fitness.p95_regression_pct:+.1f}% vs baseline")
        if best.test_fitness.violates_p95_constraint:
            print("  ⚠️  Violates p95 constraint")
        else:
            print("  ✓ Passes p95 constraint")

    print()
    print(f"Benchmark results ({summary.n_problems} real LP problems):")
    print(f"  Overall speedup: {summary.baseline_total_time_ms / summary.evolved_total_time_ms:.2f}x")
    print(f"  Geometric mean: {summary.geomean_speedup:.2f}x")
    print(f"  Problems faster: {summary.problems_faster}/{summary.n_problems}")
    print(f"  Problems slower: {summary.problems_slower}/{summary.n_problems}")
    print()
    print(f"Results saved to: {output_path}")
    print()

    return best, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run cuOpt LP AutoTuner on Lightning.ai"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick run (fewer generations, smaller population)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Full run (more generations, larger population)"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=None,
        help="Number of generations"
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=None,
        help="Population size"
    )
    parser.add_argument(
        "--problems", "-n",
        type=int,
        default=None,
        help="Number of synthetic problems"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Skip GPU/cuOpt checks (for local testing)"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  cuOpt LP AutoTuner - Lightning.ai Runner")
    print("=" * 70)
    print()

    # Check environment
    if not args.local:
        print("=== Environment Check ===\n")
        n_gpus = check_gpu()

        if not args.skip_install:
            has_cuopt = install_dependencies()
        else:
            has_cuopt = check_cuopt()

        if n_gpus == 0:
            print("\n⚠️  No GPU detected. Results will use CPU fallback.")
            print("   For GPU acceleration, run on a Lightning.ai GPU Studio.")

        if not has_cuopt:
            print("\n⚠️  cuOpt not available. Using scipy HiGHS fallback.")
            print("   For real cuOpt, use NVIDIA NGC container or DGX Cloud.")

    # Set up environment
    setup_environment()

    # Determine settings
    if args.quick:
        generations = args.generations or 10
        population = args.population or 12
        problems = args.problems or 60
    elif args.full:
        generations = args.generations or 50
        population = args.population or 32
        problems = args.problems or 200
    else:
        generations = args.generations or 30
        population = args.population or 24
        problems = args.problems or 150

    # Run evolution
    run_evolution(
        generations=generations,
        population=population,
        synthetic_problems=problems,
        output_dir=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
