"""
evaluate_population.py - Test and evaluate all initial population solutions.
"""

import json
import sys
import importlib.util
from pathlib import Path
import traceback

# Add project root and cuopt_lp_autotuner to path
project_root = Path(__file__).parent.parent.parent.parent
cuopt_dir = Path(__file__).parent.parent.parent.parent / "showcase" / "cuopt_lp_autotuner"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(cuopt_dir.parent))

from cuopt_lp_autotuner.policy import Policy
from cuopt_lp_autotuner.benchmark import Benchmark
from cuopt_lp_autotuner.lp_generator import create_corpus


def load_solution(filepath: Path):
    """Dynamically load a solution module."""
    spec = importlib.util.spec_from_file_location("solution", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_solution(filepath: Path, benchmark: Benchmark):
    """Evaluate a single solution and return results."""
    try:
        # Load the solution module
        module = load_solution(filepath)

        # Get the policy
        policy = module.get_policy()

        # Compute fitness on train split
        fitness_result = benchmark.compute_fitness(policy, split='train')

        return {
            "file": str(filepath),
            "name": policy.name,
            "valid": True,
            "fitness": fitness_result.primary_fitness,
            "throughput_score": fitness_result.throughput_score,
            "latency_score": fitness_result.latency_score,
            "p95_regression_pct": fitness_result.p95_regression_pct,
            "violates_constraint": fitness_result.violates_p95_constraint,
            "metrics": {
                "throughput_lps_per_sec": fitness_result.metrics.throughput_lps_per_sec,
                "latency_p95_ms": fitness_result.metrics.latency_p95_ms,
                "success_rate": fitness_result.metrics.success_rate,
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "file": str(filepath),
            "name": filepath.stem,
            "valid": False,
            "fitness": -float('inf'),
            "error": str(e),
        }


def main():
    print("Creating LP corpus...")
    corpus = create_corpus(n_train=30, n_valid=5, n_test=10, seed=42)
    print(f"  Train: {len(corpus['train'])} LPs")

    print("\nCreating benchmark...")
    benchmark = Benchmark(corpus, verbose=False)

    # Find all gen0_* solutions
    mutations_dir = Path(__file__).parent / "mutations"
    solution_files = sorted(mutations_dir.glob("gen0_*.py"))

    print(f"\nFound {len(solution_files)} solutions to evaluate\n")

    results = []
    best_result = None
    best_fitness = -float('inf')

    for filepath in solution_files:
        print(f"Evaluating {filepath.name}...", end=" ", flush=True)
        result = evaluate_solution(filepath, benchmark)
        results.append(result)

        if result["valid"]:
            print(f"fitness={result['fitness']:.2f}, throughput={result['metrics']['throughput_lps_per_sec']:.2f} LPs/s")
            if result["fitness"] > best_fitness:
                best_fitness = result["fitness"]
                best_result = result
        else:
            print(f"INVALID: {result.get('error', 'Unknown error')}")

    # Print summary
    print("\n" + "="*60)
    print("POPULATION SUMMARY")
    print("="*60)

    valid_count = sum(1 for r in results if r["valid"])
    print(f"\nValid solutions: {valid_count}/{len(results)}")

    if best_result:
        print(f"\nBest solution: {best_result['name']}")
        print(f"  Fitness: {best_result['fitness']:.2f}")
        print(f"  Throughput: {best_result['metrics']['throughput_lps_per_sec']:.2f} LPs/sec")
        print(f"  P95 Latency: {best_result['metrics']['latency_p95_ms']:.2f} ms")

    # Output JSON result
    output = {
        "solutions": [
            {
                "file": r["file"],
                "approach": r.get("name", "unknown"),
                "fitness": r["fitness"] if r["valid"] else None,
                "valid": r["valid"],
            }
            for r in results
        ],
        "best": {
            "file": best_result["file"] if best_result else None,
            "fitness": best_fitness if best_result else None,
        },
        "diversity_notes": "8 solutions covering: conservative baseline, aggressive transforms, speed-optimized, precision-optimized, size-aware batching, density-aware batching, scaling-heavy equilibration, minimal transforms with bound tightening"
    }

    print("\n" + "="*60)
    print("JSON OUTPUT")
    print("="*60)
    print(json.dumps(output, indent=2))

    return output


if __name__ == "__main__":
    main()
