#!/usr/bin/env python3
"""
Gen114 Runner - CMA-ES optimization for small n values.
"""

import json
import sys
import time
sys.path.insert(0, 'python')

from cmaes_optimizer import Tree, optimize_with_cmaes, compute_side_length, has_any_overlap

def load_current_best():
    """Load current best solutions."""
    with open('python/optimized_small_n.json') as f:
        return json.load(f)

def run_optimization(n_values, max_evals=3000, sigma=0.15):
    """Run CMA-ES optimization for specified n values."""
    current_best = load_current_best()
    results = {}
    improvements = []

    for n in n_values:
        n_str = str(n)
        if n_str not in current_best:
            print(f"No current solution for n={n}, skipping")
            continue

        # Load current solution
        current = current_best[n_str]
        current_side = current['side']
        trees_data = current['trees']

        # Convert to Tree objects
        initial_trees = [Tree(t[0], t[1], t[2]) for t in trees_data]

        print(f"\n{'='*60}")
        print(f"Optimizing n={n}")
        print(f"Current best side: {current_side:.6f}")
        print(f"Current s²/n: {current_side**2 / n:.6f}")

        start = time.time()
        best_side, best_trees = optimize_with_cmaes(
            initial_trees,
            max_evals=max_evals,
            sigma0=sigma,
            penalty_weight=200.0,
            verbose=True
        )
        elapsed = time.time() - start

        # Verify no overlaps
        if has_any_overlap(best_trees):
            print(f"  WARNING: Result has overlaps! Keeping original.")
            best_side = current_side
            best_trees = initial_trees

        # Store result
        results[n_str] = {
            'side': best_side,
            'trees': [[t.x, t.y, t.angle] for t in best_trees]
        }

        if best_side < current_side:
            improvement = (current_side - best_side) / current_side * 100
            score_delta = (current_side**2 - best_side**2) / n
            improvements.append({
                'n': n,
                'old_side': current_side,
                'new_side': best_side,
                'improvement_pct': improvement,
                'score_delta': score_delta
            })
            print(f"  IMPROVED: {current_side:.6f} -> {best_side:.6f} ({improvement:.2f}%)")
            print(f"  Score delta: {score_delta:.6f}")
        else:
            print(f"  No improvement")

        print(f"  Elapsed: {elapsed:.1f}s")

    return results, improvements

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 9, 10],
                       help='n values to optimize')
    parser.add_argument('--evals', type=int, default=3000,
                       help='Max evaluations per n')
    parser.add_argument('--sigma', type=float, default=0.15,
                       help='Initial step size')
    parser.add_argument('--output', type=str, default='python/gen114_optimized.json',
                       help='Output file')
    args = parser.parse_args()

    print("Gen114: CMA-ES Optimization")
    print(f"n values: {args.n}")
    print(f"Max evals: {args.evals}")
    print(f"Sigma: {args.sigma}")

    results, improvements = run_optimization(args.n, args.evals, args.sigma)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if improvements:
        total_score_delta = sum(i['score_delta'] for i in improvements)
        print(f"Improvements found: {len(improvements)}")
        for imp in improvements:
            print(f"  n={imp['n']}: {imp['old_side']:.4f} -> {imp['new_side']:.4f} "
                  f"({imp['improvement_pct']:.2f}%, score delta: {imp['score_delta']:.4f})")
        print(f"\nTotal score improvement: {total_score_delta:.4f}")
    else:
        print("No improvements found.")

if __name__ == '__main__':
    main()
