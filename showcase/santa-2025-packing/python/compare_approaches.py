#!/usr/bin/env python3
"""
Compare NFP-based packing with Rust greedy approach.
"""

import subprocess
import sys
import time
sys.path.insert(0, '/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/santa-2025-packing/python')

from nfp_packer import NFPCache, greedy_nfp_pack, validate_packing, compute_side_length

def run_rust_for_n(n: int) -> float:
    """Run Rust solver for a single n value and return side length."""
    # We'll use the ultimate_submission binary which outputs side lengths
    # For now, use the exported packing approach
    result = subprocess.run(
        ['./target/release/ultimate_submission', str(n), str(n), '/dev/null'],
        capture_output=True, text=True,
        cwd='/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/santa-2025-packing/rust'
    )

    # Parse output for side length - look for lines like "n=5: side=1.234"
    for line in result.stdout.split('\n'):
        if f'n={n}:' in line and 'side=' in line:
            # Extract side length
            parts = line.split('side=')
            if len(parts) > 1:
                try:
                    return float(parts[1].split()[0].strip(','))
                except:
                    pass

    return float('inf')


def compare_n_range(start_n: int, end_n: int, nfp_cache: NFPCache):
    """Compare NFP vs Rust for a range of n values."""
    print(f"\n{'n':>4} | {'NFP':>10} | {'Rust':>10} | {'Diff':>10} | {'Winner':>8}")
    print("-" * 55)

    nfp_total = 0.0
    rust_total = 0.0
    nfp_wins = 0
    rust_wins = 0

    for n in range(start_n, end_n + 1):
        # NFP packing
        nfp_side, trees = greedy_nfp_pack(n, nfp_cache)
        valid, overlaps = validate_packing(trees)
        if not valid:
            nfp_side = float('inf')
            print(f"  [NFP INVALID for n={n}: {overlaps} overlaps]")

        # Rust packing - run the solver
        rust_side = run_rust_for_n(n)

        # Compare
        nfp_score = nfp_side**2 / n if nfp_side < float('inf') else float('inf')
        rust_score = rust_side**2 / n if rust_side < float('inf') else float('inf')

        diff = nfp_side - rust_side if both_valid(nfp_side, rust_side) else float('inf')
        winner = 'NFP' if nfp_side < rust_side else ('Rust' if rust_side < nfp_side else 'Tie')

        if nfp_side < rust_side:
            nfp_wins += 1
        elif rust_side < nfp_side:
            rust_wins += 1

        nfp_total += nfp_score
        rust_total += rust_score

        print(f"{n:>4} | {nfp_side:>10.4f} | {rust_side:>10.4f} | {diff:>+10.4f} | {winner:>8}")

    print("-" * 55)
    print(f"\nNFP total score:  {nfp_total:.4f}")
    print(f"Rust total score: {rust_total:.4f}")
    print(f"NFP wins: {nfp_wins}, Rust wins: {rust_wins}")


def both_valid(a, b):
    return a < float('inf') and b < float('inf')


def main():
    """Run comparison."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1, help='Start n')
    parser.add_argument('--end', type=int, default=10, help='End n')
    parser.add_argument('--angles', type=int, default=8, help='NFP rotation angles')

    args = parser.parse_args()

    print("Initializing NFP cache...")
    nfp_cache = NFPCache(angle_steps=args.angles)

    compare_n_range(args.start, args.end, nfp_cache)


if __name__ == '__main__':
    main()
