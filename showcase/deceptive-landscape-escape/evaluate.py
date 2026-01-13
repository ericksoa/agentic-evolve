#!/usr/bin/env python3
"""Evaluate solutions for the deceptive landscape problem.

The fitness function is designed to be DECEPTIVE:
- Local optimum: High density of 1s leads to fitness ~75
- Global optimum: Alternating pattern (101010...) achieves fitness 100

Without diversity monitoring, evolution tends to converge on the local
optimum because mutations that add more 1s appear beneficial until
the population gets stuck.

Usage:
    python evaluate.py solution.py --json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_solution(path: str):
    """Load a solution module from file path."""
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deceptive_fitness(bitstring: str) -> float:
    """Calculate fitness for a bit string using a deceptive function.

    This function has two attractors:
    1. Local optimum (~75): Blocks of 1s with some structure (greedy trap)
    2. Global optimum (100): Perfect alternating pattern "101010..."

    The deception: Increasing 1-density with local structure leads to ~75,
    but reaching 100 requires the non-obvious perfect alternation.

    Args:
        bitstring: A string of '0's and '1's

    Returns:
        Fitness score (0-100)
    """
    if not bitstring or not all(c in "01" for c in bitstring):
        return 0.0

    n = len(bitstring)
    if n == 0:
        return 0.0

    # Component 1: Reward for 1-density (the trap)
    # This makes "add more 1s" look like progress
    ones_count = bitstring.count("1")
    ones_ratio = ones_count / n
    density_score = ones_ratio * 50  # Max 50 points

    # Component 2: Reward for LOCAL structure (pairs/blocks)
    # This creates the trap: "11001100" patterns score well
    pair_score = 0
    for i in range(0, n - 1, 2):
        if i + 1 < n:
            # Reward matching pairs (00 or 11)
            if bitstring[i] == bitstring[i + 1]:
                pair_score += 1
    max_pairs = n // 2
    if max_pairs > 0:
        pair_normalized = (pair_score / max_pairs) * 25  # Max 25 points
    else:
        pair_normalized = 0

    # Total for local optimum path
    fitness = density_score + pair_normalized  # Max ~75

    # Check for perfect alternation (the global optimum)
    perfect_alt_0 = "".join("01"[i % 2] for i in range(n))
    perfect_alt_1 = "".join("10"[i % 2] for i in range(n))

    if bitstring == perfect_alt_0 or bitstring == perfect_alt_1:
        fitness = 100.0  # Perfect score for global optimum

    # Near-perfect alternation bonus
    alternation_count = sum(1 for i in range(n - 1) if bitstring[i] != bitstring[i + 1])
    if alternation_count == n - 1 and fitness < 100:
        # Perfect alternation but wrong start/length
        fitness = 95.0

    return round(fitness, 2)


def analyze_solution(bitstring: str) -> dict:
    """Analyze a solution to understand its structure."""
    n = len(bitstring)
    ones = bitstring.count("1")
    zeros = bitstring.count("0")

    # Count alternations
    alternations = sum(1 for i in range(n - 1) if bitstring[i] != bitstring[i + 1])

    # Detect patterns
    is_all_ones = bitstring == "1" * n
    is_all_zeros = bitstring == "0" * n
    is_alternating = alternations == n - 1

    return {
        "length": n,
        "ones": ones,
        "zeros": zeros,
        "ones_ratio": round(ones / n, 3) if n > 0 else 0,
        "alternations": alternations,
        "max_alternations": n - 1,
        "alternation_ratio": round(alternations / (n - 1), 3) if n > 1 else 0,
        "is_all_ones": is_all_ones,
        "is_all_zeros": is_all_zeros,
        "is_perfect_alternating": is_alternating,
    }


def evaluate_solution(path: Path) -> dict:
    """Evaluate a single solution file."""
    try:
        module = load_solution(str(path))

        if not hasattr(module, "solve"):
            return {"valid": False, "fitness": 0, "error": "No solve() function found"}

        result = module.solve()

        if not isinstance(result, str):
            return {"valid": False, "fitness": 0, "error": f"solve() returned {type(result)}, expected str"}

        fitness = deceptive_fitness(result)
        analysis = analyze_solution(result)

        return {
            "valid": True,
            "fitness": fitness,
            "solution": result if len(result) <= 50 else result[:50] + "...",
            "analysis": analysis,
        }

    except Exception as e:
        return {"valid": False, "fitness": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate deceptive landscape solutions")
    parser.add_argument("solution", help="Path to solution file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    result = evaluate_solution(Path(args.solution))

    if args.json:
        print(json.dumps(result))
    else:
        if result["valid"]:
            print(f"Fitness: {result['fitness']}")
            if args.verbose:
                print(f"Solution: {result.get('solution', 'N/A')}")
                analysis = result.get("analysis", {})
                print(f"  Ones ratio: {analysis.get('ones_ratio', 'N/A')}")
                print(f"  Alternation ratio: {analysis.get('alternation_ratio', 'N/A')}")
                if analysis.get("is_perfect_alternating"):
                    print("  ** GLOBAL OPTIMUM FOUND **")
                elif analysis.get("ones_ratio", 0) > 0.7:
                    print("  (Trapped in local optimum - high 1-density)")
        else:
            print(f"Invalid: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
