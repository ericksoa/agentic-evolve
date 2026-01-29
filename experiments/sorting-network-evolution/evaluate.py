#!/usr/bin/env python3
"""
Sorting Network Evaluator

Evaluates sorting networks for:
1. Correctness - Must sort all possible inputs (or large sample for bigger n)
2. Comparator count - Fewer is better (area/gates)
3. Depth - Fewer stages is better (latency)

A sorting network is represented as a list of (i, j) tuples where i < j.
Each comparator swaps positions i and j if out of order.

Known optimal comparator counts:
- n=4: 5 comparators
- n=5: 9 comparators
- n=6: 12 comparators
- n=8: 19 comparators
- n=16: 60 comparators (best known)
"""

import json
import sys
import itertools
import random
from pathlib import Path
from typing import List, Tuple


def apply_network(network: List[Tuple[int, int]], arr: List[int]) -> List[int]:
    """Apply sorting network to an array."""
    result = arr.copy()
    for i, j in network:
        if result[i] > result[j]:
            result[i], result[j] = result[j], result[i]
    return result


def is_sorted(arr: List[int]) -> bool:
    """Check if array is sorted."""
    return all(arr[i] <= arr[i+1] for i in range(len(arr) - 1))


def compute_depth(network: List[Tuple[int, int]], n: int) -> int:
    """
    Compute the depth (number of parallel stages) of a sorting network.
    Comparators that don't share indices can be parallelized.
    """
    if not network:
        return 0

    # Track when each wire becomes available (after which stage)
    wire_available = [0] * n

    for i, j in network:
        # This comparator must wait for both wires to be ready
        stage = max(wire_available[i], wire_available[j]) + 1
        wire_available[i] = stage
        wire_available[j] = stage

    return max(wire_available)


def verify_correctness(network: List[Tuple[int, int]], n: int, exhaustive_limit: int = 10) -> Tuple[bool, str]:
    """
    Verify that a sorting network correctly sorts all inputs.

    Uses the zero-one principle: a comparator network sorts all sequences
    iff it sorts all 2^n binary sequences.

    For n <= exhaustive_limit, test exhaustively. Otherwise, sample.
    """
    if n <= exhaustive_limit:
        # Exhaustive test using zero-one principle
        for bits in range(2**n):
            arr = [(bits >> i) & 1 for i in range(n)]
            result = apply_network(network, arr)
            if not is_sorted(result):
                return False, f"Failed on binary input: {arr}"
        return True, "Passed all 2^n binary tests"
    else:
        # Sample-based testing for larger n
        # Test all binary sequences up to a limit
        sample_size = min(2**n, 100000)
        for _ in range(sample_size):
            bits = random.randint(0, 2**n - 1)
            arr = [(bits >> i) & 1 for i in range(n)]
            result = apply_network(network, arr)
            if not is_sorted(result):
                return False, f"Failed on binary input: {arr}"

        # Also test some random permutations
        for _ in range(10000):
            arr = list(range(n))
            random.shuffle(arr)
            result = apply_network(network, arr)
            if not is_sorted(result):
                return False, f"Failed on permutation: {arr}"

        return True, f"Passed {sample_size} binary + 10000 permutation tests"


def validate_network(network: List[Tuple[int, int]], n: int) -> Tuple[bool, str]:
    """Validate network structure."""
    if not isinstance(network, list):
        return False, "Network must be a list of tuples"

    for idx, comp in enumerate(network):
        if not isinstance(comp, (list, tuple)) or len(comp) != 2:
            return False, f"Comparator {idx} must be a (i, j) tuple"
        i, j = comp
        if not isinstance(i, int) or not isinstance(j, int):
            return False, f"Comparator {idx} indices must be integers"
        if i < 0 or j < 0 or i >= n or j >= n:
            return False, f"Comparator {idx} indices out of range [0, {n-1}]"
        if i >= j:
            return False, f"Comparator {idx}: must have i < j, got ({i}, {j})"

    return True, "Valid structure"


def load_solution(solution_path: str) -> Tuple[List[Tuple[int, int]], int]:
    """
    Load a sorting network solution from a Python file.

    The file must define:
    - N: int - number of inputs
    - NETWORK: List[Tuple[int, int]] - the sorting network
    """
    path = Path(solution_path)
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")

    # Execute the solution file to get N and NETWORK
    namespace = {}
    exec(path.read_text(), namespace)

    if 'N' not in namespace:
        raise ValueError("Solution must define N (number of inputs)")
    if 'NETWORK' not in namespace:
        raise ValueError("Solution must define NETWORK (list of comparators)")

    return namespace['NETWORK'], namespace['N']


def evaluate(solution_path: str) -> dict:
    """
    Evaluate a sorting network solution.

    Returns JSON with:
    - valid: bool - whether solution is correct
    - fitness: float - optimization score (higher is better)
    - comparators: int - number of comparators
    - depth: int - network depth (stages)
    - n: int - input size
    - error: str - error message if invalid
    """
    try:
        network, n = load_solution(solution_path)
    except Exception as e:
        return {
            "valid": False,
            "fitness": 0,
            "error": f"Failed to load solution: {e}"
        }

    # Validate structure
    valid_struct, msg = validate_network(network, n)
    if not valid_struct:
        return {
            "valid": False,
            "fitness": 0,
            "error": msg
        }

    # Verify correctness
    correct, correctness_msg = verify_correctness(network, n)
    if not correct:
        return {
            "valid": False,
            "fitness": 0,
            "n": n,
            "comparators": len(network),
            "error": f"Incorrect sorting: {correctness_msg}"
        }

    # Calculate metrics
    comparators = len(network)
    depth = compute_depth(network, n)

    # Fitness: we want FEWER comparators, so fitness = large_number - comparators
    # This makes higher fitness = better (fewer comparators)
    # We also add a small bonus for lower depth
    fitness = 1000 - comparators - (depth * 0.1)

    return {
        "valid": True,
        "fitness": fitness,
        "comparators": comparators,
        "depth": depth,
        "n": n,
        "correctness": correctness_msg
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <solution.py> [--json]", file=sys.stderr)
        sys.exit(1)

    solution_path = sys.argv[1]
    json_output = "--json" in sys.argv

    result = evaluate(solution_path)

    if json_output:
        print(json.dumps(result))
    else:
        if result["valid"]:
            print(f"Valid sorting network for n={result['n']}")
            print(f"  Comparators: {result['comparators']}")
            print(f"  Depth: {result['depth']}")
            print(f"  Fitness: {result['fitness']:.1f}")
            print(f"  {result['correctness']}")
        else:
            print(f"INVALID: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
