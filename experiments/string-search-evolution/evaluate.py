#!/usr/bin/env python3
"""
String Search Evaluator
=======================

Evaluates string search implementations for:
1. Correctness - must find all occurrences
2. Performance - searches per second

Usage:
    python evaluate.py <solution.py> [--json]
"""

import argparse
import importlib.util
import json
import random
import string
import sys
import time
from pathlib import Path


def generate_test_cases(seed: int = 42) -> list[tuple[str, str, list[int]]]:
    """Generate test cases with known answers."""
    random.seed(seed)
    cases = []

    # Case 1: Simple repeating pattern
    text1 = "AABAACAADAABAAABAA"
    pattern1 = "AABA"
    cases.append((text1, pattern1, [0, 9, 13]))

    # Case 2: No matches
    text2 = "ABCDEFGHIJKLMNOP"
    pattern2 = "XYZ"
    cases.append((text2, pattern2, []))

    # Case 3: Overlapping matches
    text3 = "AAAAA"
    pattern3 = "AA"
    cases.append((text3, pattern3, [0, 1, 2, 3]))

    # Case 4: Pattern at boundaries
    text4 = "PATTERNMIDDLEPATTERN"
    pattern4 = "PATTERN"
    cases.append((text4, pattern4, [0, 13]))

    # Case 5: Single character
    text5 = "ABCABC"
    pattern5 = "B"
    cases.append((text5, pattern5, [1, 4]))

    # Case 6: Worst case for naive (many partial matches)
    text6 = "A" * 100 + "B"
    pattern6 = "A" * 10 + "B"
    cases.append((text6, pattern6, [90]))

    # Case 7: DNA-like sequence
    dna = "".join(random.choices("ACGT", k=1000))
    dna_pattern = "ACGT"
    # Find actual positions
    dna_positions = []
    for i in range(len(dna) - len(dna_pattern) + 1):
        if dna[i : i + len(dna_pattern)] == dna_pattern:
            dna_positions.append(i)
    cases.append((dna, dna_pattern, dna_positions))

    # Case 8: English-like text
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    eng_text = " ".join(words * 50)
    eng_pattern = "the"
    eng_positions = []
    for i in range(len(eng_text) - len(eng_pattern) + 1):
        if eng_text[i : i + len(eng_pattern)] == eng_pattern:
            eng_positions.append(i)
    cases.append((eng_text, eng_pattern, eng_positions))

    return cases


def generate_benchmark_data(
    text_size: int = 100000, pattern_size: int = 20, seed: int = 42
) -> tuple[str, str]:
    """Generate large random text and pattern for benchmarking."""
    random.seed(seed)
    # Use limited alphabet to create more potential matches
    alphabet = "ABCD"
    text = "".join(random.choices(alphabet, k=text_size))
    pattern = "".join(random.choices(alphabet, k=pattern_size))
    return text, pattern


def load_solution(path: str):
    """Load a solution module from path."""
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_correctness(search_func, test_cases: list) -> tuple[bool, str]:
    """Verify solution produces correct results."""
    for i, (text, pattern, expected) in enumerate(test_cases):
        try:
            result = search_func(text, pattern)
            if sorted(result) != sorted(expected):
                return False, f"Case {i + 1}: expected {expected}, got {result}"
        except Exception as e:
            return False, f"Case {i + 1}: exception {e}"
    return True, "All tests passed"


def benchmark(search_func, iterations: int = 100) -> float:
    """Benchmark search function, return searches per second."""
    text, pattern = generate_benchmark_data()

    # Warmup
    for _ in range(5):
        search_func(text, pattern)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        search_func(text, pattern)
    elapsed = time.perf_counter() - start

    return iterations / elapsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate string search solution")
    parser.add_argument("solution", help="Path to solution file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations"
    )
    args = parser.parse_args()

    # Load solution
    try:
        module = load_solution(args.solution)
        search_func = module.search
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "valid": False}))
        else:
            print(f"Error loading solution: {e}")
        sys.exit(1)

    # Verify correctness
    test_cases = generate_test_cases()
    correct, message = verify_correctness(search_func, test_cases)

    if not correct:
        if args.json:
            print(json.dumps({"error": message, "valid": False, "fitness": 0}))
        else:
            print(f"FAILED: {message}")
        sys.exit(1)

    # Benchmark
    ops_per_sec = benchmark(search_func, args.iterations)

    if args.json:
        print(
            json.dumps(
                {
                    "valid": True,
                    "fitness": ops_per_sec,
                    "metric": "searches/sec",
                    "message": message,
                }
            )
        )
    else:
        print(f"Correctness: PASSED")
        print(f"Performance: {ops_per_sec:,.1f} searches/sec")


if __name__ == "__main__":
    main()
