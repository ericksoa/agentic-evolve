#!/usr/bin/env python3
import sys
sys.path.append('mutations')
from gen4x import NETWORK
import random

def test_sorting_network(network, n=16):
    """Test if the network correctly sorts all possible inputs"""

    # Test a representative sample for large n
    test_cases = []

    # Add specific test cases
    test_cases.append(list(range(n)))  # Already sorted
    test_cases.append(list(range(n-1, -1, -1)))  # Reverse sorted
    test_cases.append([0] * n)  # All same
    test_cases.append([i % 3 for i in range(n)])  # Pattern

    # Add random permutations
    for _ in range(100):
        test_case = list(range(n))
        random.shuffle(test_case)
        test_cases.append(test_case)

    failures = 0
    for test_input in test_cases:
        result = test_input.copy()

        # Apply network
        for i, j in network:
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]

        if result != sorted(test_input):
            failures += 1
            if failures <= 3:  # Show first few failures
                print(f'FAILED on input: {test_input}')
                print(f'Expected: {sorted(test_input)}')
                print(f'Got: {result}')

    return failures == 0, failures

# Test the network
success, failures = test_sorting_network(NETWORK, 16)
print(f'Network correctness test: {"PASSED" if success else f"FAILED ({failures} failures)"}')
print(f'Network size: {len(NETWORK)} comparisons')

# Also print network for verification
print(f'First 10 comparisons: {NETWORK[:10]}')
print(f'Last 10 comparisons: {NETWORK[-10:]}')