#!/usr/bin/env python3

import sys
sys.path.append('mutations')

from gen5x import NETWORK

def apply_network(arr, network):
    """Apply sorting network to array."""
    arr = arr.copy()
    for i, j in network:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    return arr

def test_network():
    """Test the hybrid network for correctness."""
    print(f"Network size: {len(NETWORK)}")
    print(f"Sample comparisons: {NETWORK[:5]}")

    # Test cases
    test_cases = [
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse
        [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 16],   # Alternating
        [8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9],   # Two halves
        list(range(1, 17)),  # Already sorted
        [1] * 16  # All same
    ]

    all_passed = True
    for i, test in enumerate(test_cases):
        result = apply_network(test, NETWORK)
        expected = sorted(test)
        passed = result == expected
        all_passed = all_passed and passed
        print(f"Test {i+1}: {'PASS' if passed else 'FAIL'}")

    print(f"All tests passed: {all_passed}")
    return all_passed

if __name__ == "__main__":
    test_network()