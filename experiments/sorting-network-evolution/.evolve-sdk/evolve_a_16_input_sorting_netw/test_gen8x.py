#!/usr/bin/env python3

import sys
sys.path.append('mutations')
from gen8x import NETWORK

def test_network(arr, network):
    arr = arr.copy()
    for i, j in network:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    return arr

# Test basic properties
print(f'Network length: {len(NETWORK)}')
print(f'First 10 comparisons: {NETWORK[:10]}')
print(f'Last 10 comparisons: {NETWORK[-10:]}')

# Verify all comparisons are valid
valid = all(0 <= a < b < 16 for a, b in NETWORK)
print(f'All comparisons valid: {valid}')

# Test cases
test_cases = [
    [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse sorted
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Already sorted
    [8, 3, 15, 1, 12, 6, 9, 4, 11, 2, 14, 7, 10, 5, 13, 16], # Random
]

print("\nTest results:")
for i, test in enumerate(test_cases):
    result = test_network(test, NETWORK)
    expected = sorted(test)
    success = result == expected
    print(f'Test {i+1}: {"PASS" if success else "FAIL"} - {result}')
    if not success:
        print(f'  Expected: {expected}')