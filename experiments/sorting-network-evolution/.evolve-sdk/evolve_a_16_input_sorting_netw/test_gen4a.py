#!/usr/bin/env python3
import sys
sys.path.append('mutations')
from gen4a import NETWORK

# Test that network is generated
print(f'Network size: {len(NETWORK)}')

# Test that all pairs are valid (indices in range)
valid = True
for i, j in NETWORK:
    if i < 0 or j < 0 or i >= 16 or j >= 16 or i >= j:
        print(f'Invalid pair: ({i}, {j})')
        valid = False

if valid:
    print('All pairs are valid!')
else:
    print('Found invalid pairs!')
    sys.exit(1)

# Basic sorting test
def apply_network(arr):
    result = arr[:]
    for i, j in NETWORK:
        if result[i] > result[j]:
            result[i], result[j] = result[j], result[i]
    return result

# Test with reverse sorted array
test_input = list(range(15, -1, -1))
result = apply_network(test_input)
expected = list(range(16))

print(f'Input: {test_input}')
print(f'Output: {result}')
print(f'Expected: {expected}')
print(f'Correctly sorted: {result == expected}')

if result == expected:
    print("SUCCESS: Network correctly sorts!")
else:
    print("FAILURE: Network does not sort correctly!")
    sys.exit(1)