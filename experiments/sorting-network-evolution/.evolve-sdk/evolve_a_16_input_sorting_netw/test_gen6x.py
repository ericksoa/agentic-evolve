#!/usr/bin/env python3

import sys
sys.path.append('mutations')
from gen6x import NETWORK

def test_network():
    print(f'Network size: {len(NETWORK)}')

    # Verify all comparisons are valid
    valid = True
    for i, (a, b) in enumerate(NETWORK):
        if not (0 <= a < 16 and 0 <= b < 16 and a < b):
            print(f'Invalid comparison at index {i}: ({a}, {b})')
            valid = False

    if valid:
        print('All comparisons are valid!')
    else:
        print('Found invalid comparisons!')
        return False

    # Test network can actually sort
    def apply_network(arr, network):
        arr = arr[:]
        for i, j in network:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        return arr

    # Test with reverse sorted
    test = list(range(16, 0, -1))
    result = apply_network(test, NETWORK)
    expected = list(range(16))

    if result == expected:
        print('Sorting test: PASSED')
        return True
    else:
        print('Sorting test: FAILED')
        return False

if __name__ == "__main__":
    success = test_network()
    sys.exit(0 if success else 1)