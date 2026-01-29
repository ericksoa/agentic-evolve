#!/usr/bin/env python3

# Import the mutated solution
import sys
sys.path.append('.evolve-sdk/evolve_a_16_input_sorting_netw/mutations')
import gen4c

def test_sorting_network(network, test_input):
    """Test if network correctly sorts the input."""
    data = test_input[:]

    for i, j in network:
        if data[i] > data[j]:
            data[i], data[j] = data[j], data[i]

    return data

def is_sorted(arr):
    """Check if array is sorted."""
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

# Test with a few different inputs
test_cases = [
    [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse sorted
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Already sorted
    [8, 3, 15, 1, 12, 7, 4, 11, 2, 16, 9, 6, 13, 5, 10, 14],  # Random
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],        # All same
]

network = gen4c.NETWORK
print(f"Network has {len(network)} comparisons")

all_passed = True
for i, test_case in enumerate(test_cases):
    result = test_sorting_network(network, test_case)
    passed = is_sorted(result)
    print(f"Test {i+1}: {'PASS' if passed else 'FAIL'} - {test_case[:5]}... -> {result[:5]}...")
    if not passed:
        all_passed = False

print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")