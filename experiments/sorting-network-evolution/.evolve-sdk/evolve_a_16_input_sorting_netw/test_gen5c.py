import sys
sys.path.append('mutations')
exec(open('mutations/gen5c.py').read())

# Test multiple inputs to verify correctness
test_cases = [
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # Reverse sorted
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],   # Already sorted
    [8, 3, 15, 1, 12, 7, 0, 9, 14, 2, 11, 6, 13, 4, 10, 5],   # Random
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],         # All same
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]          # Duplicates
]

all_correct = True

for i, test_input in enumerate(test_cases):
    result = test_input.copy()

    for a, b in NETWORK:
        if result[a] > result[b]:
            result[a], result[b] = result[b], result[a]

    expected = sorted(test_input)
    correct = result == expected
    all_correct &= correct

    print(f"Test {i+1}: {'PASS' if correct else 'FAIL'}")
    if not correct:
        print(f"  Input: {test_input}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")

print(f"\nOverall: {'ALL TESTS PASS' if all_correct else 'SOME TESTS FAILED'}")
print(f"Network size: {len(NETWORK)}")