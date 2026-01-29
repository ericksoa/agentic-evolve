import sys
sys.path.append('mutations')
exec(open('mutations/gen8b.py').read())

# Test with multiple inputs
test_inputs = [
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # Reverse sorted
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Already sorted
    [8, 3, 15, 1, 12, 7, 0, 9, 5, 11, 2, 14, 6, 10, 4, 13],  # Random
    [0] * 16,  # All same
    [i % 4 for i in range(16)]  # Repeated pattern
]

all_correct = True

for i, test_input in enumerate(test_inputs):
    result = test_input.copy()

    for i, j in NETWORK:
        if result[i] > result[j]:
            result[i], result[j] = result[j], result[i]

    expected = sorted(test_input)
    correct = result == expected
    all_correct &= correct

    print(f"Test {i+1}: {'PASS' if correct else 'FAIL'}")
    if not correct:
        print(f"  Input: {test_input}")
        print(f"  Expected: {expected}")
        print(f"  Result: {result}")

print(f"\nAll tests passed: {all_correct}")
print(f"Network size: {len(NETWORK)}")