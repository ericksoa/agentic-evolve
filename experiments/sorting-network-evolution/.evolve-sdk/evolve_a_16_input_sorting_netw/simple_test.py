import sys
sys.path.append('mutations')
exec(open('mutations/gen4x.py').read())

# Simple test
test_input = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
result = test_input.copy()

for i, j in NETWORK:
    if result[i] > result[j]:
        result[i], result[j] = result[j], result[i]

expected = sorted(test_input)
print(f"Input: {test_input}")
print(f"Expected: {expected}")
print(f"Result: {result}")
print(f"Correct: {result == expected}")
print(f"Network size: {len(NETWORK)}")