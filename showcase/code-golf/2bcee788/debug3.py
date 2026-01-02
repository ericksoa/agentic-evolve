import json

# Example 2 - vertical markers
task = json.loads('''{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 6, 3, 3, 3, 3], [3, 3, 3, 3, 6, 3, 3, 3, 3, 3], [3, 3, 3, 6, 6, 3, 3, 3, 3, 3], [3, 3, 3, 6, 6, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 6, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}''')

inp = task["input"]
out = task["output"]

print("Input:")
for i, r in enumerate(inp):
    print(f"{i}: {r}")

print("\nOutput:")
for i, r in enumerate(out):
    print(f"{i}: {r}")

print("\nMarkers (2):")
for r in range(10):
    for c in range(10):
        if inp[r][c] == 2:
            print(f"  ({r}, {c})")

print("\nShape (6):")
for r in range(10):
    for c in range(10):
        if inp[r][c] == 6:
            print(f"  ({r}, {c})")

print("\nReflected cells (6 in output but not 6 in input):")
for r in range(10):
    for c in range(10):
        if out[r][c] == 6 and inp[r][c] != 6:
            print(f"  ({r}, {c})")
