import json

task = json.loads('''{"train": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 2, 0, 0, 0, 0], [0, 0, 4, 4, 4, 2, 0, 0, 0, 0], [0, 0, 0, 0, 4, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 4, 4, 3, 3, 3, 3], [3, 3, 4, 4, 4, 4, 4, 4, 3, 3], [3, 3, 3, 3, 4, 4, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}]}''')

inp = task["train"][0]["input"]
out = task["train"][0]["output"]

print("Input:")
for r in inp:
    print(r)

print("\nOutput:")
for r in out:
    print(r)

# Find markers and shape
for r in range(10):
    for c in range(10):
        if inp[r][c] == 2:
            print(f"Marker at ({r}, {c})")
        elif inp[r][c] == 4:
            print(f"Shape at ({r}, {c}) -> output at ({r}, {c}): {out[r][c]}")

# Check reflected positions
print("\nReflected shape cells in output:")
for r in range(10):
    for c in range(10):
        if out[r][c] == 4 and inp[r][c] != 4:
            print(f"New shape at ({r}, {c})")
