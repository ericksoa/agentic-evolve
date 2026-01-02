import json

# Example 2 (index 1 in train)
task = json.loads('''{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 6, 3, 3, 3, 3], [3, 3, 3, 3, 6, 3, 3, 3, 3, 3], [3, 3, 3, 6, 6, 3, 3, 3, 3, 3], [3, 3, 3, 6, 6, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 3, 3, 3, 3, 3], [3, 3, 3, 3, 6, 6, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}''')

inp = task["input"]
out = task["output"]

print("Example 2 (train index 1):")
print("Markers at row 3, cols 3,4 (horizontal line)")
print("Shape at rows 4,5,6")

# Col 5 has shape at row 6 only (cell (6,5))
# But col 5 is NOT in marker_cols!

print("\nThe issue: my code only looks at columns that have markers")
print("But col 5 has shape but no marker")
print("Col 5 shape: row 6")
print("Expected output col 5: 6 at rows 1 and 6")

# Hmm the output shows 6 at (1,5) and (6,5)
# (1,5) is new, (6,5) is original
# But marker row is 3, not touching col 5

print("\nActual output col 5:")
for r in range(10):
    if out[r][5] == 6:
        print(f"  row {r}")

# So reflection happens even for columns without markers!
# The marker defines the axis, and ALL shape cells reflect across it

print("\n\nExample 4 (train index 3):")
task4 = json.loads('''{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 8, 8, 8, 3, 3, 3, 3], [3, 3, 3, 3, 3, 8, 3, 3, 3, 3], [3, 3, 3, 3, 3, 8, 3, 3, 3, 3], [3, 3, 3, 8, 8, 8, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}''')

inp4 = task4["input"]
out4 = task4["output"]

print("Markers:")
for r in range(10):
    for c in range(10):
        if inp4[r][c] == 2:
            print(f"  ({r}, {c})")

print("Shape:")
for r in range(10):
    for c in range(10):
        if inp4[r][c] == 8:
            print(f"  ({r}, {c})")

print("\nReflected (8 in output but not in input):")
for r in range(10):
    for c in range(10):
        if out4[r][c] == 8 and inp4[r][c] != 8:
            print(f"  ({r}, {c})")
