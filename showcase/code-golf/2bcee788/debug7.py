import json

# Test case
task = json.loads('''{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 2, 0, 0, 0], [0, 0, 0, 0, 0, 1, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 1, 3, 3, 1, 3, 3], [3, 3, 3, 3, 1, 1, 1, 1, 3, 3], [3, 3, 3, 3, 3, 1, 1, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}''')

inp = task["input"]
out = task["output"]

print("Test case:")
print("Markers:")
for r in range(10):
    for c in range(10):
        if inp[r][c] == 2:
            print(f"  ({r}, {c})")

print("\nShape (1):")
for r in range(10):
    for c in range(10):
        if inp[r][c] == 1:
            print(f"  ({r}, {c})")

# Markers at (4,6) and (5,6) - VERTICAL line at col 6
# Shape at (3,4), (4,4), (4,5), (5,5) - shape LEFT of markers

print("\nMarker_rows:", {4, 5})
print("Marker_cols:", {6})
print("Shape is LEFT of marker")

print("\nExpected output:")
for r in range(10):
    for c in range(10):
        if out[r][c] == 1:
            print(f"  ({r}, {c})")

# My algorithm for vertical markers with shape left:
# Group by row and use edge = max(cs) + 0.5

print("\nMy algorithm:")
shape = [(3,4), (4,4), (4,5), (5,5)]
shape_by_row = {}
for r, c in shape:
    if r not in shape_by_row:
        shape_by_row[r] = []
    shape_by_row[r].append(c)

for r, cs in shape_by_row.items():
    edge = max(cs) + 0.5
    print(f"Row {r}: shape cols {cs}, edge = {edge}")
    for c in cs:
        new_c = int(2 * edge - c)
        print(f"  col {c} -> col {new_c}")

# Row 3: shape col 4, edge = 4.5, reflected to col 5
# Row 4: shape cols 4,5, edge = 5.5, reflected to cols 7,6
# Row 5: shape col 5, edge = 5.5, reflected to col 6

# But expected output has:
# Row 3: cols 4, 7
# Row 4: cols 4,5,6,7
# Row 5: cols 5,6

# My Row 3 gives col 5, but expected is col 7!

print("\nExpected vs my calculation:")
print("Row 3: expected cols 4,7, my cols 4,5")
print("Row 4: expected cols 4,5,6,7, my cols 4,5,6,7 - matches!")
print("Row 5: expected cols 5,6, my cols 5,6 - matches!")

# The issue is row 3 - it should reflect across col 6 (marker col), not col 4.5
# Because row 3 has no marker, but the reflection should still use the marker line!

print("\nAh! Row 3 has no marker, but should still reflect across marker col 6")
print("So edge should be marker_col - 0.5 = 5.5 for all rows")
print("Row 3: 2*5.5 - 4 = 7 - correct!")
