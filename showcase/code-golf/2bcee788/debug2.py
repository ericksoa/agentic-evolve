import json

task = json.loads('''{"train": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 2, 0, 0, 0, 0], [0, 0, 4, 4, 4, 2, 0, 0, 0, 0], [0, 0, 0, 0, 4, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 4, 4, 3, 3, 3, 3], [3, 3, 4, 4, 4, 4, 4, 4, 3, 3], [3, 3, 3, 3, 4, 4, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}]}''')

inp = task["train"][0]["input"]
out = task["train"][0]["output"]

# Markers at column 5, rows 3,4,5
# Shape at (3,4), (4,2), (4,3), (4,4), (5,4)

# Using 2*mc - c formula:
# For shape at (4,2): 2*5 - 2 = 8 -> would put at (4,8) but output has nothing at (4,8)
# For shape at (4,3): 2*5 - 3 = 7 -> (4,7) output has 4 YES
# For shape at (4,4): 2*5 - 4 = 6 -> (4,6) output has 4 YES

print("Expected output row 4:", out[4])
print("Output positions with 4:", [c for c in range(10) if out[4][c] == 4])

# Hmm, the formula works for some cells but not (4,2)
# Wait, maybe the reflection is centered on the EDGE between marker and shape?

# The edge is between col 4 and col 5 (between shape and marker)
# So reflect across col 4.5:
# (4,2) -> 2*4.5 - 2 = 7? No that's still wrong...

# Let me check what reflection would give the correct output
# Shape: 2,3,4. Output: 2,3,4,5,6,7
# So we need to map 4->6, 3->7, 2->8? But 8 is not in output...

# Actually wait, output row 4 has 4s at indices 2,3,4,5,6,7
# That's the original shape (2,3,4) PLUS the markers (5) PLUS reflected (6,7)
# So reflected are just 6 and 7. Original shape cells at 3,4 reflect to 6,7.
# Cell at 2 doesn't have a corresponding reflection in output!

# This means only cells ADJACENT to markers are reflected!
# Shape cell at (4,4) is adjacent to marker at (4,5) -> reflects to (4,6)
# Shape cell at (4,3) is adjacent to (4,4) which is adjacent to marker -> reflects to (4,7)
# Shape cell at (4,2) is not in the reflection chain

# Actually let me think differently: the shape is reflected across the marker line
# But the reflection ONLY includes cells that are on the same rows as markers

# Markers at rows 3,4,5. Shape cells at those rows get reflected.
# Shape at (3,4) -> 2*5-4=6 -> (3,6)
# Shape at (4,2) -> 2*5-2=8 -> (4,8) but NOT in output
# Shape at (4,3) -> 2*5-3=7 -> (4,7) YES
# Shape at (4,4) -> 2*5-4=6 -> (4,6) YES
# Shape at (5,4) -> 2*5-4=6 -> (5,6)

# But (3,6) is not in output! Output row 3: [3, 3, 3, 3, 4, 4, 3, 3, 3, 3]
# Has 4 at positions 4,5 only

print("Output row 3:", out[3])
print("Expected: 4 at 4 (original), 4 at 5 (marker)")

# Wait - the marker at (3,5) becomes 4. And the reflection of (3,4) across 5 would be 6.
# But output row 3 only has 4s at 4 and 5, not at 6!

# So maybe only the directly adjacent cells get reflected?
# Or perhaps the reflection is JUST the marker cells getting the shape color?

# Let me re-examine: What if the rule is:
# 1. Background 0 -> 3
# 2. Shape stays as is
# 3. Markers (2) become shape color
# 4. For each marker, extend the shape by reflecting the adjacent shape cell

print("\nLet's verify the adjacency reflection theory:")
# Marker at (3,5). Adjacent to shape at (3,4). Reflect (3,4) -> (3,6)? No, (3,6) is 3 in output.
# So that's not it either.

# Let me try another interpretation:
# The markers show axis of symmetry. The entire shape+markers is made symmetric.

# Actually I wonder if the rule is simpler:
# Each row with a marker: make symmetric around the marker column
# The symmetry includes the shape cells on that row

# Row 3: Shape at 4, Marker at 5. Make symmetric: 4,5 -> 5,4 -> result is 4,5
# Row 4: Shape at 2,3,4, Marker at 5.
#   Symmetric of 2 around 5 is 8 - not in output
#   Symmetric of 3 around 5 is 7 - YES in output
#   Symmetric of 4 around 5 is 6 - YES in output
#   So result: 2,3,4,5,6,7 - but wait, 2 didn't reflect to 8?

# I see now - the reflection might be bounded by the grid or some other constraint.
# Let me check: output row 4 has 4s at 2,3,4,5,6,7. That's 6 cells.
# Original: 3 shape cells + 1 marker + 2 reflected = 6. But which 2 are reflected?
# If it's reflecting 3->7 and 4->6, that matches!
# So the cell at position 2 doesn't reflect because... it's too far?

# Maybe only cells that are within a certain distance of the marker reflect?
# Cells at (4,4) is distance 1 from marker -> reflects
# Cell at (4,3) is distance 2 from marker -> reflects
# Cell at (4,2) is distance 3 from marker -> doesn't reflect?

# Let me check row 3 and 5:
# Row 3: Shape at 4 (distance 1), reflects to 6. Output row 3: 4,5 only. NO 6!
# Hmm that breaks the pattern.

print("Output row 3:", out[3])
# [3, 3, 3, 3, 4, 4, 3, 3, 3, 3]
# 4s at positions 4 and 5 only

# OH! Maybe the reflection has a different interpretation:
# Positions that were 2 become the shape color
# AND the reflection is of the combined shape+marker pattern

# Original row 3: 0,0,0,0,4,2,0,0,0,0 -> shape at 4, marker at 5
# Output row 3: 3,3,3,3,4,4,3,3,3,3 -> 4s at 4 and 5

# The "2" just became "4". No actual reflection in row 3.

# Original row 4: 0,0,4,4,4,2,0,0,0,0 -> shape at 2,3,4, marker at 5
# Output row 4: 3,3,4,4,4,4,4,4,3,3 -> 4s at 2,3,4,5,6,7

# Here there IS reflection. But row 3 didn't have it.

# The difference: row 3 has only 1 shape cell, row 4 has 3.

# New theory: reflect only shape cells that are MORE than distance 1 from marker?
# Row 3: cell at 4 is distance 1 -> no reflection
# Row 4: cell at 4 is distance 1 -> no reflection
#        cell at 3 is distance 2 -> reflects to 7
#        cell at 2 is distance 3 -> reflects to 8? But 8 is not in output

# Hmm still doesn't work for cell at 2.

# Let me check if maybe the shape cells reflect based on the local structure
# For row 4: shape spans 2,3,4. Marker at 5.
# The extent of the shape is 3 cells. So the reflection also spans 3 cells?
# From marker 5: 5,6,7 (marker + 2 more)

# That would mean: original shape extent = 3, so reflected extent (excluding marker) = 2
# Which gives us 2,3,4 (original) + 5 (marker) + 6,7 (reflected) = 2,3,4,5,6,7 MATCH!

# Let me verify with row 3:
# Row 3: shape at 4 only. Extent = 1. Marker at 5.
# Original: 4 (extent 1), Marker: 5, Reflected: nothing extra (extent - 1 = 0)
# Result: 4,5 MATCH!

# And row 5:
# Row 5: shape at 4 only (same as row 3). Marker at 5.
# Result should be: 4,5
print("Output row 5:", out[5])  # [3, 3, 3, 3, 4, 4, 3, 3, 3, 3] = 4s at 4,5 MATCH!

print("\nTheory: Reflect shape across marker, but reflected extent = original extent on that row")
