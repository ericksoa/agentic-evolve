# Task 2204b7a8

## Pattern
Replace 3s with the color of the nearer border (top-left vs bottom-right), where borders can be vertical (left/right columns) or horizontal (top/bottom rows).

## Algorithm
First detect border orientation: if g[0][0]==g[1][0], borders are vertical (column 0 uniform). Get corner colors a=g[0][0] and b=g[-1][-1]. For each cell with value 3, compare position to halfway point: for vertical borders compare column j, for horizontal compare row i. Replace 3 with a if in first half, b if in second half.

## Key Tricks
- `[c,X][c==3]` pattern for conditional replacement (3 bytes shorter than ternary)
- `v:=a==g[1][0]` walrus operator to compute v inline in comprehension
- `[i,j][v]` and `[g,r][v]` for unified row/column selection with boolean index
- `len([g,r][v])` to get relevant dimension (height or width)
- `2*pos >= dim` for "second half" check (handles odd dimensions correctly)
- `E=enumerate` alias saves 4 bytes with two uses

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 167 | Initial working solution with explicit loops |
| v2 | 160 | Convert to list comprehension |
| v3 | 158 | Inline b variable |
| v4 | 148 | Use E=enumerate alias, c-3 condition |
| v5 | 146 | Use `len([g,r][v])` instead of `[len(g),len(r)][v]` |
| v6 | 141 | Simplify with list indexing trick |
| v7 | 138 | `[c,X][c==3]` pattern |
| v8 | 137 | Walrus operator for v inside comprehension |
