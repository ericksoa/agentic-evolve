# Task 1bfc4729

## Pattern
Draw two rectangular frames - top half uses color from row 2 marker, bottom half uses color from row 7 marker.

## Algorithm
The input always has two colored pixels at rows 2 and 7. The output divides the 10x10 grid at row 5:
- Top region (rows 0-4): uses color from row 2
- Bottom region (rows 5-9): uses color from row 7

Each region draws a rectangular frame pattern:
- "Full" rows (row 0, marker row, row 9): entire row filled with color
- Other rows: only edge columns (0 and 9) filled

The full rows are {0, 2, 7, 9} which equals 645 in binary (1010000101), enabling a bit-test for row type.

## Key Tricks
- `max(g[2])` / `max(g[7])` - extract marker color from known rows (columns vary)
- `645>>i&1` - bit mask checks if row i is a "full" row (645 = 0b1010000101)
- `j%9<1` - true when j=0 or j=9 (edge columns)
- `(a,b)[i>4]` - select color based on region (shorter than ternary)
- `color*(condition)` - multiplication by bool (0 or 1) instead of `if/else 0`
- `R=range(10)` - cache range to avoid repeating

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 406 | Initial solution with nested loops and explicit conditionals |
| v2 | 108 | Hardcode row positions, bit mask for full rows, multiplication trick |
