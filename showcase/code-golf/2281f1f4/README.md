# Task 2281f1f4

## Pattern
Row 0 has 5s marking columns; column 9 has 5s marking rows. Fill intersection cells with 2.

## Algorithm
Iterate through each row R. For each cell, pair the value with the corresponding marker
from row 0 using zip. If both the row marker (R[-1]) and column marker (m) are 5, output 2;
otherwise keep the original value.

## Key Tricks
- `R[-1]>4<m` - chained comparison checks both markers are 5 (only value > 4 in grid)
- `[v,2][condition]` - list indexing for conditional selection (shorter than ternary)
- `zip(g[0],R)` - pairs column markers with row values without enumerate
- Lambda instead of def saves 6 bytes

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 67 | Initial golfed solution using zip and chained comparison |
