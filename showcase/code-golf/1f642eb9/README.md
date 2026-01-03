# Task 1f642eb9

## Pattern
Project colored markers toward a central 8-block, placing them at the block's edges.

## Algorithm
Find the rectangular bounding box of all 8-valued cells (block edges a,b,c,d for top/bottom/left/right).
Collect all non-zero, non-8 marker positions. For each marker at (r,j):
- If it's vertically aligned with the block (c<=j<=d), project to the top row `a` if above, bottom row `b` if below
- If it's horizontally aligned with the block (a<=r<=b), project to the left column `c` if left, right column `d` if right

## Key Tricks
- `M+=(r,j,v),` - tuple append saves 1 byte over `M+=[(r,j,v)]`
- `(a,b)[r>b]` - boolean index to select between two values without if/else
- Single-pass boundary detection with min/max chaining
- Combined loop collects markers while finding block boundaries

## Evolution Summary (AlphaEvolve-Inspired)
11 generations, 44 mutations tested. Final: **266 bytes** (-24%, -84 bytes from initial 350)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 2 | Combined marker collection loop | 327 | -23 |
| 4 | Tuple append trick | 326 | -1 |
| 6 | Direct indexing with `(a,b)[r>b]` | 298 | -28 |
| 7 | Clean conditional projection | 266 | -32 |

### Failed Approaches
- Walrus operators for inline assignment (added overhead)
- __setitem__ with boolean guards (more verbose)
- List comprehension for boundary finding (longer than loop)
- Separate loops for rows/columns (duplicated logic)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 350 | Initial solution with zip(*[...]) for boundaries |
| v2 | 327 | Combined marker collection with boundary finding |
| v3 | 326 | Tuple append trick |
| v4 | 298 | Boolean indexing for projection target |
| v5 | 266 | Clean conditional structure |
