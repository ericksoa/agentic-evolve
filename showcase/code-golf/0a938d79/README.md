# Task 0a938d79

## Pattern
Two non-zero cells define alternating stripes that repeat across the grid, either horizontally or vertically.

## Algorithm
Find the two colored cells and sort them by position. Calculate row and column deltas between them. If column delta is 0 (same column) or row delta is smaller and positive, create horizontal stripes; otherwise vertical stripes. The stripe period is 2×delta, with the first value at positions 0, 2d, 4d... and second value at positions d, 3d, 5d... relative to the starting point. Only fill positions at or after the starting coordinate.

## Key Tricks
- `sorted((i,j,v)for i,r in enumerate(g)for j,v in enumerate(r)if v)` - find cells with double enumerate
- `(a,b,u),(c,e,w)=M` - unpack both points in one statement
- `h=d<1or 0<r<d` - condition for horizontal mode (triggers syntax warning but works)
- `D=[d,r][h]` - select delta using boolean index
- `[j-b,i-a][h]` - select column or row offset based on mode
- `[u,w][k//D%2]` - alternates between v1 and v2 based on period
- `*(k%D<1)*(k>=0)` - multiplication for stripe alignment and bounds check
- `for j,_ in enumerate(R)` - avoid `range(len(g[0]))`

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 539 | Initial solution with explicit loops and conditionals |
| v2 | 262 | List comprehension, dict.get approach |
| v3 | 241 | Shorter variable names |
| v4 | 240 | enumerate instead of range(len(...)) |
| v5 | 237 | One-liner with semicolons |

## Challenges
- Two different modes (horizontal/vertical) require conditional logic
- The condition `d<1or 0<r<d` creates a syntax warning for "invalid decimal literal" (`1or` looks like a number) but executes correctly
- Need to handle the boundary condition (only fill at/after starting position)

## Potential Improvements
- The `d<1or` syntax warning could be avoided with `1>d or` (+1 byte) or bitwise `(d<1)|(0<r<d)` (+2 bytes)
- Finding a unified formula that doesn't require the h conditional could save bytes
