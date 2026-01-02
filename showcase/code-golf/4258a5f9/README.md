# Task 4258a5f9

## Pattern

Draw a 3×3 box of 1s around each 5 in a 9×9 grid, keeping the 5 in the center.

## Algorithm

For each cell with value 5:
1. Iterate over the 3×3 neighborhood (d,e in {-1,0,1})
2. If the neighbor is within bounds and not already 5
3. Set it to 1

## Key Tricks

- `E=enumerate` - alias used twice
- `__setitem__` - modify grid in list comprehension
- `v>4` - shorter than `v==5` (grid only has 0s and 5s)
- `9>r+d>-1<c+e<9` - chained bounds (hardcoded 9×9)
- `g[r+d][c+e]+4<9` - check if cell is not 5 (0+4<9, 5+4>=9)

## Byte History

| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 216 | Initial with nested loops |
| v2 | 185 | List comprehension with `__setitem__` |
| v3 | 172 | Hardcode 9×9 grid |
| v4 | 163 | Shorter variable names, `v>4` |
| v5 | 160 | Chain condition `9>g[r+d][c+e]+4` |

## Solution (160 bytes)

```python
def solve(g):E=enumerate;[g[r+d].__setitem__(c+e,1)for r,R in E(g)for c,v in E(R)if v>4for d in(-1,0,1)for e in(-1,0,1)if 9>r+d>-1<c+e<9>g[r+d][c+e]+4];return g
```

Score: **2340 points** (2500 - 160)
