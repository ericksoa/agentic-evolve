# ARC Task 239be575 - Code Golf Evolution

**Pattern:** Small pattern movement (Easy)
**Final Score:** 2330 (170 bytes)

## Problem Analysis

The task contains grids with values 0, 2, and 8. Each grid has exactly two 2x2 blocks filled with 2s. The goal is to output a 1x1 grid containing either 0 or 8.

**Discovery:** The output depends on:
1. The relative column positions of the two blocks (whether the second block is to the right of the first)
2. Whether BOTH "opposite corner" cells contain 8 (the cells at positions that would form the other two corners of the bounding rectangle)

Rule: Output is 8 if (block2.col > block1.col) AND (sum of corner values < 16), otherwise 0.

## Evolution Log

| Gen | Bytes | Score | Key Optimization |
|-----|-------|-------|------------------|
| 0 | 607 | 1893 | Initial working solution with explicit loops |
| 1 | 322 | 2178 | Shorter variable names, chained comparisons |
| 2 | 236 | 2264 | List comprehension for block finding |
| 3 | 214 | 2286 | Direct tuple unpacking `a,e=b` |
| 4 | 208 | 2292 | Sum check `!=16` instead of both `==8` |
| 5 | 199 | 2301 | Better variable names (r,c,R,C) |
| 6 | 191 | 2309 | Inline list comprehension unpacking |
| 7 | 187 | 2313 | Chained comparison `C>c<16!=` |
| 8 | 185 | 2315 | Product check `*==16` for block detection |
| 9 | 181 | 2319 | Enumerate with row slicing |
| 10 | 180 | 2320 | Mixed enumerate/range approach |
| 11 | 178 | 2322 | Bitwise AND for block detection `&2` |
| 12 | 177 | 2323 | zip(g,g[1:]) for row pairs |
| 13 | 176 | 2324 | Comparison chain `C>c<16>sum` |
| 14 | 170 | 2330 | Single-space indentation |

## Champion Solution (170 bytes)

```python
def solve(g):
 (r,c),(R,C)=[(i,j)for i,(a,b)in enumerate(zip(g,g[1:]))for j in range(len(a)-1)if a[j]&a[j+1]&b[j]&b[j+1]>1]
 return[[8*(C>c<16>g[r+1][C+1]+g[R+1][c+1])]]
```

## Key Golf Tricks Used

1. **Bitwise AND for block detection**: `a[j]&a[j+1]&b[j]&b[j+1]>1` - All four cells must have bit 1 set (value 2), so their AND is 2
2. **zip for consecutive rows**: `zip(g,g[1:])` - Pairs each row with the next
3. **Chained comparisons**: `C>c<16>sum` - Checks C>c AND c<16 AND 16>sum in one expression
4. **Single-space indent**: Saves 2 bytes per indented line
5. **Tuple unpacking**: `(r,c),(R,C)=` directly extracts block positions
6. **Boolean multiplication**: `8*(condition)` returns 8 if true, 0 if false
