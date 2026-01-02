# ARC Task 05f2a901 - Move Shape to Reference

**Pattern:** Move shape to reference (Medium)
**Final Score:** 2174 points (326 bytes)

## Task Description

Move a colored shape (color 2) to be adjacent to a reference 2x2 block (color 8). The reference block stays in place while the shape moves to touch it.

## Evolution Log

| Gen | Bytes | Score | Key Change |
|-----|-------|-------|------------|
| 0 | 656 | 1844 | Initial correct solution with 4-space indents |
| 1 | 512 | 1988 | Single-space indentation |
| 2 | 464 | 2036 | Shorter variable names (R,C,r,c) |
| 3 | 434 | 2066 | Replaced len(g),len(g[0]) with iteration |
| 4 | 429 | 2071 | R=C=r=c=[] initialization |
| 5 | 396 | 2104 | Restructured with P list and lambda filter |
| 6 | 392 | 2108 | Single-char variable names |
| 7 | 378 | 2122 | Used `or` trick for ternary |
| 8 | 376 | 2124 | Used `+` instead of `or` (both work) |
| 9 | 375 | 2125 | Combined d,e on same line |
| 10 | 326 | 2174 | Added H lambda for offset calculation, removed newline |

## Key Golf Techniques Used

1. **Single-space indentation**: Saves 3 bytes per indented line
2. **Lambda functions for filtering**: `F=lambda n,k:sorted(p[k]for p in P if p[2]==n)`
3. **Boolean multiplication**: `(b[0]>a[-1])*(b[0]-a[-1]-1)` for conditional values
4. **Sorted list indexing**: `sorted(...)[0]` and `sorted(...)[-1]` for min/max
5. **Semicolon chaining**: Multiple statements on one line
6. **Reusable lambda**: `H=lambda a,b:...` for repeated offset calculation
7. **No trailing newline**: Saves 1 byte

## Solution Algorithm

1. Extract all non-zero pixels as (row, col, value) tuples
2. Separate into shape (color 2) and reference (color 8) positions
3. Calculate bounding box extremes using sorted lists
4. Compute row offset `d` and column offset `e` to make shapes adjacent
5. Create zero-filled output grid
6. Place each pixel at its new position (reference stays, shape moves)

## Final Solution (326 bytes)

```python
def solve(g):
 P=[(i,j,v)for i,q in enumerate(g)for j,v in enumerate(q)if v];F=lambda n,k:sorted(p[k]for p in P if p[2]==n);R,r,C,c=F(2,0),F(8,0),F(2,1),F(8,1);H=lambda a,b:(b[0]>a[-1])*(b[0]-a[-1]-1)+(a[0]>b[-1])*(b[-1]-a[0]+1);d=H(R,r);e=H(C,c);o=[[0]*len(g[0])for w in g]
 for i,j,v in P:o[i+(v<8)*d][j+(v<8)*e]=v
 return o
```
