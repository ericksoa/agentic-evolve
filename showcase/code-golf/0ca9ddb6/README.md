# ARC Task 0ca9ddb6

## Problem Description

Given a 9x9 grid containing colored pixels:
- **Value 1 (blue)**: Add orange (7) markers in orthogonal positions (up, down, left, right)
- **Value 2 (red)**: Add yellow (4) markers in diagonal positions (corners)
- Other non-zero values (6, 8, etc.): Remain unchanged
- Markers only replace empty (0) cells

### Visual Pattern
```
For value 1:     For value 2:
    7                4   4
  7 1 7                2
    7                4   4
```

## Solution Statistics

| Metric | Value |
|--------|-------|
| **Final Byte Count** | 207 bytes |
| **Score** | 2293 (2500 - 207) |
| **Fitness** | 0.9172 |
| **Correctness** | 100% (4/4 examples) |

## Evolution Journey

| Gen | Bytes | Score | Key Change |
|-----|-------|-------|------------|
| 0 | 606 | 1894 | Initial verbose solution |
| 1 | 279 | 2221 | Removed bounds checks (bug), compact loops |
| 2 | 268 | 2232 | Fixed bounds, unified direction tuples with (k<5)+1 formula |
| 3 | 266 | 2234 | Chained comparison -1<x<len optimization |
| 4 | 252 | 2248 | Hardcoded 9 instead of len(g) |
| 5 | 236 | 2264 | Lambda conversion |
| 6 | 225 | 2275 | Used multiplication instead of if-else (7*, 4*) |
| 7 | 219 | 2281 | Nested for loops for diagonal directions |
| 8 | 216 | 2284 | R=range(9) variable reuse |
| 9 | 207 | 2293 | Explicit tuple list for orthogonal, nested for diagonal |

## Key Golf Tricks Used

1. **Lambda conversion**: `solve=lambda g:` instead of `def solve(g):`
   - Saves ~4 bytes

2. **Chained comparisons**: `8>i+a>-1<j+b<9` instead of `0<=i+a<9 and 0<=j+b<9`
   - Saves ~10 bytes

3. **Hardcoded grid size**: Use `9` instead of `len(g)`
   - Saves ~6 bytes per use

4. **Multiplication for conditional**: `7*any(...)` instead of `7 if any(...) else 0`
   - Saves ~8 bytes

5. **Range variable reuse**: `R=range(9)` used twice
   - Saves 7 bytes

6. **Nested for loops for Cartesian product**: `for a in(-1,1)for b in(-1,1)`
   - More compact than explicit list of tuples for 4 diagonal directions

7. **Exploiting `or` short-circuit**: `g[i][j]or 7*... or 4*...`
   - Only computes markers if cell is empty (0)

## Champion Solution

```python
R=range(9)
solve=lambda g:[[g[i][j]or 7*any(8>i+a>-1<j+b<9and g[i+a][j+b]==1for a,b in[(0,1),(0,-1),(1,0),(-1,0)])or 4*any(8>i+a>-1<j+b<9and g[i+a][j+b]==2for a in(-1,1)for b in(-1,1))for j in R]for i in R]
```

## Algorithm

The solution uses a list comprehension approach:
1. For each cell (i,j), check if original value is non-zero (keep it)
2. Otherwise, check if any orthogonal neighbor is 1 (blue) -> place 7 (orange)
3. Otherwise, check if any diagonal neighbor is 2 (red) -> place 4 (yellow)
4. Otherwise, cell stays 0
