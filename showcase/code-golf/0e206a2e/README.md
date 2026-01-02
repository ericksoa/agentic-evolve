# ARC Task 0e206a2e - Template Matching with Rotation

## Task Description

This ARC task involves pattern recognition and template matching with rotations/reflections.

**Pattern:**
1. The input contains multiple connected components made of a "filler" color (most common color)
2. Each component has corner markers of different colors at specific positions
3. Some corner markers appear isolated (not connected to filler cells)
4. The output places templates at the isolated marker positions, matching the orientation implied by the marker arrangement

## Solution Approach

1. **Find filler color**: Most frequently occurring non-zero color
2. **Find connected components**: DFS starting from filler cells, expanding to adjacent non-zero cells (8-connectivity)
3. **Identify templates**: Each connected component is a template with corner markers
4. **Find isolated markers**: Non-filler cells not part of any template
5. **Match templates to isolated markers**: For each isolated marker group, find which template orientation matches (8 possible rotations/reflections)
6. **Place transformed templates**: Apply the matching transformation and place at isolated marker positions

## Evolution Progress (AlphaEvolve-Inspired)

**12 generations, ~48 mutations tested. Final: 1135 bytes (-18%, -249 bytes)**

| Generation | Bytes | Delta | Key Discovery |
|------------|-------|-------|---------------|
| 0 (initial) | 1384 | - | Working solution |
| 1 | 1370 | -14 | Inline directions with `a\|b` test |
| 2 | 1356 | -14 | Inline `n` variable |
| 3 | 1337 | -19 | Inline `er,ec` and `nr,nc` variables |
| 4 | 1318 | -19 | Inline `A=set().union(*C)` |
| 5 | 1271 | -47 | Eliminate Y,Z - compute transforms inline |
| 6 | 1249 | -22 | Fully inline dr,dc transformations |
| 7 | 1210 | -39 | Replace m/break with `all()` check |
| 8 | 1168 | -42 | Eliminate M variable, inline all() |
| 9 | 1152 | -16 | Set diff `{*p[k]}-A` |
| 10 | 1149 | -3 | Semicolon chain |
| 11 | 1144 | -5 | Inline A in I computation |
| 12 | 1135 | -9 | `pop()` for DFS + `_ in g` for output |

## Final Stats

- **Initial bytes**: 1384
- **Final bytes**: 1135
- **Reduction**: 249 bytes (-18%)
- **Score improvement**: 1116 → 1365 (+249 pts)

## Key Golf Tricks

1. **DFS with pop()**: `q.pop()` instead of `q.pop(0)` saves 2 bytes (BFS→DFS)
2. **Inline directions**: `a|b` test for non-zero instead of 8 direction tuples
3. **Walrus operator**: `if v:=g[r][c]` for inline assignment
4. **all() for matching**: Replace explicit m/break loop with `all()`
5. **Set diff syntax**: `{*p[k]}-set().union(*C)` instead of list comprehension
6. **`_ in g` trick**: `for _ in g` instead of `range(h)` for output grid
7. **Semicolon chaining**: Multiple statements on one line
8. **Variable elimination**: Inline Y, Z, dr, dc, er, ec computations
9. **Chained comparisons**: `h>nr>=0<=nc<w` for bounds checking

## Failed Approaches

- Moving R check outside rotation loop (small overhead increase)
- Explicit 8-tuple for directions (longer than nested loops)
- Separate A variable (recomputing in dict comp is shorter)
- dict-set subtraction syntax `p-{f}` (wrong type)
