# ARC Task 32597951 - Code Golf Evolution

## Task Description
**Pattern:** Mark 1s adjacent to 8s within the 8-bounded region

Given a grid with 0s, 1s, and 8s:
- Find the bounding box of all cells containing 8
- Within that bounding box, find cells with value 1 that are adjacent (8-directionally) to an 8
- Change those cells from 1 to 3

## Champion Solution

**Final byte count:** 274 bytes
**Score:** 2226 points

```python
def solve(g):
 o=[*map(list,g)];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7}
 if e:I,J=zip(*e);[o[r].__setitem__(c,3)for r in range(min(I),max(I)+1)for c in range(min(J),max(J)+1)if g[r][c]==1if e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}]
 return o
```

## Evolution Log

### Generation 1 - Initial Correct Solution
**Bytes:** 967

Readable implementation with full variable names, explicit loops, and clear logic.

### Generation 2 - First Golf Pass
**Bytes:** 409 (-558)

Applied basic golf tricks:
- Single-letter variable names (g, o, e, r, c)
- Removed unnecessary whitespace
- Combined statements with semicolons
- Used list comprehension for adjacency check

### Generation 3 - Semicolon Compression
**Bytes:** 363 (-46)

Combined more statements on single lines with semicolons.

### Generation 4 - Index Encoding
**Bytes:** 326 (-37)

Experimented with encoding row/col as single index (p = r*C + c).

### Generation 5 - Alternative Tuple Approach
**Bytes:** 326 (0)

Tried enumerate with tuple unpacking - same size.

### Generation 6 - Set Operations
**Bytes:** 321 (-5)

Used set intersection (e&{...}) instead of any() for adjacency check.

### Generation 7 - zip(*e) Trick
**Bytes:** 297 (-24)

Major optimization: Used `zip(*e)` to separate row and column indices, then min/max directly on those.

### Generation 8 - Minor Optimizations
**Bytes:** 296 (-1)

Changed `v==8` to `v>7` (works since only 0, 1, 8 appear in the grid).

### Generation 9 - List Comprehension Side Effect
**Bytes:** 278 (-18)

Replaced nested for loops with list comprehension using `o[r].__setitem__(c,3)`.

### Generation 10 - Final Optimizations
**Bytes:** 275 (-3)

Multiple variations tested:
- Changed `[r[:] for r in g]` to `[*map(list,g)]` (-2 bytes)
- Used `if...if` syntax for chained conditions (-1 byte)

### Final - Trailing Newline Removal
**Bytes:** 274 (-1)

Removed trailing newline from solution file.

## Key Golf Techniques Used

1. **Single-letter variables**: g, o, e, r, c, a, b, I, J
2. **Set comprehension for 8-positions**: `{(i,j) for...if v>7}`
3. **zip(*e) for separating coordinates**: Unzips set of tuples into (rows, cols)
4. **Set intersection for adjacency**: `e&{neighbors}` instead of `any()`
5. **__setitem__ in comprehension**: Side-effect mutation without explicit loop
6. **Chained if**: `if X if Y` instead of `if X and Y` (saves 3 chars)
7. **v>7 instead of v==8**: Works when 8 is the only value >7
8. **[*map(list,g)]**: Shorter than `[r[:] for r in g]` for deep copy

## Algorithm Summary

```
1. Create output copy of grid
2. Find all positions with value 8 (the marker cells)
3. Get bounding box from min/max of 8-positions
4. For each cell in bounding box:
   - If value is 1 AND any neighbor is 8
   - Set output cell to 3
5. Return output
```
