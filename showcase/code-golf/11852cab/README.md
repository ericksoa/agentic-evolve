# Task 11852cab

## Pattern
Complete 4-fold rotational symmetry by rotating each colored cell 90°, 180°, 270° around the bounding box center.

## Algorithm
Find all non-zero cells and compute the center of their bounding box. For each colored cell,
calculate its three rotated positions (90°, 180°, 270° counterclockwise). Place the cell's
color at each rotated position if that position is empty and within bounds.

The key insight is expressing rotations using integer arithmetic with a single precomputed
value `u = (s+t)>>1` where s,t are doubled center coordinates. The three rotations of
point (i,j) are: (u-j, t+i-u), (s-i, t-j), (j+u-t, u-i).

## Key Tricks
- `eval(str(g))` for deep copy (12 chars vs 16 for `[*map(list,g)]`)
- `s+t>>1` instead of `(s+t)//2` saves 2 bytes (bit shift vs floor division)
- Eliminate `v` variable by rewriting `i-v` as `t+i-u` and `v+j` as `j+u-t`
- `o[a][b]=o[a][b]or g[i][j]` - `or` trick fills empty cells only
- Move for loop outside `if P:` block to reduce indentation bytes
- Chained comparison `R>a>=0<=b<C` for compact bounds check

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, 40 mutations tested. Final result: **280 bytes** (-53 bytes, -16% from original)

### Key Discoveries by Generation

| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 0 | Baseline (complex number rotations) | 333 | - |
| 1 | Variable inlining tests (all failed) | 289-295 | - |
| 2 | Alternative algorithms (all worse) | 295-329 | - |
| 3 | Dict/numpy attempts (mostly failed) | 280-340 | - |
| 4 | **v=u-t insight** | 284 | -5 |
| 5 | **Eliminate v entirely** | 282 | -7 |
| 6 | **Bit shift >>1** | 280 | -9 |
| 7-10 | Plateau, expression reordering only | 280 | 0 |

### Learnings

1. **Variable elimination > variable reuse**: Computing `v=(s-t)//2` costs more than expressing formulas directly in terms of `u` and `t`

2. **Bit shift for division by 2**: `x>>1` is 2 bytes shorter than `(x)//2`

3. **Failed approaches**:
   - Lambda functions add overhead for multi-statement code
   - `exec()` can't modify local variables in Python 3
   - List comprehensions with `__setitem__` are longer than explicit loops
   - `enumerate()` is longer than `range()` when indices are needed
   - Dict-based approaches have too much overhead
   - Inlining R,C loses more than it saves (multiple uses)

4. **Crossover insight**: Successful golf tricks compound - the v-elimination from Gen 5 combined with bit-shift from Gen 6 yielded the champion

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 333 | Initial solution using complex number rotations |
| v2 | 314 | `eval(str(g))` for copy, skip identity rotation |
| v3 | 305 | Integer rotation formulas instead of complex numbers |
| v4 | 299 | Precompute u,v values |
| v5 | 295 | Use `or` trick for conditional assignment |
| v6 | 289 | Move for loop outside if block |
| v7 (Gen4d) | 284 | Compute v=u-t instead of v=(s-t)//2 |
| v8 (Gen5a) | 282 | Eliminate v entirely, use t+i-u and j+u-t |
| v9 (Gen6d) | 280 | Use >>1 instead of //2 for division |
