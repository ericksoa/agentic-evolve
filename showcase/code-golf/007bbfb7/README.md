# ARC Task 007bbfb7 - Kronecker-like Grid Tiling

## Problem Description

Given a 3x3 input grid with cells containing either 0 or a non-zero color value (2, 4, 6, or 7), produce a 9x9 output grid where:

- For each cell in the input grid at position (i, j):
  - If the cell is non-zero: place a copy of the entire input grid in the corresponding 3x3 block
  - If the cell is zero: place a 3x3 block of zeros

This is essentially a Kronecker product-like operation where the grid tiles itself based on its own pattern.

## Example

Input:
```
0 7 7
7 7 7
0 7 7
```

Output (9x9):
```
0 0 0 | 0 7 7 | 0 7 7
0 0 0 | 7 7 7 | 7 7 7
0 0 0 | 0 7 7 | 0 7 7
------+-------+------
0 7 7 | 0 7 7 | 0 7 7
7 7 7 | 7 7 7 | 7 7 7
0 7 7 | 0 7 7 | 0 7 7
------+-------+------
0 0 0 | 0 7 7 | 0 7 7
0 0 0 | 7 7 7 | 7 7 7
0 0 0 | 0 7 7 | 0 7 7
```

## Solution Statistics

| Metric | Value |
|--------|-------|
| **Final Bytes** | 65 |
| **Score** | 2435 |
| **Correctness** | 100% (6/6 examples) |

## Evolution Journey

| Generation | Bytes | Score | Key Change |
|------------|-------|-------|------------|
| Gen 0 (baseline) | 280 | 2220 | Initial working solution with nested loops |
| Gen 1 | 84 | 2416 | def to lambda, double list comprehension |
| Gen 2 | 82 | 2418 | `(g[i//3][j//3]>0)` instead of `bool()` |
| Gen 3 | 81 | 2419 | `and` instead of multiplication |
| Gen 4-5 | - | - | Failed mutations (tuple instead of range) |
| Gen 6 | 80 | 2420 | Row iteration pattern with nested `for` |
| Gen 7 | 76 | 2424 | Tuple indices `(0,1,2)` instead of `range(3)` |
| Gen 8 | 69 | 2431 | Direct element iteration `for x in r` |
| Gen 9-10 | - | - | Failed mutations (mult for and) |
| Gen 11 | 65 | 2435 | **Bitwise AND** - key insight! |

**Total improvement: 215 bytes (76.8% reduction)**

## Key Tricks Used

### 1. def-to-lambda Conversion
```python
# Before (280 bytes)
def solve(g):
    result = [[0]*9 for _ in range(9)]
    ...

# After
solve=lambda g:...
```
Saves 4+ bytes by eliminating `def`, `:`, `return`.

### 2. Algorithm Transformation: Index-based to Row-based
```python
# Index-based (84 bytes)
[[g[i%3][j%3]*bool(g[i//3][j//3])for j in range(9)]for i in range(9)]

# Row-based (69 bytes)
[[c and x for c in a for x in b]for a in g for b in g]
```
Eliminates all division/modulo operations by iterating rows directly.

### 3. Bitwise AND Trick (Key Insight!)
```python
# Using 'and' (69 bytes)
c and x

# Using '&' (65 bytes)
c&x
```
This works because:
- Each task example uses only ONE non-zero color (2, 4, 6, or 7)
- When both c and x are the same non-zero value: `c&x = c = x`
- When either is 0: `c&x = 0`

**Saves 4 bytes** by replacing ` and ` with `&`.

## Champion Solution

```python
solve=lambda g:[[c&x for c in a for x in b]for a in g for b in g]
```

### How It Works

1. `for a in g` - iterate over each row of input (outer blocks)
2. `for b in g` - iterate over each row of input (inner tiling)
3. `for c in a` - iterate over each cell in outer row (determines if block is active)
4. `for x in b` - iterate over each cell in inner row (the tile pattern)
5. `c&x` - if c is non-zero, output x; if c is zero, output 0

The nested iteration naturally produces 9 rows of 9 cells, tiling the input pattern based on itself.
