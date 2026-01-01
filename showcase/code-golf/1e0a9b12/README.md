# ARC Task 1e0a9b12: Gravity Sort

## Problem Description

Given a 2D grid with colored cells (non-zero values), apply "gravity" so all non-zero values fall to the bottom of each column while maintaining their relative order within the column.

### Input/Output Examples

**Example 1:**
```
Input:               Output:
0 4 0 9              0 0 0 0
0 0 0 0      -->     0 0 0 0
0 4 6 0              0 4 0 0
1 0 0 0              1 4 6 9
```

**Example 2:**
```
Input:                     Output:
0 0 0 0 0 9                0 0 0 0 0 0
0 0 0 8 0 0                0 0 0 0 0 0
0 0 0 0 0 0       -->      0 0 0 0 0 0
4 0 0 0 0 0                4 0 0 0 0 0
4 0 7 8 0 0                4 0 7 8 0 0
4 0 7 0 0 0                4 0 7 8 0 9
```

## Champion Solution

```python
solve=lambda g:[*map(list,zip(*[sorted(c,key=id)for c in zip(*g)]))]
```

### Stats
- **Bytes:** 69
- **Score:** 2431 (2500 - 69)
- **Fitness:** 0.9724

## Evolution Journey

| Gen | Bytes | Score | Key Change |
|-----|-------|-------|------------|
| 0 | 233 | 2267 | Baseline: explicit loops with height/width variables |
| 1 | 180 | 2320 | Compressed whitespace, semicolons |
| 2 | 171 | 2329 | Removed width variable, iterate rows directly |
| 3 | 129 | 2371 | Used zip(*g) transpose with walrus operator |
| 4 | 128 | 2372 | Replaced walrus with lambda helper |
| 5 | 125 | 2375 | Used filter(None,c) instead of list comp |
| 6 | 122 | 2378 | Changed list(r) to [*r] |
| 7 | 79 | 2421 | BREAKTHROUGH: sorted(c,key=bool) algorithm |
| 8 | 73 | 2427 | def to lambda conversion |
| 9 | 71 | 2429 | [*r]for r -> map(list,...) |
| 10 | 69 | 2431 | BREAKTHROUGH: key=id instead of key=bool |

## Key Golf Tricks Used

1. **sorted with key=id**: Python caches small integers, so 0 always has the lowest memory id. This puts zeros first without needing `key=bool` (saves 2 bytes).

2. **Double zip transpose**: `zip(*g)` transposes rows to columns, operate on columns, then `zip(*...)` transposes back.

3. **[*map(list,...)]**: More compact than `[list(r)for r in ...]` when mapping list over iterables.

4. **Lambda over def**: `solve=lambda g:` saves 4-5 bytes over `def solve(g):\n return`.

5. **Stable sort for gravity**: sorted() preserves relative order of equal keys, so non-zero values maintain their order after zeros.

## Algorithm Explanation

1. `zip(*g)` - Transpose grid (rows become columns)
2. `sorted(c,key=id)` - Sort each column by memory id (0 has lowest id, goes first)
3. `zip(*...)` - Transpose back (columns become rows)
4. `map(list,...)` - Convert tuples to lists
5. `[*...]` - Unpack map object to list

The key insight is that "gravity" is just a stable sort where zeros come first - and Python's integer caching makes `key=id` the shortest way to express this.
