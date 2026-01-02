# Task 150deff5

## Pattern
Decompose a 5-shape into non-overlapping 2x2 blocks (→8s) and remaining cells (→2s), minimizing isolated 2-cells.

## Algorithm
1. Find all cells with value 5 (set F)
2. Find all possible 2x2 block positions where all 4 cells are in F (list V)
3. Enumerate all 2^n subsets of V using bitmask
4. For each subset, check non-overlap by verifying total covered cells = 4 × blocks
5. Score by (isolation count, -block count) - minimize isolated 2-cells, then maximize blocks
6. Output: 8 for cells in selected blocks, 2 for remaining 5-cells, 0 elsewhere

## Key Tricks
- `divmod(x,C)` for flat-to-2D index conversion saves nested loops
- `{(i,j),(i+1,j),(i,j+1),(i+1,j+1)}<=F` for 2x2 block membership check
- Bitmask iteration: `m>>k&1` to check if block k is selected
- `len(c)==4*len(s)` for non-overlap check (unique cells = 4 × blocks)
- `(condition)*9 or value` for penalty-based scoring (shorter than ternary)
- `~len(s)` for negation (shorter than `-len(s)` in tuples)
- `8*((i,j)in E)or 2*((i,j)in F)` for conditional output

## Challenges
- Multiple valid 2x2 tilings exist; must pick one that leaves 2-cells connected
- Isolation heuristic is essential - simple greedy/max-coverage fails
- Brute force over subsets is O(2^n) but n is small (≤10 blocks typically)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 684 | Original with itertools.combinations |
| v2 | 641 | Switch to bitmask enumeration |
| v3 | 617 | Use `(i+a,j+b)` loop for block cells |
| v4 | 558 | Return E directly from min, add `~len` |
| v5 | 509 | Store c in return tuple, use `range(n)` indexing |
| v6 | 496 | `(condition)*999 or sum(...)` trick |
| v7 | 494 | Reduce penalty to 9 |
