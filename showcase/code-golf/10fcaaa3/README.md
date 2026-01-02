# Task 10fcaaa3

## Pattern
Tile input 2×2, then fill empty cells with 8 if diagonally adjacent to any colored cell.

## Algorithm
Double the grid in both dimensions using modular indexing (`g[r%H][c%W]`). For each
cell in the output: if non-zero, keep original value; if zero, check if any colored
cell in the tiled grid is exactly diagonal (squared distance = 2). Fill with 8 if true.

## Key Tricks
- `r%H` and `c%W` tile via modular arithmetic (no explicit grid construction)
- `2in{...}` checks if squared distance 2 exists in set (diagonal = ±1,±1)
- `or 8*(...)` conditional fill: 0 stays 0, but `or 8*True` gives 8
- `range(H*2)` iteration with modular lookup replaces enumerate(g*2) - saves 2 bytes

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 198 | Initial working solution |
| v2 | 176 | Inline S, use `2in{...}` trick |
| v3 | 174 | Replace enumerate with range+modulo |

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, ~40 mutations tested. Final: **174 bytes** (-1%, -2 bytes)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 5 | range(H*2) with modular lookup vs enumerate(g*2) | 174 | -2 |

### Failed Approaches
- Direct diagonal check (boundary wrapping issues)
- Inlining H or W (worse due to multiple uses)
- Lambda/walrus combinations (longer)
- Precomputing position sets (longer)
- `abs(r-i)==1==abs(c-j)` check (longer than `2in{...}`)
