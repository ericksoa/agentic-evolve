# Task 39a8645d

## Pattern
Find the shape (8-connected component) that appears most frequently across the grid and output it with its original color.

## Algorithm
1. Use 8-connectivity flood fill to identify all connected components
2. Normalize each shape by translating to origin (subtract min row/col)
3. Count occurrences of each unique normalized shape using a dictionary
4. Find the shape with highest count using max with key function
5. Output the winning shape using list comprehension with membership check

## Key Tricks
- `D=-1,0,1` tuple alias saves 1 byte over inline tuples
- `P+=(i,j),` trailing comma for tuple append
- `S|={(i,j)}` set union for add (shorter than `.add()`)
- `zip(*P)` to separate Y,X coordinates efficiently
- `frozenset((r-min(Y),c-min(X))for r,c in P)` for normalized shape key
- `v*((r,c)in P)` multiplication for conditional fill
- `R(min(X),max(X)+1)` range directly in comprehension
- Combined count and normalize in single pass

## Evolution Summary
24 generations, ~40 mutations tested. Final: **526 bytes** (-19.5%, -127 bytes from 653)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1 | Basic structure | 651 | -2 |
| 2 | m helper lambda | 633 | -18 |
| 3 | max(G.values(),key=len) | 624 | -9 |
| 4 | R=range alias | 620 | -4 |
| 6 | S\|={(i,j)} shorter than .add() | 618 | -2 |
| 8 | D=-1,0,1 tuple alias | 617 | -1 |
| 10 | Inline count during loop | 602 | -15 |
| 11 | max with key function | 600 | -2 |
| 12 | zip(*P) for Y,X separation | 587 | -13 |
| 13 | Cache min(Y),min(X) | 585 | -2 |
| 14 | Remove m lambda, inline | 577 | -8 |
| 15 | Cache n in A, simple max key | 561 | -16 |
| 16 | List comp output with v*((r,c)in P) | 548 | -13 |
| 17 | zip in normalize expression | 534 | -14 |
| 19 | Inline min(X),min(Y) in return | 526 | -8 |

### Failed Approaches
- frozenset alias (only used once)
- set for P (adds bytes for conversion)
- Walrus operator in loop (syntax issues)
- Counter from collections (import overhead)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 653 | Initial working solution |
| v10 | 602 | Inline count during loop |
| v15 | 561 | Cache n, simple max key |
| v19 | 526 | Final optimized solution |
