# Task 228f6490

## Pattern
Match colored marker shapes to holes in 5-bordered rectangles, swap them.

## Algorithm
Find all colored regions (not 0 or 5) and compute their relative shapes. For each
color, search through all possible 5-bordered rectangles (4 nested loops: top, left,
bottom, right). When we find a rectangle where all 4 borders are exactly 5s and the
internal 0-cell pattern matches the color's shape, copy the color into the holes and
clear the original marker positions.

## Key Tricks
- `{*sum(g,[])}-{0,5}` to get all unique colors (flattening with sum)
- `exec()` inside list comprehension to execute statements without explicit loops
- Set union `{*g[a][b:d+1],*g[c][b:d+1]}|{...}` for border check
- `sorted((r-P[0][0],c-P[0][1])for r,c in P)` for relative shape normalization
- `g[r][k]<1` instead of `g[r][k]==0` for zero check

## Evolution Summary (AlphaEvolve-Inspired)

16 generations, ~35 mutations tested. Final: **520 bytes** (-25%, -177 bytes from 697)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1b | Combined all() checks with set union | 635 | -62 |
| 3b | Slice notation for row borders | 596 | -39 |
| 4b | Color-first loop structure | 578 | -18 |
| 6a | `{*sum(g,[])}` flattening trick | 552 | -26 |
| 8b | exec() in list comprehension | 527 | -25 |
| 10a | Flat list comprehension (remove nested) | 521 | -6 |
| 11a | Set spread with union operator | 520 | -1 |

### Failed Approaches
- Walrus operator (scope issues with exec)
- `*` instead of `and` (same or worse)
- Inlining R,C (longer due to repetition)
- Explicit for loops (more bytes than exec trick)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 697 | Initial 4-nested-loop solution |
| v2 | 635 | Combined border checks |
| v3 | 578 | Color-first restructure |
| v4 | 552 | Flatten with sum trick |
| v5 | 520 | exec() + list comp + set union |
