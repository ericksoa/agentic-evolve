# Task 3bdb4ada: Middle Row Stripe Pattern

## Pattern
Find 3-row colored rectangles and apply alternating stripe pattern to middle row.

## Algorithm
1. For each row r (up to len-2), scan for consecutive colored cells
2. Check if 3 rows (r, r+1, r+2) have identical colored rectangles
3. In the middle row (r+1), set every odd position (1, 3, 5...) to 0

## Key Tricks
| Trick | Saves | Description |
|-------|-------|-------------|
| `eval(str(g))` | 4 bytes | Deep copy vs `[r[:]for r in g]` |
| `w:=g[r]` walrus | 1 byte | Assign in condition |
| `w[e:e+1]==[v]` | 2 bytes | Bounds-safe slice comparison |
| `range(c+1,e,2)` | 2 bytes | Step 2 vs manual `x-c&1` check |
| `c=e or c+1` | 3 bytes | Handle zero-extent gracefully |

## Evolution Summary
| Gen | Best | Key Discovery |
|-----|------|---------------|
| 0 | 517 | Initial working solution |
| 1 | 264 | `eval(str(g))` deep copy |
| 2 | 260 | Remove R variable |
| 3 | 258 | Slice comparison `w[e:e+1]==[v]` |
| 5 | 251 | Chain comparison `w[c:e]==g[r+1][c:e]==g[r+2][c:e]` |
| 7 | 249 | Step 2 in range: `range(c+1,e,2)` |
| 8 | 245 | Remove C variable |
| 9 | 241 | Enumerate approach |
| 10 | 240 | Inline w assignment |
| 11 | 239 | Walrus operator `w:=g[r]` |

## Byte History
| Version | Bytes | Score |
|---------|-------|-------|
| Initial | 517 | 1,983 |
| Gen 1 | 264 | 2,236 |
| Gen 5 | 251 | 2,249 |
| Gen 7 | 249 | 2,251 |
| Gen 11 | 239 | 2,261 |

**Final: 239 bytes (+2,261 points)**
