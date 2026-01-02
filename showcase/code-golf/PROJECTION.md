# Score Projection Tracker

## Current Status

| Metric | Value |
|--------|-------|
| Tasks Solved | 41 / 400 |
| Total Score | 90,612 |
| Avg Score/Task | 2,210 |
| Projected Final | ~884,000 (91.9% of winner) |

---

## Tasks by Difficulty

### Easy (< 100 bytes) - 7 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 0520fde7 | 57 | 2,443 | ~2,450 | 99.7% |
| 0d3d703e | 58 | 2,442 | ~2,450 | 99.7% |
| 007bbfb7 | 65 | 2,435 | ~2,450 | 99.4% |
| 29c11459 | 68 | 2,432 | ~2,450 | 99.3% |
| 1e0a9b12 | 69 | 2,431 | ~2,450 | 99.2% |
| 27a28665 | 70 | 2,430 | ~2,450 | 99.2% |
| 017c7c7b | 80 | 2,420 | ~2,445 | 99.0% |
| **Avg** | **67** | **2,433** | **2,449** | **99.4%** |

### Medium (100-300 bytes) - 21 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 1bfc4729 | 108 | 2,392 | ~2,420 | 98.8% |
| 05269061 | 113 | 2,387 | ~2,420 | 98.6% |
| 137eaa0f | 130 | 2,370 | ~2,400 | 98.8% |
| 1cf80156 | 138 | 2,362 | ~2,400 | 98.4% |
| 08ed6ac7 | 142 | 2,358 | ~2,400 | 98.3% |
| 09629e4f | 170 | 2,330 | ~2,380 | 97.9% |
| 239be575 | 170 | 2,330 | ~2,380 | 97.9% |
| 1b2d62fb | 170 | 2,330 | ~2,380 | 97.9% |
| 10fcaaa3 | 176 | 2,324 | ~2,380 | 97.6% |
| 1190e5a7 | 188 | 2,312 | ~2,370 | 97.6% |
| 363442ee | 205 | 2,295 | ~2,360 | 97.2% |
| 0ca9ddb6 | 207 | 2,293 | ~2,360 | 97.2% |
| 1e32b0e9 | 207 | 2,293 | ~2,360 | 97.2% |
| 00d62c1b | 219 | 2,281 | ~2,350 | 97.1% |
| 0a938d79 | 237 | 2,263 | ~2,340 | 96.7% |
| 0dfd9992 | 239 | 2,261 | ~2,340 | 96.6% |
| 0962bcdd | 241 | 2,259 | ~2,340 | 96.5% |
| 1c786137 | 249 | 2,251 | ~2,330 | 96.6% |
| 025d127b | 266 | 2,234 | ~2,320 | 96.3% |
| 32597951 | 274 | 2,226 | ~2,310 | 96.4% |
| 1caeab9d | 280 | 2,220 | ~2,300 | 96.5% |
| **Avg** | **187** | **2,303** | **2,359** | **97.6%** |

### Hard (300-600 bytes) - 10 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 178fcbfb | 304 | 2,196 | ~2,280 | 96.3% |
| 05f2a901 | 326 | 2,174 | ~2,260 | 96.2% |
| 11852cab | 333 | 2,167 | ~2,250 | 96.3% |
| 06df4c85 | 378 | 2,122 | ~2,220 | 95.6% |
| 1a07d186 | 434 | 2,066 | ~2,150 | 96.1% |
| 0b148d64 | 454 | 2,046 | ~2,150 | 95.2% |
| 2bcee788 | 465 | 2,035 | ~2,140 | 95.1% |
| 150deff5 | 494 | 2,006 | ~2,100 | 95.5% |
| a64e4611 | 523 | 1,977 | ~2,100 | 94.1% |
| 045e512c | 591 | 1,909 | ~2,050 | 93.1% |
| **Avg** | **430** | **2,070** | **2,170** | **95.4%** |

### Very Hard (600+ bytes) - 3 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 2dd70a9a | 673 | 1,827 | ~1,950 | 93.7% |
| 1b60fb0c | 933 | 1,567 | ~1,800 | 87.1% |
| 0e206a2e | 1,384 | 1,116 | ~1,400 | 79.7% |
| **Avg** | **997** | **1,503** | **1,717** | **87.5%** |

---

## Projection Model

Based on 41 solved tasks with tier distribution:

| Tier | Solved | Our Avg | Assumed # | Projected | Winner Est. |
|------|--------|---------|-----------|-----------|-------------|
| Easy | 7 | 2,433 | 180 | 437,940 | 440,820 |
| Medium | 21 | 2,303 | 140 | 322,420 | 330,260 |
| Hard | 10 | 2,070 | 60 | 124,200 | 130,200 |
| V.Hard | 3 | 1,503 | 20 | 30,060 | 34,340 |
| **Total** | **41** | **2,210** | **400** | **914,620** | **935,620** |

**Conservative estimate (current avg × 400)**: 2,210 × 400 = **884,000 points**

**Optimistic estimate (tier-weighted)**: **914,620 points** (if we maintain tier averages)

---

## Improvement Opportunities

### Low-Hanging Fruit (re-golf candidates)

Tasks with byte counts significantly above tier average:

| Task | Current | Tier Avg | Gap | Potential Savings |
|------|---------|----------|-----|-------------------|
| 0e206a2e | 1,384 | 997 | +387 | High priority |
| 1b60fb0c | 933 | 997 | -64 | ✓ Near average |
| 045e512c | 591 | 430 | +161 | Medium priority |
| a64e4611 | 523 | 430 | +93 | Medium priority |

### Score Impact of Re-golfing

If we could reduce:
- `0e206a2e`: 1384→600 bytes = +784 points
- `1b60fb0c`: 933→600 bytes = +333 points
- `045e512c`: 591→400 bytes = +191 points

**Total potential gain: ~1,300+ points**

### Recent Re-golf Wins
- `1bfc4729`: 406→108 bytes = **+298 points** (73% reduction)
- `0a938d79`: 539→237 bytes = **+302 points** (56% reduction)
- `1a07d186`: 635→434 bytes = **+201 points** (32% reduction)
- `150deff5`: 684→494 bytes = **+190 points** (28% reduction)

---

## Key Insights

1. **Easy tasks: 99.4%** - Nearly optimal, minimal room for improvement
2. **Medium tasks: 97.6%** - Good performance, 2-3% gap
3. **Hard tasks: 95.4%** - 5% gap, some byte savings possible
4. **Very Hard tasks: 87.5%** - 12.5% gap, major rework needed for 0e206a2e

## Recommendations

1. **Re-golf very hard tasks** - 0e206a2e (1384 bytes) needs major algorithm rethink
2. **Target 045e512c and a64e4611** - Both 100+ bytes above Hard tier average
3. **Apply known tricks** - 645 bit mask, multiplication instead of ternary, etc.
4. **Focus on new Medium tasks** - Best effort/reward ratio for solving new tasks
