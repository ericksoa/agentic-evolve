# Score Projection Tracker

## Official Competition Winner (DO NOT MODIFY)

<!--
  ⚠️ IMPORTANT: DO NOT CHANGE THIS SECTION ⚠️

  This is the actual winning score from the Kaggle competition.
  Source: https://clist.by/standings/neurips-2025-google-code-golf-championship-optimization-custom-metric-61087802/

  This value is fixed and used to calculate our % of winner metrics.
-->

| Place | Team | Score |
|-------|------|-------|
| 1st | Code Golf International | **962,070** |
| 2nd | jailctf merger | 961,805 |
| 3rd | ox jam! | 961,784 |
| 4th | FuunAgent | 957,810 |
| 5th | HIMAGINE THE FUTURE. | 957,568 |

**Winner avg per task**: 962,070 ÷ 400 = **2,405 pts/task**

---

## Current Status

| Metric | Value |
|--------|-------|
| Tasks Solved | 41 / 400 |
| Total Score | 91,327 |
| Avg Score/Task | 2,227 |
| **% of Winner Avg** | 92.6% (2,227 ÷ 2,405) |
| Projected Final (if all 400 solved) | ~890,800 (92.6% of winner) |

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
| 1cf80156 | 130 | 2,370 | ~2,400 | 98.8% |
| 08ed6ac7 | 142 | 2,358 | ~2,400 | 98.3% |
| 09629e4f | 170 | 2,330 | ~2,380 | 97.9% |
| 239be575 | 170 | 2,330 | ~2,380 | 97.9% |
| 1b2d62fb | 58 | 2,442 | ~2,450 | 99.7% |
| 10fcaaa3 | 174 | 2,326 | ~2,380 | 97.7% |
| 1190e5a7 | 124 | 2,376 | ~2,400 | 99.0% |
| 363442ee | 144 | 2,356 | ~2,400 | 98.2% |
| 0ca9ddb6 | 207 | 2,293 | ~2,360 | 97.2% |
| 1e32b0e9 | 201 | 2,299 | ~2,360 | 97.4% |
| 00d62c1b | 219 | 2,281 | ~2,350 | 97.1% |
| 0a938d79 | 237 | 2,263 | ~2,340 | 96.7% |
| 0dfd9992 | 239 | 2,261 | ~2,340 | 96.6% |
| 0962bcdd | 241 | 2,259 | ~2,340 | 96.5% |
| 1c786137 | 249 | 2,251 | ~2,330 | 96.6% |
| 025d127b | 266 | 2,234 | ~2,320 | 96.3% |
| 32597951 | 274 | 2,226 | ~2,310 | 96.4% |
| 1caeab9d | 207 | 2,293 | ~2,360 | 97.2% |
| 178fcbfb | 217 | 2,283 | ~2,350 | 97.1% |
| 11852cab | 280 | 2,220 | ~2,300 | 96.5% |
| **Avg** | **190** | **2,308** | **2,358** | **97.9%** |

### Hard (300-600 bytes) - 8 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 05f2a901 | 326 | 2,174 | ~2,260 | 96.2% |
| 06df4c85 | 378 | 2,122 | ~2,220 | 95.6% |
| 1a07d186 | 434 | 2,066 | ~2,150 | 96.1% |
| 0b148d64 | 454 | 2,046 | ~2,150 | 95.2% |
| 2bcee788 | 465 | 2,035 | ~2,140 | 95.1% |
| 150deff5 | 494 | 2,006 | ~2,100 | 95.5% |
| a64e4611 | 523 | 1,977 | ~2,100 | 94.1% |
| 045e512c | 591 | 1,909 | ~2,050 | 93.1% |
| **Avg** | **456** | **2,042** | **2,143** | **95.3%** |

### Very Hard (600+ bytes) - 3 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 2dd70a9a | 673 | 1,827 | ~1,950 | 93.7% |
| 1b60fb0c | 933 | 1,567 | ~1,800 | 87.1% |
| 0e206a2e | 1,135 | 1,365 | ~1,500 | 91.0% |
| **Avg** | **914** | **1,586** | **1,750** | **90.6%** |

---

## Projection Model

Based on 41 solved tasks with tier distribution:

| Tier | Solved | Our Avg | Assumed # | Projected | Winner Est. |
|------|--------|---------|-----------|-----------|-------------|
| Easy | 8 | 2,435 | 180 | 438,300 | 441,000 |
| Medium | 22 | 2,318 | 140 | 324,520 | 330,120 |
| Hard | 8 | 2,042 | 60 | 122,520 | 128,580 |
| V.Hard | 3 | 1,586 | 20 | 31,720 | 35,000 |
| **Total** | **41** | **2,227** | **400** | **917,060** | **934,700** |

**Conservative estimate (current avg × 400)**: 2,227 × 400 = **890,800 points**

**Optimistic estimate (tier-weighted)**: **917,060 points** (if we maintain tier averages)

---

## Improvement Opportunities

### Low-Hanging Fruit (re-golf candidates)

Tasks with byte counts significantly above tier average:

| Task | Current | Tier Avg | Gap | Status |
|------|---------|----------|-----|--------|
| 0e206a2e | ~~1,384~~ **1,135** | 914 | +221 | ✅ Re-golfed! (-249 bytes) |
| 1b60fb0c | 933 | 914 | +19 | ✓ Near average |
| 045e512c | 591 | 456 | +135 | Medium priority |
| a64e4611 | 523 | 456 | +67 | Medium priority |

### Score Impact of Re-golfing

Remaining opportunities:
- `1b60fb0c`: 933→600 bytes = +333 points (needs README first)
- `045e512c`: 591→400 bytes = +191 points
- `a64e4611`: 523→400 bytes = +123 points

**Total potential gain: ~647 points**

### Recent Re-golf Wins
- `0e206a2e`: 1384→1135 bytes = **+249 points** (18% reduction) - 12 gens evolution
- `1bfc4729`: 406→108 bytes = **+298 points** (73% reduction)
- `0a938d79`: 539→237 bytes = **+302 points** (56% reduction)
- `1a07d186`: 635→434 bytes = **+201 points** (32% reduction)
- `150deff5`: 684→494 bytes = **+190 points** (28% reduction)
- `178fcbfb`: 304→217 bytes = **+87 points** (29% reduction) - 10 gens evolution
- `1caeab9d`: 280→207 bytes = **+73 points** (26% reduction) - 10 gens evolution
- `1190e5a7`: 188→124 bytes = **+64 points** (34% reduction) - 11 gens evolution
- `363442ee`: 205→144 bytes = **+61 points** (30% reduction) - 10 gens evolution
- `11852cab`: 333→280 bytes = **+53 points** (16% reduction) - 10 gens evolution
- `1e32b0e9`: 207→201 bytes = **+6 points** (3% reduction) - 10 gens evolution (plateau)
- `10fcaaa3`: 176→174 bytes = **+2 points** (1% reduction) - 10 gens evolution (near plateau)
- `1b2d62fb`: 170→58 bytes = **+112 points** (66% reduction) - 8 gens evolution (MASSIVE breakthrough)
- `1cf80156`: 138→130 bytes = **+8 points** (6% reduction) - 10 gens evolution (near plateau)

---

## Key Insights

1. **Easy tasks (7): 99.4%** - Nearly optimal, minimal room for improvement
2. **Medium tasks (23): 97.8%** - Good performance, 2% gap
3. **Hard tasks (8): 95.3%** - 5% gap, some byte savings possible
4. **Very Hard tasks (3): 90.6%** - 9.4% gap, improved after 0e206a2e re-golf

## Recommendations

1. **Re-golf 1b60fb0c** - Last remaining V.Hard task (933 bytes, needs README)
2. **Target 045e512c and a64e4611** - Both above Hard tier average
3. **Apply known tricks** - all() for matching, inline transforms, DFS pop(), etc.
4. **Focus on new Medium tasks** - Best effort/reward ratio for solving new tasks
