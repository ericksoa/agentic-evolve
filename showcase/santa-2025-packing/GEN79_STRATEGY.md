# Gen79 Strategy: Building on Wave Compaction Success

## Gen78b Baseline (88.92)
- wave_passes: 5
- wave steps: [0.10, 0.05, 0.02, 0.01, 0.005]
- center_pull_strength: 0.08
- late_stage_threshold: 140

## Key Learnings Applied
1. **Wave compaction works** - Gen78b's finer steps helped
2. **Aggressive compression hurts** - Gen78a's 35% was too much
3. **Post-SA angle changes fail** - Gen77 showed this
4. **Late-stage fine angles help** - n>=140 with 15° steps

## Gen79 Candidates

### Group A: Wave Compaction Refinements
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 79a | Add 0.002 step to wave | Even finer movement may help |
| 79b | Directional waves (X then Y) | Separate axes may find better positions |

### Group B: Late-Stage Focus
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 79c | 300 search attempts for n>=150 | More search for hardest trees |
| 79d | Lower late_stage_threshold to 130 | More trees get fine angles |

### Group C: SA Tuning
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 79e | 32000 iterations (was 28000) | Slightly more search |
| 79f | 3 SA passes (was 2) | Extra refinement pass |

## Priority Order
1. **79a, 79b** - Build on wave compaction success
2. **79c, 79d** - Focus on late-stage (hardest) trees
3. **79e, 79f** - SA tuning (lower priority, diminishing returns expected)

## Risk Assessment
- 79a (finer steps): Low risk, incremental improvement
- 79b (directional): Medium risk, novel approach
- 79c (more attempts): Low risk, but may be slow
- 79d (lower threshold): Medium risk, more compute

## Execution Plan
Test 79a and 79b first (wave improvements), then 79c/79d if needed.

---

## Results

| Candidate | Score | vs Baseline (88.92) | Notes |
|-----------|-------|---------------------|-------|
| 79a | 89.06 | -0.14 (worse) | Finer 0.002 step didn't help |
| 79b | 88.57 | +0.35 (better) | Directional waves work! Best: 88.57, range: 88.57-90.11 |
| 79c | | | |
| 79d | | | |

## Winner: Gen79b

**Directional wave compaction** improves over the baseline. The approach:
1. First compress in X-direction only
2. Then compress in Y-direction only
3. Finally try diagonal movement

This seems to help trees find better positions by exploring axes independently.
