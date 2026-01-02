# Gen80 Strategy: Metacognitive Analysis

## Recent Breakthrough Trends

### What's Working (Gen78-79)
| Gen | Score | Innovation | Why It Worked |
|-----|-------|------------|---------------|
| 78b | 88.92 | 5 wave passes + 0.005 step | More passes = more settling opportunities |
| 79b | 88.57 | Directional waves (X→Y→diagonal) | Decoupling axes finds blocked positions |

### Pattern Recognition
1. **Wave compaction is the frontier** - Last 2 improvements came from wave refinements
2. **Decomposition helps** - Breaking movement into components (X, Y, diagonal) improved results
3. **Granularity sweet spot** - 0.005 step helped, 0.002 didn't (too fine = noise)
4. **SA is stable** - Changes to SA parameters consistently hurt

### Metacognitive Questions

**Q1: What assumptions are we making?**
- Assumption: Trees should move toward geometric center
- Challenge: Maybe optimal center is off-center (asymmetric packing)?
- Assumption: All trees should use same wave parameters
- Challenge: Maybe different trees need different treatment?

**Q2: What's the gap to top solutions?**
- Top: ~69, Us: 88.57, Gap: 28%
- Top uses: continuous angles, global rotation, different algorithm
- Our SA framework doesn't handle continuous angles well

**Q3: What decompositions haven't we tried?**
- We did: X, Y, diagonal (3-phase)
- Untried: 4 cardinal directions (up/down/left/right)
- Untried: Radial directions (inward rays from 8 angles)
- Untried: Concentric shells (inner trees first, then outer)

**Q4: What inversions might help?**
- Current: Compress outside-in (far trees first)
- Inversion: Compress inside-out (close trees first)?
- Current: Same steps for all waves
- Inversion: Different steps per wave (aggressive→gentle)?

## Gen80 Candidates

### Group A: Wave Order Experiments
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 80a | Y→X→diagonal (reverse XY order) | Maybe Y-compression first is better for tree shape |
| 80b | 4-direction: up/down/left/right separately | Cardinal directions may unlock blocked moves |

### Group B: Wave Strategy Experiments
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 80c | Inside-out wave (close trees first) | Settling center first may create better structure |
| 80d | Progressive steps: [0.15, 0.08, 0.04, 0.02, 0.01] per wave | Aggressive→gentle may help |

### Group C: Tree-Specific Treatment
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 80e | Extra wave passes for n>=180 (last 10%) | Hardest trees need more compaction |
| 80f | Smaller step sizes for boundary trees | Fine-tune bbox-defining trees |

## Priority Order
1. **80a, 80b** - Wave order experiments (builds on 79b success)
2. **80c, 80d** - Wave strategy (novel approaches)
3. **80e, 80f** - Tree-specific (lower priority)

## Risk Assessment
- 80a: Low risk - simple order change
- 80b: Medium risk - more phases = more compute
- 80c: Medium risk - inverts proven strategy
- 80d: Low risk - parameter tuning

## Execution Plan
Test 80a and 80b first (wave order), then 80c if time permits.

---

## Results

| Candidate | Score | vs Baseline (88.57) | Notes |
|-----------|-------|---------------------|-------|
| 80a | 88.52 | +0.05 (marginal) | Y→X→diagonal order - slight improvement |
| **80b** | **88.44** | **+0.13** | **4 cardinal (R→L→U→D→diag) - NEW BEST!** |
| 80c | - | - | Not tested |
| 80d | - | - | Not tested |

## Winner: Gen80b

**4-cardinal wave compaction** outperforms the 3-phase approaches:
1. Move RIGHT (trees left of center move right)
2. Move LEFT (trees right of center move left)
3. Move UP (trees below center move up)
4. Move DOWN (trees above center move down)
5. Diagonal movement (final polish)

**Key insight**: More granular directional control (4 phases vs 2-3) finds additional compaction opportunities. The improvement continues the trend of decomposition helping.
