# Evolution State - Gen96 Complete (Plateau Confirmed)

## Current Champion
- **Gen91b** (rotation-first optimization)
- **Score: ~87-88** (varies with runs, lower is better)
- Location: `rust/src/evolved.rs`

## Gen96 Results Summary

Generation 96 tried **fundamentally different paradigms** (separation-based packing, continuous angles) after Gen92-95's incremental improvements all failed. **All mutations were rejected** - none beat the champion.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen96 (separation-based) | 88.42 | Dense placement → separate → compact | REJECTED - Worse & 2x slower |
| Gen96b (relaxed SA) | TIMEOUT | Allow temporary overlaps, then resolve | REJECTED - Too slow |
| Gen96c (continuous angles) | 88.43 | Continuous angle polish for boundary trees | REJECTED - No improvement |

## Gen96 Key Learnings

1. **Separation-based packing doesn't help**: Starting with dense overlapping placement and separating trees expands the packing too much, worse than greedy incremental.

2. **Relaxed SA is too slow**: Allowing temporary overlaps during SA and resolving them later is computationally infeasible.

3. **Continuous angles don't help**: Fine-grained rotation optimization beyond the 45° discrete steps doesn't improve boundary tree positions.

---

## Gen95 Results Summary

Generation 95 tried **global optimization approaches** after Gen92-94's local optimization approaches all failed. **All mutations were rejected** - none beat the champion.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen95e (annealing overhaul) | 89.39 | Temp 2.0, cooling 0.99998, 100k iters | REJECTED - Worse than champion |
| Gen95a (full config SA) | 88.69 | SA on complete configuration after placement | REJECTED - Worse than champion |
| Gen95c (global rotation) | 88.58 | Optimize all rotations together, positions fixed | REJECTED - Worse than champion |
| Gen95d (center-first) | 88.57 | Re-center packing and compress from new origin | REJECTED - Worse than champion |

## Gen95 Key Learnings

1. **More aggressive SA doesn't help**: Gen95e with 4.4x higher temperature and 3.6x more iterations actually performed worse (89.39 vs ~87-88). The existing SA schedule is well-tuned.

2. **Global SA on complete config doesn't help**: Gen95a's approach of running SA on all trees simultaneously after incremental placement didn't improve scores. The incremental optimization during placement already finds good configurations.

3. **Decoupled rotation optimization doesn't help**: Gen95c's idea of optimizing rotations separately while keeping positions fixed didn't work. The current exhaustive rotation search at placement time (Gen91b's key innovation) already handles this well.

4. **Re-centering doesn't help**: Gen95d's center-of-mass re-centering and compression didn't improve the bounding box. The existing wave compaction already effectively centers the packing.

## Performance Summary
- Champion (Gen91b): ~87-88 (varies with runs)
- Best Gen95 attempt: 88.57 (Gen95d)
- Target (leaderboard top): ~69
- Gap to target: 26-28%

## Cumulative Plateau Analysis (Gen92-96)

After **five full generations** of failed attempts, we've confirmed a fundamental plateau:

**Gen92 (Parameter Tuning) - All Failed**:
- More rotations, finer precision, more iterations
- Different wave passes, SA temperatures, cooling rates

**Gen93 (Algorithmic Changes) - All Failed**:
- Relocate moves, coarse-to-fine, aspect ratio penalty
- Force-directed compression, combined parameters

**Gen94 (Paradigm Shifts within Greedy) - All Failed**:
- Multi-start (high variance, unreliable)
- Hexagonal grid seeding
- Genetic algorithm (crossover creates overlaps)

**Gen95 (Global Optimization) - All Failed**:
- Annealing schedule overhaul (more exploration)
- Full configuration SA (optimize all at once)
- Global rotation optimization (decouple position/rotation)
- Center-first placement (re-center and compress)

**Gen96 (Paradigm Shifts) - All Failed**:
- Separation-based packing (dense start → push apart → compact)
- Relaxed SA (allow temporary overlaps, resolve later) - too slow
- Continuous angle optimization (fine-grained rotation beyond 45°)

## What's Working (Gen91b)
- Exhaustive 8-rotation search at each position
- 5-pass wave compaction with bidirectional order (4 outside-in + 1 inside-out)
- Greedy backtracking for boundary trees
- Multi-strategy evaluation with cross-pollination
- Well-tuned SA parameters (temp 0.45, cooling 0.99993, 28k iters)

## What Doesn't Help
- More rotations, finer precision, more iterations
- Post-processing moves (relocate, force-directed)
- Different scoring functions (aspect ratio penalty)
- Multi-start optimization (high variance)
- Genetic algorithms (crossover creates overlaps)
- Alternative grid patterns (hexagonal)
- More aggressive SA (higher temp, slower cooling)
- Global SA on complete configuration
- Decoupled rotation optimization
- Re-centering and compression
- Separation-based packing (dense start → push apart → compact)
- Continuous angle optimization (fine-grained rotation beyond 45°)

## Gap to Target (26-28%)

The significant and persistent gap to leaderboard (~69) suggests top solutions use **fundamentally different approaches** that we haven't tried:

1. **Non-greedy global optimization**: Branch-and-bound, integer linear programming (ILP), constraint satisfaction
2. **Problem-specific geometric insights**: The Christmas tree shape may have exploitable symmetries or packing patterns
3. **Simultaneous placement**: Place all trees at once rather than incrementally
4. **Learning-based methods**: Train a neural network on good packings

## File Locations
- Champion code: `rust/src/evolved.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`

## Recommendation

At this point, **all local optimization approaches AND paradigm shifts have been exhaustively tried**. Five generations (92-96) of mutations have failed to improve on Gen91b. The framework has reached its fundamental limit.

Further progress likely requires:

1. **Integer Linear Programming (ILP)**: Formulate as optimization problem with commercial solvers
2. **Simultaneous placement**: Place all trees at once, not incrementally
3. **Learn from winners**: Study published Kaggle solutions after competition ends
4. **Different geometry**: The ~69 score solutions likely exploit tree shape properties we haven't discovered

The evolution has plateaued at 26-28% gap to target. The greedy incremental approach with local search cannot close this gap.
