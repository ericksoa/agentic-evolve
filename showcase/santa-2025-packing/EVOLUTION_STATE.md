# Evolution State - Gen93 Complete (No Improvement)

## Current Champion
- **Gen91b** (rotation-first optimization)
- **Score: 87.29** (lower is better)
- Location: `rust/src/evolved.rs`

## Gen93 Results Summary

Generation 93 tried fundamentally different algorithmic approaches after Gen92 exhaustively tried parameter tuning. **All mutations were rejected** - none beat the champion score of 87.29.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen93a (relocate moves) | 88.52 | Remove and re-place trees during SA | REJECTED - High variance |
| Gen93c (coarse-to-fine) | 91.15 | Coarse placement + fine refinement | REJECTED - Much worse |
| Gen93e (aspect ratio 0.25) | 87.93 | Strong penalty for non-square boxes | REJECTED - Close |
| Gen93e-v2 (aspect ratio 0.15) | 88.86 | Moderate penalty | REJECTED - Worse than v1 |
| Gen93g (force-directed) | 88.41 | Physics simulation compression | REJECTED |
| Gen93i (combined search) | 88.76 | 300 attempts + 0.0005 precision | REJECTED |
| Gen93j (rotation-focused) | 88.84 | 70% rotation moves for boundary | REJECTED |

## Key Learnings from Gen93

1. **Relocate moves increase variance**: Gen93a had occasional good runs (86.59) but couldn't consistently beat champion. High variance makes it unreliable.

2. **Coarse-to-fine hurts badly**: Reducing binary search precision from 0.001 to 0.05 sped up but degraded quality significantly (91.15 vs 87.29).

3. **Aspect ratio penalty**: Original 0.10 is near optimal. Both higher (0.25) and lower (0.15) performed worse.

4. **Force-directed optimization**: Physics-based compression didn't help - the wave compaction already does this effectively.

5. **Combined parameter changes**: Increasing both search attempts (300) and precision (0.0005) didn't stack benefits - each hurt individually too.

6. **Rotation-focused SA**: Forcing more rotation moves for boundary trees didn't help - the current ~20% rotation probability is well-tuned.

## Performance Summary
- Champion (Gen91b): 87.29
- Best Gen93 attempt: 87.93 (Gen93e aspect ratio 0.25)
- Target (leaderboard top): ~69
- Gap to target: 26.5%

## Algorithm Plateau Analysis

After Gen92 (parameter tuning) and Gen93 (algorithmic changes) both failed, we've identified a fundamental plateau:

**What's Working (Gen91b)**:
- Exhaustive 8-rotation search at each position
- 5-pass wave compaction with bidirectional order
- Greedy backtracking for boundary trees
- Multi-strategy evaluation with cross-pollination

**What Doesn't Help**:
- More rotations (16 → worse)
- Finer placement (0.0005 → no improvement)
- Post-processing moves (relocate, force-directed)
- Different scoring (aspect ratio penalty)
- More iterations/attempts

## Gap to Target (26.5%)

The significant gap to leaderboard (~69) suggests top solutions use fundamentally different approaches:

1. **Non-incremental methods**: Maybe placing all trees simultaneously rather than one-by-one
2. **Different representations**: Polar coordinates, space-filling curves
3. **Global optimization**: Branch-and-bound, ILP formulations
4. **Domain-specific insights**: Exploiting the identical tree assumption more effectively

## File Locations
- Champion code: `rust/src/evolved.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`

## Next Directions (Gen94+)

At this point, incremental mutations have been exhausted. Breakthrough requires:

1. **Research leaderboard solutions**: Study what methods achieve ~69
2. **Non-greedy global optimization**: MILP, constraint programming
3. **Population-based meta-search**: Genetic algorithm over configurations
4. **Completely different paradigm**: Strip packing, guillotine cuts, etc.
