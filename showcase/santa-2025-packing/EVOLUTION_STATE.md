# Evolution State - Gen92 Complete (No Improvement)

## Current Champion
- **Gen91b** (rotation-first optimization)
- **Score: 87.29** (lower is better)
- Location: `rust/src/evolved.rs`

## Gen92 Results Summary

Generation 92 explored many mutations to improve on Gen91b's rotation-first optimization. **All mutations were rejected** - none beat the champion score of 87.29.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen92a (16 rotations) | 95.81 | 22.5° rotation granularity | REJECTED - Much worse |
| Gen92c (micro-adjust) | 88.46 | Position micro-adjustment after placement | REJECTED |
| Gen92e (N-scaled) | 88.38 | Scale search attempts by n | REJECTED |
| Gen92f (compute realloc) | 87.61 | 150 search, 35k SA iterations | REJECTED - Close |
| Gen92g (aggressive SA) | 87.90 | 150 search, 40k SA iterations | REJECTED |
| Gen92h (7 wave passes) | 89.01 | Increased wave passes from 5 to 7 | REJECTED |
| Gen92i (5 greedy passes) | 89.27 | Increased greedy backtracking from 3 to 5 | REJECTED |
| Gen92j (finer binary) | 87.72 | Binary search precision 0.0005 | REJECTED - Close |
| Gen92k (squeeze pass) | 89.06 | Final squeeze toward center | REJECTED |

## Key Learnings from Gen92

1. **More rotations hurt badly**: 16 rotations (22.5° steps) increased score from 87.29 to 95.81 - nearly 10% worse. The extra computation doesn't pay off.

2. **Parameter tuning hits a wall**: Multiple attempts at adjusting search_attempts, sa_iterations, wave_passes, and greedy passes all failed to beat 87.29.

3. **Closest attempts**:
   - Gen92f (compute realloc): 87.61 - Only 0.32 worse
   - Gen92j (finer binary): 87.72 - Only 0.43 worse

4. **Post-processing doesn't help**: Position micro-adjustment and final squeeze pass both made things worse.

## Performance Summary
- Champion (Gen91b): 87.29
- Best Gen92 attempt: 87.61 (Gen92f)
- Target (leaderboard top): ~69
- Gap to target: 26.5%

## What We Learned Overall

The Gen91b rotation-first optimization appears to be a local optimum. Simple parameter adjustments and post-processing additions don't improve it. Breaking through the 87.29 barrier may require:

1. **Fundamentally different placement strategy** - Not just parameter tuning
2. **Better initial placement** - More intelligent direction selection
3. **Different tree representation** - Maybe coordinate transformations
4. **Hybrid approaches** - Combining different algorithms

## File Locations
- Champion code: `rust/src/evolved.rs`
- Champion backup: `rust/src/evolved_champion.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`

## Next Directions (Gen93+)

Consider:
1. **Simulated annealing on placement order** - Try different tree placement sequences
2. **Gradient-free optimization** - CMA-ES or other evolutionary strategies on parameters
3. **Completely different algorithm** - Maybe hexagonal packing, force-directed layout, or other approaches
4. **Specialized handling for different n ranges** - Different strategies for small/medium/large n
