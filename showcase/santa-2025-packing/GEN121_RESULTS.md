# Gen121 Results

## Summary
- **Starting score**: 85.10 (Gen120)
- **Final score**: 85.10 (no improvement)
- **Goal**: ~69.02 (top leaderboard)
- **Gap**: Still 23.3%

## What Was Tried

### 1. Evolve SDK - Novel Algorithm Discovery
Used the Evolve SDK to generate and evaluate 6 different packing algorithms:

| Approach | Fitness | Description |
|----------|---------|-------------|
| Bottom-Left-Fill (champion) | 27.71 | Fast BLF with optimized data structures |
| Greedy Spiral | 17.67 | Golden-ratio spiral placement |
| Grid-Based DP | 16.36 | Discrete grid + dynamic programming |
| Physics-Based Simulation | 14.62 | Force-directed tree pushing |
| Genetic Algorithm | 14.59 | Evolve positions/orientations |
| Multi-Scale SA | 13.37 | Simulated annealing with temp schedules |

**Results**:
- Ran 5 generations
- Tried 20 mutations, ~15 valid
- Champion (gen0_d.py - Bottom-Left-Fill) never improved
- Stopped at plateau threshold (4 gens without improvement)

**Key Finding**: Novel algorithms from LLM-generated mutations were worse than our evolved Rust algorithm. The Rust solution's combination of greedy + SA + wave compaction is hard to beat.

### 2. Best-of-100 Variance Exploitation
Ran the Rust solver 100 times with 4 threads to exploit stochastic variance:

| Metric | Value |
|--------|-------|
| First run score | 89.89 |
| Best-of-100 score | 85.39 |
| Improvement from variance | +5.01% |
| N values improved | 199/200 |
| Compute time | 93 minutes |
| CPU threads | 4 (RAYON_NUM_THREADS=4) |

**Results**:
- Best-of-100 scored 85.39 (vs our 85.10)
- The 85.39 is Rust-only; doesn't include Python refinements
- Shows variance exploitation works - almost every n improved!
- But 100 runs wasn't enough to beat Python-refined solutions

### 3. Lightning.ai Cloud Best-of-100
Tested running the Rust solver on lightning.ai cloud compute to enable massive parallelism:

| Metric | Local | Cloud |
|--------|-------|-------|
| Score | 85.39 | 85.53 |
| Time | 93 min | 384 min (6.4h) |
| CPU | M3 Mac, 4 threads | Intel Xeon, 4 cores |
| Throughput | 1.07 runs/min | 0.26 runs/min |

**Results**:
- Cloud is **4.1x slower** than local M3 Mac
- 27% CPU steal time from VM sharing
- Score slightly worse due to variance (85.53 vs 85.39)
- Lightning.ai CPU unsuitable for this compute-bound workload

**Key Finding**: Cloud CPUs with shared resources cannot compete with dedicated local hardware for this workload. For cloud scaling, would need many parallel studios (not more runs per studio).

## Key Learnings

1. **Evolved Rust algorithm is near-optimal**: Novel LLM-generated algorithms (BLF, spiral, physics-based, GA) all scored worse than our evolved greedy + SA + wave compaction approach.

2. **Variance exploitation has diminishing returns**: Best-of-20 to Best-of-100 shows improvement but doesn't beat Python-refined solutions. Need either:
   - More runs (1000+)
   - Python refinement on Best-of-100 output

3. **Python refinements are crucial**: The gap from 85.39 (Rust-only) to 85.10 (Rust+Python) shows CMA-ES and full-config SA add real value.

4. **Cloud compute has overhead**: Lightning.ai CPU studios have ~27% steal time and slower per-core performance, making them 4x slower than local M3 Mac. Cloud is only beneficial when parallelizing across many independent studios, not for single-studio workloads.

5. **23% gap requires paradigm shift**: Neither brute-force variance nor novel algorithms bridged the gap to top solutions (~69). The leaders likely use:
   - Constraint programming (ILP/CP-SAT)
   - Learned heuristics trained on many examples
   - Problem-specific geometric insights we haven't discovered

## Files Created
- `evolve_config.json` - Evolve SDK configuration
- `evaluate_santa.py` - Evaluation script (handles PlacedTree + tuple formats)
- `python/gen121_starter.py` - Starter solution for evolution
- `.evolve-sdk/evolve_better_christmas_tree_p/` - Evolution state and mutations
- `run_bestof_lightning.py` - Lightning.ai cloud execution script
- `rust_solver.tar.gz` - Packaged Rust solver for cloud upload

## Resource Management
**CRITICAL**: Used max 4 parallel threads (RAYON_NUM_THREADS=4) to avoid overwhelming the system. This rule is now documented in:
- `~/CLAUDE.md` (global)
- `/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/CLAUDE.md` (project)

## Recommendations for Future Generations

1. **Try OR-Tools CP-SAT** for exact small n optimization
2. **Combine Best-of-100 with Python refinement** - run CMA-ES/SA on the 85.39 solution
3. **Study 70.1 solution approach** more deeply - continuous angles + global rotation
4. **Consider tree shape exploitation** - the concave regions might allow nesting
5. **Try hierarchical packing** - optimal small groups merged together
