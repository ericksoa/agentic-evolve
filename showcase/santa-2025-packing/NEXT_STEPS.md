# Next Steps for Santa 2025 Packing Evolution

**Last Updated**: Gen86 complete (champion unchanged at 87.36)
**Current Champion**: Gen84c (87.36)

## Quick Start

```bash
cd rust
cargo build --release
./target/release/benchmark  # Test current champion
```

## Current Champion: Gen84c (87.36)

Key innovation: **Extreme 4+1 bidirectional wave split**
- First 4 waves: outside-in (far trees first)
- Last 1 wave: inside-out (close trees first)

## Recent Evolution History

### Gen86 Results (ALL REJECTED)

| Candidate | Strategy | Best Score | Delta | Status |
|-----------|----------|------------|-------|--------|
| 86a | Adaptive step sizes by tree count | 88.59 | +1.23 | REJECTED |
| 86b | Phase-specific ordering (H/V asymmetric) | 88.94 | +1.58 | REJECTED |
| 86c | Graduated 3-1-1 split (mixed middle) | 88.58 | +1.22 | REJECTED |
| 86d | Enhanced 8-directional diagonal | 89.09 | +1.73 | REJECTED |

**Key learnings from Gen86**:
- Adaptive/dynamic parameters hurt consistency
- Asymmetric phase ordering reduces packing efficiency
- Mixed-ordering middle wave adds noise without benefit
- More directions in diagonal phase = diminishing returns + overhead

### Gen85 Results (ALL REJECTED)

| Candidate | Strategy | Best Score | Delta | Status |
|-----------|----------|------------|-------|--------|
| 85a | 6 waves with 5+1 split | 88.30 | +0.94 | REJECTED |
| 85b | Larger step sizes [0.15, 0.08, ...] | 89.36 | +2.00 | REJECTED |
| 85c | 7 waves with 6+1 split | 89.56 | +2.20 | REJECTED |
| 85d | Stronger center pull (0.12) | 88.23 | +0.87 | REJECTED |

**Key learnings from Gen85**:
- More waves hurt (6 and 7 both regressed)
- Larger steps hurt precision
- Stronger center pull disrupts balance
- 5 waves with 4+1 remains optimal

## Gen87 Plan: Fundamentally Different Approaches

Gen85-86 exhausted parameter variations on wave compaction. Time for structural changes.

### Candidates to Explore

#### 87a: Rotation-Aware Wave Compaction
- During diagonal phase, try rotating trees to unlock tighter fit
- Only accept rotation if it reduces bounding box
- Hypothesis: Trees may "interlock" better with slight rotations

#### 87b: Density-Guided Tree Ordering
- Instead of distance-from-center, sort by local density
- Process trees in sparse regions first (more room to move)
- Hypothesis: Sparse-first may create cascading improvements

#### 87c: Axis-Aligned Compression
- Replace diagonal phase with sequential X then Y compression
- Move all trees left to tight boundary, then all trees down
- Hypothesis: Separating X/Y may avoid diagonal deadlocks

#### 87d: Greedy Backtracking Wave
- After wave compaction, try moving boundary trees inward aggressively
- If overlap, try rotating, then backtrack
- Hypothesis: Post-wave greedy pass may find missed opportunities

### Alternative Research Directions

1. **Initial placement quality**: Can we place trees better initially?
2. **Rotation search during placement**: Binary search on angle too?
3. **Gap detection + targeted movement**: Find voids and pull trees toward them
4. **Boundary-aware SA**: Focus SA iterations on boundary trees only

## Score Progression (for reference)

| Gen | Split | Score | Notes |
|-----|-------|-------|-------|
| 62 | N/A (radius) | 88.22 | Original best |
| 80b | 5+0 | 88.44 | All outside-in |
| 82a | 0+5 | 88.62 | All inside-out |
| 83a | 3+2 | 88.22 | First crossover success |
| **84c** | **4+1** | **87.36** | **CURRENT BEST** |
| 85a | 5+1 | 88.30 | Too many waves |
| 85d | 4+1 stronger | 88.23 | Center pull too strong |
| 86a | adaptive | 88.59 | Dynamic steps hurt |
| 86c | 3-1-1 | 88.58 | Mixed wave no help |

## What Works (Don't Break These)

1. **Discrete 45 deg angles in SA** - Continuous angles break the framework
2. **6 parallel placement strategies** - Diversity helps
3. **Hot restarts from elite pool** - Escape local optima
4. **Bidirectional wave processing** - Better than single direction
5. **Cardinal phase order** (R->L->U->D->diagonal) - Optimal sequence
6. **4+1 outside-in:inside-out ratio** - Optimal split found
7. **5 wave passes** - Sweet spot, more hurts
8. **Step sizes [0.10, 0.05, 0.02, 0.01, 0.005]** - Fine-grained works

## What Doesn't Work (Avoid)

1. More SA iterations (diminishing returns)
2. Finer step sizes in wave compaction (too fine = noise)
3. More than 5 wave phases (undoes positioning)
4. Alternating O-I-O-I-O pattern (88.36, worse than 4+1)
5. Inverse split 2+3 (88.39, order matters)
6. Higher compression probability (88.98, 35% hurt)
7. Adaptive step sizes (88.59, inconsistent)
8. Phase-specific ordering (88.94, asymmetry hurts)
9. Mixed middle wave (88.58, noise without benefit)
10. 8-directional diagonal (89.09, overhead outweighs benefit)
11. Larger steps [0.15, 0.08, ...] (89.36, too coarse)
12. More waves (6 or 7) (88.30-89.56, worse)
13. Stronger center pull 0.12 (88.23, disrupts balance)

## Files to Reference

- `mutations/gen84c_extreme_split.rs` - Current champion
- `mutations/gen83a_bidirectional_wave.rs` - Previous best crossover
- `mutations/gen86*.rs` - Latest rejected experiments
- `GEN84_STRATEGY.md` - Full analysis of split ratios
- `GEN83_STRATEGY.md` - Crossover breakthrough details

## Commands Reference

```bash
# Test a mutation
cp mutations/gen87a_xxx.rs rust/src/evolved.rs
cd rust && cargo build --release && ./target/release/benchmark

# Generate visualization
./target/release/visualize
open packing_n200.svg

# Submit to Kaggle
./target/release/submit
kaggle competitions submit -c santa-2025 -f submission.csv -m "Gen87a: description"
```

## Target

- **Current**: 87.36
- **Leaderboard top**: ~69
- **Gap**: 26.6%

Each 1-point improvement is significant progress toward the leaderboard.
