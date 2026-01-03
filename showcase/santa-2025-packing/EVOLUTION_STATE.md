# Evolution State - Gen100 Complete (Sparrow Algorithm Tested)

## Current Status
- **Champion: Gen91b** (rotation-first optimization)
- **Score: ~89.6** (varies with runs, lower is better)
- **Target: ~70** (top leaderboard)
- **Gap: ~30%**

## Competition Status
- Competition ends: January 30, 2026
- Current #1: Rafbill - 69.99
- 2,412 teams participating

## Gen100 Results Summary - Sparrow Algorithm

Generation 100 implemented the **Sparrow algorithm** from recent research (arxiv.org/html/2509.13329) - a state-of-the-art approach for 2D nesting problems.

### Sparrow Key Ideas
1. **Temporary overlap tolerance** - Allow collisions, use penetration depth as continuous metric
2. **Guided Local Search** - Dynamic weights on persistently colliding pairs
3. **Two-phase architecture** - Exploration (aggressive) then Compression (refinement)

### Results (n=1-20)

| Approach | Score | vs Champion | Notes |
|----------|-------|-------------|-------|
| Evolved (champion) | 10.88 | Baseline | Greedy + SA |
| Sparrow | 12.33 | +13% worse | Pure Sparrow approach |
| Hybrid | 11.20 | +3% worse | Evolved + Sparrow refinement |

### Why Sparrow Didn't Help

The Sparrow algorithm is designed for **strip packing** (infinite length, fixed width), not **square box minimization**. Our evolved algorithm already effectively:
- Navigates the "desert of infeasibility"
- Uses simulated annealing for escape from local optima
- Applies intensive local search via wave compaction

### Files Added
- `rust/src/sparrow.rs` - Sparrow-inspired algorithm
- `rust/src/hybrid.rs` - Evolved + Sparrow refinement

---

## Gen99 Results Summary - ILP/Optimization Analysis

Generation 99 explored **non-greedy optimization approaches** including ILP, global optimization, and pattern-based packing. **All experimental approaches performed worse than the champion.**

### Key Findings from Geometry Analysis

1. **Tree Properties**:
   - Area: 0.2456 (only 35% of bounding box)
   - Height: 1.0, Width: 0.7
   - 45° rotation gives smallest bbox: 0.813 x 0.813

2. **Packing Efficiency Analysis**:
   - Current efficiency: ~56% (area used / box area)
   - Target efficiency: ~70% (what top solutions achieve)
   - Gap: ~25% improvement needed in area efficiency

3. **Side Length Scaling**:
   - If side = k * sqrt(n): current k ≈ 0.64, target k ≈ 0.59
   - Top solutions are ~10% tighter per dimension
   - This translates to ~20% better score

### Experimental Approaches Tested

| Approach | Score (n=1-20) | vs Champion | Notes |
|----------|---------------|-------------|-------|
| NFP-guided greedy (Python) | 5.54 (n=1-8) | ~Similar | Scipy optimization, NFP constraints |
| Global optimizer (Rust) | 9.19 (n=1-15) | Worse | Differential evolution, population-based |
| Pattern-based (Rust) | 15.71 (n=1-20) | Much worse | Herringbone, diamond, spiral patterns |
| Champion (evolved) | ~12.0 (n=1-20) | Baseline | Greedy + SA |

### Why Alternative Approaches Failed

1. **Global optimization**: Too many local minima, slow convergence
2. **Pattern-based**: Fixed patterns don't adapt to n, suboptimal interlocking
3. **ILP/constraint**: Computationally intractable for n>10 with non-convex polygons

### Files Created

- `python/nfp_optimizer.py` - NFP-based optimization
- `python/analyze_tree_geometry.py` - Geometry analysis
- `rust/src/global_opt.rs` - Differential evolution optimizer
- `rust/src/pattern_based.rs` - Pattern-based packing

### What Would Actually Help (Hypothesis)

1. **Commercial ILP solvers** (Gurobi, CPLEX) for small N
2. **Machine learning** trained on good packings
3. **Study winning solutions** after competition ends
4. **Problem-specific insights** we're missing about tree interlocking

## Gen98 Results Summary

Generation 98 tried **multiple optimization approaches**. All mutations were rejected - none beat the champion.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen98 (relocation) | - | Remove boundary tree, re-place elsewhere | REJECTED - Too slow |
| Gen98b (16 rotations) | 96.85 | 22.5° rotation increments instead of 45° | REJECTED - Much worse |
| Gen98c (finer search) | 88.52 | 300 attempts, 0.0005 precision | REJECTED - Similar, 33% slower |
| Gen98d (5x SA iters) | 88.72 | 140k iterations instead of 28k | REJECTED - Similar, 2x slower |

## Gen98 Key Learnings

1. **Relocation moves are too expensive**: Removing a boundary tree and re-placing it requires full placement search, making SA too slow.

2. **Non-45° rotations hurt performance**: The tree shape has natural symmetry at 45° increments. Finer angles (22.5°) create suboptimal interlocking patterns.

3. **More search doesn't help**: Increasing search attempts (200→300) and precision (0.001→0.0005) gives marginal improvement at 33% time cost.

4. **More SA iterations don't help**: The SA is already well-tuned. 5x more iterations (28k→140k) with adjusted cooling doesn't improve results.

## Cumulative Plateau Analysis (Gen92-98)

After **seven full generations** of failed attempts, we've confirmed a fundamental plateau:

**Gen92 (Parameter Tuning) - All Failed**
**Gen93 (Algorithmic Changes) - All Failed**
**Gen94 (Paradigm Shifts within Greedy) - All Failed**
**Gen95 (Global Optimization) - All Failed**
**Gen96 (Paradigm Shifts) - All Failed**
**Gen97 (Winning Solution Techniques) - All Failed**
**Gen98 (Optimization Intensification) - All Failed**

## What's Working (Gen91b Champion)
- Exhaustive 8-rotation search at each position (45° increments)
- 5-pass wave compaction with bidirectional order
- Greedy backtracking for boundary trees
- Multi-strategy evaluation with cross-pollination (6 strategies)
- Well-tuned SA parameters (temp 0.45, cooling 0.99993, 28k iters)

## What Doesn't Help (Exhaustive List)
- More rotations (16 vs 8)
- Non-45° rotation angles
- More search attempts
- Finer binary search precision
- More SA iterations
- Slower SA cooling
- Relocation moves
- Continuous angle optimization
- Separation-based packing
- Global SA on complete configuration
- Re-centering and compression
- Different scoring functions
- Multi-start optimization
- Genetic algorithms
- Hexagonal grid patterns

## Gap to Target (20-26%)

The persistent gap to leaderboard (~70) confirms top solutions use **fundamentally different approaches**:

1. **Non-greedy global optimization**: ILP, constraint satisfaction, branch-and-bound
2. **Simultaneous placement**: Place all trees at once, not incrementally
3. **Problem-specific insights**: The tree shape may have exploitable geometric properties
4. **Learning-based methods**: Neural networks trained on good packings

## File Locations
- Champion code: `rust/src/evolved.rs` (Gen91b)
- Champion backup: `rust/src/evolved_champion.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`

## Recommendation

The **greedy incremental approach has reached its fundamental limit**. Seven generations of mutations have failed to improve on Gen91b. Further progress requires:

1. **ILP formulation** with commercial solvers (Gurobi, CPLEX)
2. **Complete algorithm redesign** (non-incremental placement)
3. **Wait for competition end** to study winning solutions

The evolution has plateaued at 20-26% gap to target.
