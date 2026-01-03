# Gen93+ Evolution Plan: Breaking the Local Optimum

## Current State
- **Champion**: Gen91b (rotation-first optimization)
- **Best Score**: 87.29
- **Target**: ~69 (leaderboard top)
- **Gap**: ~26.5%

## What's Been Exhaustively Tried (Don't Repeat)

### Gen92 (All Failed)
- 16 rotations (22.5°): 95.81 - MUCH WORSE
- Position micro-adjustment: 88.46
- N-scaled search attempts: 88.38
- Compute reallocation (150/35k): 87.61
- Aggressive SA (150/40k): 87.90
- 7 wave passes: 89.01
- 5 greedy passes: 89.27
- Finer binary search (0.0005): 87.72
- Final squeeze pass: 89.06

### Gen91 (Previously Tried)
- Size-ordered placement: SKIPPED (all trees identical)
- BLF hybrid: 88.62
- SA temp tuning (0.45→0.60): 87.64
- SA iterations (28k→35k): 88.32
- More search attempts (200→300): 87.99

## Gen93 Strategy: Fundamentally Different Approaches

Since parameter tuning hit a wall, try algorithmic changes:

### Phase 1: Placement Order Optimization
**Rationale**: Maybe the sequential placement order matters. Currently placing in order of strategy, but what if we could find better orders?

**Gen93a: Randomized placement order**
- Instead of always placing tree n after trees 1..n-1
- Try random shuffling of placement order within SA
- May find configurations unreachable by sequential placement

**Gen93b: Reverse placement order**
- Try placing from high n to low n
- Then re-optimize with SA

### Phase 2: Multi-Scale Optimization
**Rationale**: Current approach optimizes at one scale. Try coarse-to-fine.

**Gen93c: Coarse placement + refinement**
- First place all trees with low precision (0.1 tolerance)
- Then run fine-grained SA to refine positions
- May escape local optima

**Gen93d: Cluster-based placement**
- Group trees into clusters
- Optimize cluster positions
- Then optimize within clusters

### Phase 3: Different Scoring Functions
**Rationale**: Current scoring may not capture true optimality.

**Gen93e: Aspect ratio penalty**
- Add strong penalty for non-square bounding boxes
- May encourage more balanced layouts

**Gen93f: Convex hull scoring**
- Score based on convex hull area instead of bounding box
- Different optimization landscape

### Phase 4: Hybrid Algorithms
**Rationale**: Combine multiple optimization approaches.

**Gen93g: SA + local beam search**
- Keep top K configurations during search
- Cross-pollinate between them

**Gen93h: Genetic algorithm on configurations**
- Population of full packings
- Crossover by swapping tree subsets
- Mutation by SA on individuals

## Implementation Order

1. **Gen93a**: Randomized placement order (simple test)
2. **Gen93c**: Coarse-to-fine optimization
3. **Gen93e**: Aspect ratio penalty
4. **Gen93g**: SA + beam search

## Benchmark Protocol

For each candidate:
1. Build: `cargo build --release`
2. Test: `./target/release/benchmark 200 3`
3. If score < 87.29, run 5 more times to verify
4. If consistently better, update champion

## Key Insight

The gap to target (26.5%) suggests we're missing something fundamental. Top solutions likely use:
- Different tree representations
- Non-greedy global optimization
- Problem-specific geometric insights

We need to think differently, not just tune parameters.
