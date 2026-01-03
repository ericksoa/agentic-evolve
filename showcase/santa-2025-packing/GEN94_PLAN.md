# Gen94 Plan: Radical Paradigm Shifts

## Current State
- **Champion**: Gen91b (rotation-first optimization)
- **Score**: ~87.29 (with variance ±1.5)
- **Target**: ~69 (leaderboard top)
- **Gap**: 26.5%

## What's Been Exhaustively Tried

### Gen92 (Parameter Tuning) - All Failed
- More rotations, finer precision, more iterations, etc.

### Gen93 (Algorithmic Changes) - All Failed
- Relocate moves, coarse-to-fine, force-directed, aspect ratio, etc.

## Gen94 Strategy: Radical Departures

Since incremental changes failed, try paradigm shifts:

### Gen94a: Multi-Start Best-of-N
**Rationale**: Given high variance (~87-90), run many independent optimizations and take the best.

- Run the current algorithm 10 times with different seeds
- Keep the best result for each n
- Trade compute time for quality

### Gen94b: Polar Coordinate Representation
**Rationale**: Trees placed at (r, θ) from center might pack differently.

- Convert placement search to polar coordinates
- Search radius r and angle θ instead of Cartesian (x, y)
- May find different local optima

### Gen94c: Hexagonal Grid Seeding
**Rationale**: Hexagonal packing is optimal for circles; try seeding from hex grid.

- Initialize trees on hexagonal lattice positions
- Then optimize with SA
- May find more symmetric packings

### Gen94d: Strip Packing Approach
**Rationale**: Place trees in horizontal strips, then compress.

- Group trees by y-coordinate bands
- Pack each strip tightly
- Then compress strips vertically

### Gen94e: Genetic Algorithm on Full Configurations
**Rationale**: Population-based search may escape local optima.

- Population of 20 complete packings
- Crossover: swap tree subsets between packings
- Mutation: SA on individual packings
- Selection: keep best K

### Gen94f: Constraint-Based Placement
**Rationale**: Use constraints to ensure tight packing.

- For each tree, compute minimum gap to neighbors
- Place trees to minimize maximum gap
- Different optimization objective

## Implementation Order

1. **Gen94a**: Multi-start (simplest, may help immediately)
2. **Gen94c**: Hexagonal seeding (geometric insight)
3. **Gen94e**: Genetic algorithm (population-based)
4. **Gen94b**: Polar coordinates (representation change)

## Benchmark Protocol

For each candidate:
1. Build: `cargo build --release`
2. Test: `./target/release/benchmark 200 3`
3. If score < 87.29, verify with 5 runs
4. Update EVOLUTION_STATE.md with results

## Key Insight

The 26.5% gap suggests we're fundamentally approaching the problem wrong. Top solutions likely use:
- Global optimization (not greedy incremental)
- Different representations
- Problem-specific geometric insights (all trees identical)
