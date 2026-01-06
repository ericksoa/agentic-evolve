# Gen117 Results

## Summary
- Starting score: 85.45
- Final score: 85.45 (no change)
- Improvement: 0.00 points

## What Was Tried

### 1. Pattern-Based CMA-ES Initialization
Created `python/gen117_patterns.py` and `python/gen117_fast.py` with:
- Radial pattern (concentric circles)
- Hexagonal close-packing pattern
- Spiral (Fibonacci/golden ratio) pattern
- Grid with 90-degree rotations
- Diagonal patterns

**Result**: Patterns produced worse solutions (5.x) than current (3.x) for n=20-25. CMA-ES from patterns found no valid improvements.

### 2. Local Refinement with CMA-ES
Created `python/gen117_local.py` with:
- Very small sigma (0.02-0.1) to stay near current optimum
- Multiple restarts with different sigma values
- High penalty weight (500) to avoid overlaps

**Result**: No improvements found for n=20-25. The current solutions are already at local optima.

### 3. Boundary-Focused Simulated Annealing
Created `python/gen117_boundary.py` with:
- Identifies trees that define the bounding box boundary
- Only moves boundary trees (10-22 trees depending on n)
- SA with 30,000-50,000 iterations per group

**Result**: No improvements found for n=50-100. Boundary trees couldn't be moved inward without overlaps.

## Why the Rust Solver is Near-Optimal

Analyzed `rust/src/evolved.rs` (Gen91b) and found an extremely sophisticated algorithm:

### Greedy Placement
- 200 search attempts per new tree
- All 8 rotations tested at each position
- Binary search for closest valid placement
- Sophisticated scoring: side length + balance + gap penalty + density bonus

### 6 Parallel Strategies
- ClockwiseSpiral, CounterclockwiseSpiral, Grid, Random, BoundaryFirst, ConcentricRings
- All run in parallel, best selected

### SA Local Search (28,000 iterations)
- 85% probability on boundary trees
- Edge-aware moves (push trees toward center based on edge)
- Compression, fill, rotation moves
- Elite pool with hot restarts
- Multiple passes with different temperatures

### Post-Processing
- 5-pass wave compaction (outside-in then inside-out)
- Greedy backtracking on boundary-defining trees

### Best-of-N Selection
- Run entire algorithm 20 times
- Pick best solution for each n

## Key Learnings

1. **CMA-ES is ineffective** for this problem at medium-to-large n because:
   - The parameter space (3n dimensions) is too large
   - Valid configurations are sparse (most have overlaps)
   - The Rust solver's greedy + SA already finds near-optimal solutions

2. **Pattern initialization doesn't help** because:
   - The Rust solver already uses 6 different initialization strategies
   - Patterns that look optimal in theory often have worse packing in practice
   - The tree shape (asymmetric with protrusions) doesn't pack like circles

3. **Boundary optimization is limited** because:
   - Boundary trees are already tightly packed against neighbors
   - Moving them inward requires moving interior trees first
   - The wave compaction in Rust already does this extensively

4. **The Rust solver is highly evolved** (literally through /evolve):
   - Gen91b came from many generations of algorithmic evolution
   - Every parameter and move operator has been tuned
   - Additional improvements require fundamental algorithmic changes

## Recommendations for Future Generations

1. **Algorithm-level changes**: Look for entirely new algorithmic approaches rather than parameter tuning

2. **Machine learning**: Train a model to predict good initial configurations

3. **Exact solvers**: For small n (2-10), consider exact optimization methods

4. **Focus on verification**: The gap to leaderboard (85.45 vs ~69) suggests competitors may have:
   - Different problem interpretations
   - Unreported constraint relaxations
   - Fundamentally different algorithms

## Files Created
- `python/gen117_patterns.py` - Pattern-based CMA-ES (comprehensive, slow)
- `python/gen117_fast.py` - Optimized pattern CMA-ES with Shapely fast check
- `python/gen117_local.py` - Local refinement with tight sigma
- `python/gen117_boundary.py` - Boundary-focused SA
