# Gen123 Results: CP-SAT and Interlocking Search

## Summary
- **Starting score**: 85.10 (Gen120)
- **Final score**: 85.10 (no improvement)
- **Goal**: ~69.02 (top leaderboard)
- **Gap**: Still 23.3%

## Approaches Tried

### 1. OR-Tools CP-SAT Solver

**Goal**: Use constraint programming for exact small-n solutions.

**Implementation**:
- Created Python 3.12 environment with OR-Tools
- Implemented CP-SAT solver with:
  - Integer position variables (scaled by precision)
  - 8 discrete rotation angles (45° steps)
  - Bounding box separation constraints
  - Minimize bounding square side

**Results**:
| n | Current Best | CP-SAT | Status |
|---|--------------|--------|--------|
| 1 | 0.8132 | 0.8000 | Invalid (precision artifact) |
| 2 | 0.9496 | 1.2900 | Worse |
| 3 | 1.1423 | inf | Overlaps |
| 4 | 1.2946 | inf | Overlaps |
| 5 | 1.4781 | inf | Overlaps |

**Why It Failed**:
- Bounding box constraints are too loose for tree shapes
- Valid CP-SAT solutions have actual polygon overlaps
- Would need exact NFP-based constraints (very complex to encode)

### 2. NFP-Based Packer

**Goal**: Use No-Fit Polygons for exact boundary placement.

**Results**:
- NFP packer for n=3: side=1.4070 (current best: 1.1423)
- Significantly worse than current solution

**Why It Failed**:
- Greedy NFP placement doesn't optimize globally
- Our Rust+SA+CMA-ES pipeline is more sophisticated

### 3. Tree Interlocking Search

**Goal**: Find configurations where trees interlock (tips in concave regions).

**Implementation**:
- Multi-resolution grid search for n=2
- Phase 1: Coarse 30° angles, 0.1 position step
- Phase 2: Medium refinement around top candidates
- Phase 3: Fine 2° angles, 0.005 position step
- Phase 4: Very fine 0.5° angles, 0.001 position step

**Results**:
- Best found: side=0.9706 for n=2
- Current best: side=0.9496
- **No improvement** (current is already optimal or near-optimal)

**Why It Failed**:
- Our current solution was already optimized by Rust+SA+CMA-ES
- Python search is slower and less thorough than compiled Rust
- Tree shape has limited interlocking potential

## Key Learnings

### 1. Current Solution is Near-Optimal for Greedy Approaches
The 85.10 score from Gen120 represents the best achievable with:
- Greedy placement + simulated annealing
- Multiple random restarts (best-of-N)
- CMA-ES continuous refinement

### 2. Paradigm Shift Required for 23% Improvement
The gap to 69.02 (top leaderboard) is too large for incremental optimization:
- 16 points = 19% improvement needed
- Cannot be achieved through parameter tuning alone
- Requires fundamentally different algorithm

### 3. Tree Shape Limits Interlocking
The Christmas tree polygon has concave regions but:
- Concavities are shallow (0.0625-0.1 depth)
- Tip width at y=0.5 is 0.25
- Limited nesting potential

### 4. CP-SAT Needs Exact Geometry
For CP-SAT to work correctly:
- Need exact NFP-based separation constraints
- Bounding box approximations cause invalid solutions
- Complexity explodes with exact constraints

## Files Created
- `python/cpsat_packer.py` - CP-SAT solver implementation
- `python/analyze_interlocking.py` - Interlocking analysis script
- `python/fast_n2_search.py` - Multi-resolution n=2 search
- `python/cpsat_results_1_5.json` - CP-SAT test results
- `ortools_env/` - Python 3.12 virtual environment with OR-Tools

## What Would Actually Help

Based on analysis, closing the 23% gap would require:

1. **Machine Learning**: Learned placement heuristics or GNN-based optimization
2. **Massive Compute**: Millions of restarts with cloud infrastructure
3. **Exact Methods**: ILP/MIP with exact polygon constraints (very slow)
4. **Domain Expertise**: Competition-specific tricks we haven't discovered

## Recommendation

The current approach (Rust+SA+CMA-ES) is performing well for a greedy/metaheuristic approach. Further improvements would require:
- Fundamental algorithmic changes (weeks of R&D)
- Significant compute resources (cloud)
- Or accepting the current score as competitive for this approach

## Score Breakdown (Current)

| n Range | Avg Score/n | Total Contribution |
|---------|-------------|-------------------|
| 1-10 | 0.456 | 4.56 |
| 11-50 | 0.475 | 19.00 |
| 51-100 | 0.421 | 21.05 |
| 101-150 | 0.410 | 20.50 |
| 151-200 | 0.399 | 19.95 |
| **Total** | | **85.10** |
