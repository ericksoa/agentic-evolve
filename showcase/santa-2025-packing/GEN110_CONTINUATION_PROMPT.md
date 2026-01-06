# Gen110 Continuation Prompt

Copy everything below this line after clearing context:

---

Continue working on the Santa 2025 packing competition. Read GEN110_PLAN.md for the full plan.

## Quick Context

**Competition**: Pack n Christmas tree polygons (15-vertex shape) into smallest square (n=1 to 200)
**Score formula**: Sum of (side_length² / n) for all n. Lower is better.
**Current best**: 86.13 (Gen103, recovered from git commit 2457880)
**Leader (#1)**: ~69
**Gap**: 25% - significant room for improvement

## Key Finding from Gen109

The Gen109 SA improvements (combined_move, squeeze_interval) actually **hurt** performance. Reverted to champion config:
- `sa_iterations: 28000`
- `combined_move_prob: 0.0` (disabled)
- `squeeze_interval: 999999` (disabled)

## Gen110 Strategy: Frontier Optimization

Implement in priority order:

### Priority 1: NFP (No-Fit Polygon) Based Placement
The mathematical foundation used by all serious packing solvers.
- Compute Minkowski sum for tree polygon pairs
- Precompute NFP for all 8x8 rotation pairs
- Use NFP to find ALL valid placements, not just sampled ones
- Select placement minimizing side length increase

### Priority 2: CMA-ES Global Optimization
State-of-the-art continuous optimizer.
```bash
pip install cma
```
- Initialize from greedy solution
- Optimize 3n parameters (x, y, θ per tree)
- Use soft overlap penalty
- Snap to valid with local search

### Priority 3: ILP for Small n (n ≤ 15)
Provably optimal solutions via constraint programming.
```bash
pip install ortools
```
- Discretize positions (100x100 grid) and angles (8 steps)
- Precompute NFP lookup tables
- Formulate CP-SAT model
- 5-minute timeout per n

### Priority 4: Integration
Combine all approaches, select best for each n.

## Key Files

```
rust/src/evolved.rs           # Main Rust algorithm (reverted to champion)
rust/src/bin/ultimate_submission.rs  # Best submission generator
python/ilp_solver.py          # ILP solver (exists, needs NFP)
python/gen109_runner.py       # Runner script

submission_gen109.csv         # Current best submission (86.13)
GEN110_PLAN.md               # Full plan with code examples
```

## Commands

```bash
# Build Rust
cd showcase/santa-2025-packing/rust
cargo build --release

# Run Rust submission generator
./target/release/ultimate_submission 5 5 submission_new.csv

# Score a submission
python3 -c "
import csv, math
# ... scoring code in GEN110_PLAN.md
"

# Install Python deps for Gen110
pip install cma shapely ortools torch
```

## Tree Polygon Vertices (for reference)
```python
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
```

## Success Metrics

- [ ] NFP placement beats Rust greedy for >50% of n values
- [ ] CMA-ES improves at least 10% of n values
- [ ] ILP finds optimal for n=1..10
- [ ] Combined score < 80 (7.5% improvement)
- [ ] Stretch: score < 75 (13% improvement)

## Start Here

1. Read GEN110_PLAN.md for detailed code examples
2. Start with **Priority 1: NFP** - implement `compute_nfp()` using Shapely
3. Test NFP placement on n=10, compare to Rust greedy
4. If NFP works, move to CMA-ES integration

The 25% gap to #1 suggests there's a fundamentally better approach we haven't found yet. NFP + global optimization is how industrial packing solvers work.
