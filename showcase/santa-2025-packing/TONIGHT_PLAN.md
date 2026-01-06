# Tonight's 3-Generation Plan (Gen121-123)

## Current State
- Score: 85.10 (Gen120)
- Target: ~69 (top leaderboard)
- Gap: 23.3% (16 points)

## The Gap Analysis

| Metric | Us | Top | Efficiency |
|--------|-----|-----|------------|
| Score | 85.10 | 69.02 | - |
| Avg per n | 0.426 | 0.345 | - |
| Packing efficiency | ~57% | ~70% | 13% gap |

Top solutions are 70% efficient. We're 57% efficient. Need ~13% improvement in space utilization.

---

## Gen121: Massive Parallelism (Best-of-100)

**Hypothesis**: The Rust solver uses Best-of-20. What if we use Best-of-100?

**Time estimate**: 30-45 minutes

### Implementation
```bash
# Modify Rust to run 100 trials per n instead of 20
cd rust
# Edit src/evolved.rs: change BEST_OF_N from 20 to 100
cargo build --release
./target/release/submit > submission_best100.csv
```

### Expected Improvement
- Gen103 showed Best-of-5→20 gave +3.87%
- Diminishing returns, but maybe +0.5-1% more?
- Even 0.5% = ~0.4 points

### Quick Commands
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Run with more trials
cd rust && cargo build --release
./target/release/ultimate_submission 100 1 ../submission_best100.csv

# Validate
python3 python/validate_submission.py submission_best100.csv

# Compare
python3 python/analyze_submission.py submission_best100.csv
```

---

## Gen122: OR-Tools Exact Optimization (n≤15)

**Hypothesis**: Small n values contribute ~8% of total score. Exact optimization might find global optima.

**Time estimate**: 45-60 minutes

### Implementation
1. Install OR-Tools: `pip install ortools`
2. Formulate as constraint satisfaction:
   - Variables: (x, y, angle) per tree
   - Constraints: No overlaps (discretized polygon)
   - Objective: Minimize bounding box

### Approach
```python
from ortools.sat.python import cp_model

def optimize_n_exact(n, max_time=60):
    model = cp_model.CpModel()

    # Discretize to integers (multiply by 1000)
    scale = 1000

    # Variables
    x = [model.NewIntVar(-3*scale, 3*scale, f'x{i}') for i in range(n)]
    y = [model.NewIntVar(-3*scale, 3*scale, f'y{i}') for i in range(n)]
    angle = [model.NewIntVar(0, 359, f'a{i}') for i in range(n)]  # degrees

    # Bounding box
    side = model.NewIntVar(0, 10*scale, 'side')

    # Constraints: trees don't overlap
    # (Use no-overlap constraint with polygon approximation)

    # Objective: minimize side
    model.Minimize(side)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)
```

### Expected Improvement
- If we can find global optima for n=2-15
- Current n=2-15 contribute ~8% of score
- Could improve these by 5-10%? = ~0.4-0.8 points

---

## Gen123: Hybrid: Rust 100x + OR-Tools + Deep Refinement

**Hypothesis**: Combine all improvements, then do multi-pass refinement.

**Time estimate**: 45-60 minutes

### Implementation
1. Start with Best-of-100 result from Gen121
2. Apply OR-Tools improvements from Gen122 for n≤15
3. Run deep SA refinement (100k iterations) on all n
4. Final position+angle refinement pass

### Pipeline
```bash
# Step 1: Get best-100 baseline
cp submission_best100.csv submission_hybrid.csv

# Step 2: Apply OR-Tools improvements
python3 python/gen122_ortools.py --input submission_hybrid.csv --output submission_hybrid.csv

# Step 3: Deep SA refinement
python3 python/gen120_full_sa.py --input submission_hybrid.csv --output submission_hybrid.csv \
  --n-start 2 --n-end 50 --restarts 5 --iters 100000

# Step 4: Final position+angle pass
python3 python/gen119_position_refine.py --input submission_hybrid.csv --output submission_hybrid.csv

# Validate
python3 python/validate_submission.py submission_hybrid.csv
```

### Expected Improvement
- Combination of all techniques
- Target: 84.5 or lower (0.6+ points from 85.10)

---

## Quick Start Commands

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Current score
python3 python/analyze_submission.py submission_best.csv | tail -5

# Run visualization
./rust/target/release/visualize submission_best.csv

# Validate before submit
python3 python/validate_submission.py submission_best.csv

# Submit
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions submit \
  -c santa-2025 -f submission_best.csv -m "GenXXX: description"
```

---

## Summary

| Gen | Approach | Time | Expected Gain |
|-----|----------|------|---------------|
| 121 | Best-of-100 | 30-45 min | 0.3-0.5 points |
| 122 | OR-Tools exact | 45-60 min | 0.3-0.8 points |
| 123 | Hybrid + deep refinement | 45-60 min | 0.2-0.5 points |

**Total expected**: 85.10 → ~84.0-84.5 (0.6-1.0 points improvement)

**Note**: The 23% gap (16 points) likely requires a fundamentally different algorithm paradigm that top solutions have discovered. These incremental improvements won't close the gap but will move us in the right direction.
