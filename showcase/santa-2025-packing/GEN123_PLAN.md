# Gen123 Plan: Constraint Programming & NFP Optimization

## Current State
- **Score**: 85.10 (Gen120)
- **Target**: ~69.02 (top leaderboard)
- **Gap**: 23.3% (16 points)

## What We've Learned

### Failed Approaches (Gen121-122)
| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Evolve SDK (novel algorithms) | No improvement | LLM-generated algorithms worse than evolved Rust |
| Best-of-100 local | 85.39 | Rust-only, no Python refinement |
| Best-of-100 cloud | 85.53 | 4x slower, same result |
| Crossover/mutation | Regressed | Mutations produced 0.15x-0.80x fitness |

### Key Insights
1. **Our Rust algorithm is near-optimal** for greedy + SA approaches
2. **23% gap requires paradigm shift**, not incremental optimization
3. **Top solutions likely use** constraint programming or learned heuristics
4. **Tree shape has concave regions** that might allow nesting

## Gen123 Strategy: Paradigm Shift

### Approach 1: OR-Tools CP-SAT for Small N

Use Google's OR-Tools constraint programming solver for exact solutions on small n values (n ≤ 20).

**Why it might work:**
- CP-SAT can find globally optimal solutions
- Small n values have significant score contribution (side²/n)
- Exact solutions for n=1..20 could improve by 5-10%

**Implementation:**
```python
from ortools.sat.python import cp_model

def solve_packing_cpsat(n: int, precision: int = 100):
    """
    Model tree packing as constraint satisfaction.

    Variables:
    - x[i], y[i]: Position of tree i (scaled to integers)
    - angle[i]: Rotation index (0-7 for 45° steps)
    - side: Bounding square side length

    Constraints:
    - No overlaps (using NFP or SAT encoding)
    - All trees inside [0, side] x [0, side]

    Objective:
    - Minimize side
    """
    model = cp_model.CpModel()
    # ... constraint encoding
```

**Expected improvement:** 2-5 points on small n values

### Approach 2: No-Fit Polygon (NFP) Optimization

Compute the No-Fit Polygon between tree shapes to enable tighter boundary placement.

**What is NFP:**
- NFP(A, B) = locus of reference point of B when B slides around A without overlap
- Enables finding exact boundary positions where trees touch
- Critical for achieving theoretical optimal packing density

**Implementation:**
```python
from shapely.geometry import Polygon
from shapely.ops import unary_union

def compute_nfp(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """Compute no-fit polygon using Minkowski sum."""
    # NFP = Minkowski sum of A and reflected B
    reflected_b = scale(poly_b, -1, -1, origin=(0, 0))
    return poly_a.buffer(0).union(reflected_b.buffer(0)).convex_hull
```

**Expected improvement:** 3-8 points from tighter packing

### Approach 3: Continuous Angle Optimization

Current solver uses 8 discrete angles (45° steps). Top solutions use continuous angles.

**Implementation:**
- Add angle as continuous variable in CMA-ES
- Use gradient-free optimization (scipy.optimize)
- Fine-tune angles after greedy placement

**Expected improvement:** 1-3 points from better angle choices

### Approach 4: Tree Shape Exploitation

The Christmas tree polygon has concave regions (the tiered branches). Other trees could potentially fit into these gaps.

**Analysis needed:**
1. Compute exact tree polygon geometry
2. Find concave pockets
3. Test if smaller trees can nest into pockets
4. Design placement heuristic for nesting

**Expected improvement:** Unknown, potentially significant

## Implementation Order

### Phase 1: Quick Wins (1-2 hours)
1. **Continuous angle optimization** - Modify existing CMA-ES to include angles
2. **Test on subset** - Run on n=1..50 to validate approach

### Phase 2: CP-SAT (4-8 hours)
1. **Install OR-Tools** - `pip install ortools`
2. **Encode small cases** - Start with n=1..5
3. **Validate solutions** - Ensure no overlaps
4. **Scale up** - Extend to n=1..20 if feasible

### Phase 3: NFP (4-8 hours)
1. **Compute tree NFP** - Pre-compute NFP between tree shapes
2. **Integrate with placement** - Use NFP for boundary detection
3. **Optimize placement order** - Try different tree orderings

## Execution Commands

### Phase 1: Continuous Angles
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Modify CMA-ES to optimize angles
python3 python/optimize_with_angles.py --n-range 1-50 --output submission_gen123_angles.csv

# Validate
python3 python/validate_submission.py submission_gen123_angles.csv
```

### Phase 2: CP-SAT
```bash
# Install OR-Tools
pip install ortools

# Run CP-SAT solver on small n
python3 python/cpsat_packer.py --n-range 1-20 --output submission_gen123_cpsat.csv

# Merge with existing best
python3 python/create_hybrid_submission.py \
  --base submission_best.csv \
  --overlay submission_gen123_cpsat.csv \
  --output submission_gen123_hybrid.csv
```

### Phase 3: NFP
```bash
# Compute NFPs
python3 python/compute_nfp.py --output nfp_cache.pkl

# Run NFP-aware placement
python3 python/nfp_packer.py --nfp-cache nfp_cache.pkl --output submission_gen123_nfp.csv
```

## Success Criteria

| Approach | Target Improvement | Success |
|----------|-------------------|---------|
| Continuous angles | 1-3 points | Score < 84 |
| CP-SAT (n≤20) | 2-5 points | Score < 83 |
| NFP optimization | 3-8 points | Score < 80 |
| Combined | 5-10 points | Score < 78 |

## Recovery Notes

If session restarts:
1. Read this file for context
2. Check which phase was in progress
3. Look for `submission_gen123_*.csv` files
4. Continue from last checkpoint

## Critical Constraints

**NEVER start more than 4 parallel processes locally.**
- CP-SAT solver is single-threaded by default
- Use `--workers 4` flag if parallelizing

## Risk Assessment

| Approach | Complexity | Risk | Potential |
|----------|------------|------|-----------|
| Continuous angles | Low | Low | Medium |
| CP-SAT | High | Medium | High |
| NFP | High | Medium | High |
| Tree nesting | Very High | High | Unknown |

## Recommendation

Start with **Phase 1 (continuous angles)** as it's low-risk and builds on existing infrastructure. If successful, proceed to **Phase 2 (CP-SAT)** for small n optimization.

## Progress Log
- [x] Phase 1: Continuous angle optimization (already done in Gen118, no new improvement)
- [x] Phase 2: CP-SAT solver for small n (failed - bounding box constraints too loose)
- [x] Phase 3: NFP optimization (tested - worse than current solution)
- [x] Tree interlocking search (no improvement found)
- [x] Document results

## Final Status: NO IMPROVEMENT
See GEN123_RESULTS.md for detailed analysis.
