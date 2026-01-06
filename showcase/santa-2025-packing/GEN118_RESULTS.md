# Gen118 Results

## Summary
- **Starting score**: 85.45
- **Final score**: 85.41
- **Improvement**: 0.04 points (0.05%)

## What Worked

### Post-SA Continuous Angle Refinement
Created `python/gen118_continuous_refine.py` that:
1. Loads existing submission (with discrete 45° angles from Rust SA)
2. For each tree, searches for optimal angle within ±10° of current
3. Uses coarse (2°) + fine (0.5°) search steps
4. Validates with strict segment-intersection checking (matches Kaggle)

**Key Innovation**: Previous attempts at continuous angles failed because they were applied **during** SA, hurting convergence. Gen118 applies continuous refinement **after** SA is complete, preserving the discrete structure that SA optimizes well.

**Results by n range**:
- n=11-50: 26 groups improved, 0.0168 points
- n=51-200: 110 groups improved, 0.0223 points
- Total: 136 groups improved, 0.039 points

### Strict Validation
The Shapely area-based overlap check (`intersection.area > 1e-10`) missed edge crossings, causing invalid submissions. Gen118 uses:
1. Shapely STRtree for fast spatial filtering
2. Segment-intersection checking for final validation (same as Kaggle)

This caught 3 invalid results that would have been rejected on submission.

## What Didn't Work

### Large Angle Ranges
Testing ±20° or larger ranges didn't find additional improvements - trees are tightly packed and can only move small amounts.

### Multiple Passes
Running 2+ passes on each group found diminishing returns. Most improvement comes from first pass.

## Gap Analysis

Score breakdown (current 85.41 vs target ~69):
- Small n (1-30): ~16.5% of score - mostly optimized by CMA-ES
- Large n (51-200): ~73% of score - limited by placement algorithm

The 24% gap to leaderboard likely requires:
1. **Different algorithm paradigm**: ILP/SAT solvers, or fundamentally different placement
2. **Different problem interpretation**: Perhaps allowing different overlap tolerance
3. **Better initial placement**: The greedy placement quality limits SA refinement

## Files Created
- `python/gen118_continuous_refine.py` - Post-SA continuous angle optimizer
- `submission_gen118.csv`, `submission_gen118_full.csv` - Intermediate results

## Next Steps
1. **Analyze top solutions** if publicly available
2. **Small n exact optimization** - For n=2-5, try exhaustive/analytical approaches
3. **ILP formulation** - May find global optima for small n
