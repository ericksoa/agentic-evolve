# Gen113 Results

## Summary

**Score improvement: 85.59 -> 85.56 (0.04% improvement, 0.036 score delta)**

## What Was Implemented

### 1. Exhaustive N=2 Search
- Discovered that two trees oriented 180 degrees apart can interlock efficiently
- Searched over angle pairs and positioning directions
- Found optimal configuration: trees at ~66 and ~246 degrees
- **Improvement**: side 0.988 -> 0.951 (3.8% reduction)
- **Score delta**: 0.036

### 2. Intensive N=6 and N=7 Optimization (Limited Success)
- Attempted SA optimization with strict overlap validation
- The strict validation (edge intersection + point-in-polygon) is computationally expensive
- Didn't complete due to time constraints
- Current n=6 and n=7 solutions remain unchanged

## Score Breakdown

| n | Before | After | Side Change | Score Delta |
|---|--------|-------|-------------|-------------|
| 2 | 0.9884 | 0.9510 | -3.8% | +0.0363 |
| 6 | 1.7152 | 1.7152 | 0% | 0 |
| 7 | 1.8940 | 1.8940 | 0% | 0 |
| **Total** | 85.59 | **85.56** | - | **+0.036** |

## Key Learnings

1. **180-degree patterns work well for n=2**: Trees pointing in opposite directions can interlock efficiently.

2. **Strict validation is critical**: Shapely's area-based overlap check (area > 1e-9) isn't strict enough. Need edge intersection + point-in-polygon checks.

3. **Optimization speed trade-off**: Strict validation makes SA much slower. For larger n, need either:
   - Faster overlap checking (spatial indexing)
   - More iterations/time budget
   - Use Rust for validation

## Files Created

- `python/n2_exhaustive.py` - Exhaustive n=2 search
- `python/n2_fast.py` - Fast n=2 search with binary search
- `python/n6_intensive.py` - Intensive n=6 optimizer
- `python/gen113_quick.py` - Quick SA optimizer with strict validation
- `python/gen113_n2.json` - N=2 optimized solution
- `submission_gen113.csv` - New best submission

## Score Progression

| Generation | Score | Delta from Prev |
|------------|-------|-----------------|
| Gen109 | 86.17 | - |
| Gen110 | 85.76 | -0.41 |
| Gen111 | 85.67 | -0.09 |
| Gen112 | 85.59 | -0.08 |
| **Gen113** | **85.56** | **-0.03** |

## Next Steps (Gen114+)

1. **Rust-based strict validation**: Port the strict overlap checker to Rust for faster SA
2. **N=6,7 optimization with more time**: Current solutions likely suboptimal
3. **Medium n focus (10-30)**: These have higher s^2/n ratios
4. **Global optimization methods**: Try CMA-ES or genetic algorithms for small n
