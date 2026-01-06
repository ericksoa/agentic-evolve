# Gen119 Results

## Summary
- **Starting score**: 85.41
- **Final score**: 85.17
- **Improvement**: 0.24 points (0.29%)

## What Worked

### Combined Position + Angle Refinement
Created `python/gen119_position_refine.py` that extends Gen118's angle-only refinement to also perturb tree positions:

**Search space per tree:**
- Position: ±0.05 in x and y (coarse 0.02 steps → fine 0.005 steps)
- Angle: ±10° (coarse 2° → fine 0.5°)
- Total: ~5000 candidates per tree in fine mode

**Key innovations:**
1. **Boundary trees first**: Trees on the bounding box edge have most impact
2. **Coarse-to-fine search**: Fast initial scan, then zoom into promising regions
3. **Fast mode for large n**: Coarser grid (±0.03, 0.015 steps) for n>100
4. **Strict segment-intersection validation**: Catches edge cases Shapely misses

**Results by n range:**
- n=2-100: 52 groups improved, 0.1870 points
- n=101-200: 62 groups improved, 0.0572 points
- **Total: 114 groups improved, 0.2442 points**

### Performance Optimization
Large n values were very slow (30+ min per group). Solutions:
1. Coarser search grid for n>100
2. Chunked processing (n=2-100, then n=101-200)
3. Strict validation only on candidate moves (not full grid)

## What Didn't Work

### Shapely-only Validation
Initial fast mode used only Shapely's `intersection.area > 1e-12` check, which:
- Missed zero-area edge crossings
- Caused invalid submissions for n=131

**Fix**: Added strict segment-intersection check after each improvement candidate.

### Longer Search Ranges
Testing ±0.1 position range didn't improve results:
- Trees are tightly packed
- Larger moves cause overlaps
- Diminishing returns beyond ±0.05

## Gap Analysis

Score breakdown (current 85.17 vs target ~69):
- Total improvement from Gen118 + Gen119: ~0.28 points
- Remaining gap: ~16 points (19%)

The position + angle refinement helps but doesn't address the fundamental gap. The greedy initial placement algorithm limits how much local refinement can achieve.

## Files Created
- `python/gen119_position_refine.py` - Combined position + angle optimizer
- `submission_gen119_part1.csv` - n=2-100 results
- `submission_gen119_part2.csv` - Full n=2-200 results

## Next Steps
1. **Better initial placement**: Algorithmic improvements to greedy placement
2. **ILP/SAT solvers**: Exact optimization for small n
3. **Analyze leaderboard solutions**: What fundamentally different approach achieves ~69?
