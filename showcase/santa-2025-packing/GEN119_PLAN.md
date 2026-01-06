# Gen119 Plan

## Current State
- **Score**: 85.41 (Gen118)
- **Target**: ~69 (24% gap)
- **Gap**: 16+ points needed

## Analysis of Score Distribution

From Gen118 analysis:
- n=1-10: ~5.6% of score (already optimized by CMA-ES)
- n=11-30: ~10.9% of score (modest improvements from continuous angles)
- n=31-200: ~83.5% of score (limited by placement algorithm)

The large n values dominate the score but are hardest to improve.

## Gen119 Focus: Combined Position + Angle Refinement

Gen118 only refined **angles**. Gen119 will refine **both position and angle** together.

### Approach: Local Search with Position Perturbation

For each tree, search a small neighborhood:
1. Position: ±0.05 in x and y (grid search with 0.01 steps)
2. Angle: ±10° (0.5° steps, from Gen118)
3. Accept if: no overlaps AND smaller bounding box

This is O(11 × 11 × 41) = 4,961 evaluations per tree, but with early termination.

### Optimization Strategy

1. **Boundary trees first**: Trees that define the bounding box edge have most impact
2. **Greedy acceptance**: Accept any improvement immediately
3. **Multiple passes**: Repeat until no improvement found
4. **Strict validation**: Segment-intersection check (matches Kaggle)

### Expected Improvements

- Position refinement can "slide" trees closer together
- Combined with angle can find tighter interlocking configurations
- Most benefit for medium n (20-100) where trees aren't perfectly packed

## Alternative Approaches (if position refinement fails)

### A. Small n Exact Solutions (n=2-5)
- n=2: Only 2 trees, can exhaustively search angle combinations
- n=3-5: Grid search over positions + angles
- These contribute ~2% of score but are most improvable

### B. Genetic Algorithm on Configurations
- Population of packings for each n
- Crossover: Exchange tree positions between parents
- Mutation: Small position/angle perturbations
- Selection: Keep best valid configurations

### C. Different Overlap Tolerance
- Current: Any segment intersection = invalid
- Test: What if Kaggle allows touching edges?
- This could explain the 24% gap

## Implementation Plan

1. Create `python/gen119_position_refine.py`
2. Implement combined position + angle search
3. Add boundary tree prioritization
4. Test on n=20-50 first (moderate complexity)
5. If successful, run on full range
6. Validate with strict checker before saving

## Success Criteria

- Any improvement over 85.41 is a win
- Target: 0.1+ point improvement
- Stretch: 0.5+ point improvement
