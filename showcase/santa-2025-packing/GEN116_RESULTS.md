# Gen116 Results: Medium-N CMA-ES Optimization

## Summary
- **Starting score**: 85.45 (Gen115)
- **Final score**: 85.45 (no improvement)
- **Improvement**: 0 points

## Approach
Priority 1 from GEN116_PLAN.md: Run CMA-ES on medium n (11-30) with large search budget:
- 10 restarts per n value
- 20,000 evaluations per restart
- Sigma = 0.25 (large exploration radius)
- High overlap penalty (5000)

## Initial Results (Shapely-based validation)
First run with Shapely area-based overlap checking showed improvements:

| n | Old Side | New Side | Improvement |
|---|----------|----------|-------------|
| 11 | 2.3233 | 2.2182 | +4.52% |
| 12 | 2.4725 | 2.3134 | +6.43% |
| 13 | 2.5262 | 2.4718 | +2.15% |
| 14 | 2.6285 | 2.6016 | +1.02% |
| 15 | 2.6937 | 2.6605 | +1.23% |

**Total: +0.15 score points**

## Critical Discovery: Validation Bug
When merging these "improvements" into submission_best.csv, validation failed:
```
✗ Found 2 errors:
  n=11: Trees 1 and 5 overlap
  n=12: Trees 2 and 11 overlap
```

The Kaggle validator uses **segment intersection** checking, which catches edge crossings that Shapely's `intersection.area` misses. Zero-area edge crossings are invalid but pass Shapely.

## Corrected Results (Strict validation)
Re-ran with strict segment-intersection validation:

| n | Result |
|---|--------|
| 11 | No improvement |
| 12 | No improvement |
| 13 | No improvement |
| 14 | No improvement |
| 15 | No improvement |

With proper validation, CMA-ES couldn't find improvements over Rust Best-of-20.

## Key Learnings

1. **Validation mismatch is critical**: Shapely area-based checking is insufficient. Must use segment intersection + point-in-polygon (matching Kaggle's checker).

2. **Rust Best-of-20 is near-optimal for medium n**: The greedy + SA approach with 20 runs already finds good solutions that CMA-ES can't improve.

3. **False improvements are common**: Many CMA-ES solutions look better but have zero-area edge crossings. Always validate before accepting.

4. **Updated optimizer**: `gen116_medium_n.py` now uses hybrid validation:
   - Shapely fast filter (skip non-intersecting pairs)
   - Segment intersection for edge cases

## Code Changes
- `python/gen116_medium_n.py`: Added strict segment intersection validation
- `CLAUDE.md`: Added post-improvement workflow with validation checklist
- `README.md`: Fixed outdated "Gen103" references, documented two-layer approach

## Files Modified/Created
- `python/gen116_medium_n.py` - Medium-n optimizer with strict validation
- `CLAUDE.md` - Added workflow documentation
- `README.md` - Updated algorithm description
- `GEN116_RESULTS.md` - This file

## Next Steps (for Gen117)
Since medium n couldn't be improved with CMA-ES:
1. Try Priority 2: Pattern-based initialization (radial/hexagonal layouts)
2. Try Priority 3: Large n boundary optimization
3. Try Priority 4: CMA-ES + SA hybrid for different local optima exploration

The 24% gap to leaderboard leaders likely requires fundamentally different approaches (exact solvers, continuous optimization with tighter bounds).
