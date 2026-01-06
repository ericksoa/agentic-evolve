# Gen111 Results

## Summary

**Score improvement: 85.77 → 85.67 (0.12% improvement, 0.10 score delta)**

## What Worked

### 1. Multi-start Simulated Annealing for n=4 (Big Win!)
- **Method**: Random initialization + SA with 80,000-100,000 iterations
- **Result**: 1.452 → 1.327 (improvement of 0.125, score delta +0.082)
- **Why it worked**: Random starts explored more of the solution space than refining the Rust solution

### 2. Refinement SA for n=5, n=7, n=8
- Started from existing Rust solutions and refined with SA
- n=5: 1.599 → 1.593 (score delta +0.004)
- n=7: 1.895 → 1.894 (score delta +0.001)
- n=8: 2.054 → 2.038 (score delta +0.008)

## What Didn't Work

### 1. Extended optimizer for n=11-20
- Rust solutions were already better for these n values
- Grid search + refinement couldn't beat SA-optimized Rust solutions

### 2. Random-start SA for n=7,8,9,10
- Random initialization produced very poor initial packings
- SA couldn't compress them enough to beat Rust solutions
- Need better initialization strategy for larger n

### 3. n=6 refinement
- Found an improvement but it had overlaps (failed validation)
- Need stricter tolerance during optimization

## Key Learnings

1. **n=4 is special**: Multi-start random SA found much better solutions than refining existing ones
2. **Refinement works for small improvements**: Starting from good solutions can squeeze out 0.5-1% improvements
3. **Validation is critical**: Always run overlap check before submission
4. **Rust SA is strong**: For n>10, the Rust solver's SA is hard to beat with Python

## Score Breakdown Comparison

| n | Before | After | Δ Score |
|---|--------|-------|---------|
| 4 | 1.452 | 1.327 | +0.082 |
| 5 | 1.599 | 1.593 | +0.004 |
| 7 | 1.895 | 1.894 | +0.001 |
| 8 | 2.054 | 2.038 | +0.008 |
| **Total** | 85.77 | 85.67 | +0.10 |

## Files Created

- `python/sa_optimizer.py` - Multi-start SA optimizer
- `python/optimize_small_n_extended.py` - Extended small n optimizer
- `python/sa_n4_best.json`, `python/sa_n4_best2.json` - Best n=4 solutions
- `python/refined_quick.json` - Refinement results
- `submission_gen111.csv`, `submission_gen111_v2.csv`, `submission_gen111_v3.csv` - Submission versions

## Next Steps (Gen112)

1. **Pattern learning**: Analyze the n=4 solution to understand what makes it good
2. **Better SA initialization**: Use known good patterns instead of random
3. **Longer refinement**: Run refinement for much longer on promising solutions
4. **Fix n=6**: Re-run with stricter overlap tolerance
5. **Target medium n (15-30)**: These still have room for improvement
