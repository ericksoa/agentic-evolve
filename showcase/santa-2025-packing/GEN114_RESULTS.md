# Gen114 Results

## Summary

**Score improvement: 85.56 → 85.50 (0.07% improvement, 0.053 score delta)**

## What Was Implemented

### CMA-ES Optimization

Used CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for global optimization of tree positions for n=3-10.

- Library: `cma` (Python)
- Parameters: 5000 evaluations per n, sigma=0.2
- Starting point: Gen113 submission values (not outdated optimized_small_n.json)

### Key Learnings

1. **Baseline matters**: The `optimized_small_n.json` file had outdated values. Gen113 submission had better solutions for some n values (e.g., n=4 was 1.2946 in submission but 1.4931 in json). Always use the actual submission as baseline.

2. **Validation is critical**: CMA-ES found a good-looking n=7 solution (1.894 → 1.795, 5.2%) but it had overlaps! Must validate strictly before using.

3. **CMA-ES escapes local optima**: Successfully improved n=8 (3.3%) and n=9 (1.9%) where simple SA couldn't.

## Score Breakdown

| n | Gen113 | Gen114 | Change | Valid |
|---|--------|--------|--------|-------|
| 7 | 1.8940 | 1.7953 | -5.2% | **NO** (overlap) |
| 8 | 2.0378 | 1.9701 | -3.3% | YES |
| 9 | 2.1293 | 2.0881 | -1.9% | YES |

Only n=8 and n=9 improvements were used in final submission.

## Score Progression

| Generation | Score | Delta from Prev |
|------------|-------|-----------------|
| Gen109 | 86.17 | - |
| Gen110 | 85.76 | -0.41 |
| Gen111 | 85.67 | -0.09 |
| Gen112 | 85.59 | -0.08 |
| Gen113 | 85.56 | -0.03 |
| **Gen114** | **85.50** | **-0.05** |

## Files Created/Modified

- `python/gen114_runner.py` - CMA-ES optimization runner
- `python/gen114_optimized.json` - Optimization results
- `python/optimized_small_n.json` - Updated with Gen113 baseline + Gen114 improvements
- `submission_gen114.csv` - New submission
- `GEN114_PLAN.md` - Planning document

## Technical Notes

### CMA-ES vs Simulated Annealing

| Aspect | CMA-ES | SA |
|--------|--------|-----|
| Global vs Local | Better at escaping local optima | Can get stuck |
| Speed | Slower per iteration | Faster per iteration |
| Parameter tuning | Adaptive (self-tuning) | Requires temperature schedule |
| Parallelizable | Yes (population-based) | No (sequential) |

### Validation Issue

The CMA-ES optimizer uses a penalty function for overlaps but doesn't guarantee valid solutions. The repair step helps but isn't foolproof. For n=7:

- CMA-ES found: side=1.7953
- Repair step accepted it
- Strict validation: **FAILED** (area overlap > 1e-9)

Solution: Always run strict validation before accepting any solution.

## Next Steps (Gen115+)

1. **Fix n=7 optimization**: Run with stricter overlap penalty or different starting configuration
2. **Try n=3-6**: May have room for improvement with different strategies
3. **Pattern search**: Try systematic patterns (radial, 180° pairs) for small n
4. **Medium n optimization**: n=11-30 may have significant room for improvement
5. **Rust-based optimizer**: Port to Rust for faster evaluation
