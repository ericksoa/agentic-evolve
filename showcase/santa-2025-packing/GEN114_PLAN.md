# Gen114 Plan

## Current State
- **Score**: 85.56 (Gen113)
- **Leader**: ~69
- **Gap**: 24%

## Analysis

Looking at score contributions:
- n=1-10: highest individual contributions (0.43-0.66 per group)
- n=11-30: medium contributions (~0.5-0.6 per group)
- Cumulative: 200 groups means even small per-group improvements add up

Key observations from Gen113:
1. n=2 improved 3.8% using 180° opposed pattern
2. n=6,7 couldn't be optimized due to slow validation
3. Strict validation (edge intersection) is critical but slow in Python

## Gen114 Strategy

### 1. Fix Data Consistency
- Update `optimized_small_n.json` with Gen113 n=2 improvement
- Ensure submission_best.csv uses all best known solutions

### 2. CMA-ES Optimization for Small N
Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for n=3-10:
- Better at escaping local minima than SA
- Handles correlated parameters well (x, y, angle are often correlated)
- Python library: `cma` (pip install cma)

### 3. Pattern Discovery
For each small n, try systematic patterns:
- **Radial**: trees pointing outward from center
- **180° pairs**: opposing trees that interlock (worked for n=2)
- **Spiral**: trees arranged in spiral with progressive angles
- **Grid-like**: regular spacing with optimal angles

### 4. Rust Validation (if time permits)
Port strict overlap checking to Rust for 10-100x speedup

## Implementation Order

1. [x] Create GEN114_PLAN.md
2. [ ] Update optimized_small_n.json with gen113 improvements
3. [ ] Install cma package: `pip install cma`
4. [ ] Create CMA-ES optimizer for n=3-10
5. [ ] Run optimization with 2-5 min per n value
6. [ ] Validate results with strict checker
7. [ ] Create new submission and validate
8. [ ] Document results in GEN114_RESULTS.md

## Expected Improvements

| n | Current Side | Target Side | Potential Score Gain |
|---|-------------|-------------|---------------------|
| 3 | 1.1441 | 1.05 | 0.07 |
| 4 | 1.4931 | 1.35 | 0.10 |
| 5 | 1.5992 | 1.45 | 0.09 |
| 6 | 1.7152 | 1.55 | 0.10 |
| 7 | 1.8641 | 1.70 | 0.08 |
| Total | - | - | ~0.44 |

Conservative target: 0.2-0.3 score improvement → 85.56 → 85.3
