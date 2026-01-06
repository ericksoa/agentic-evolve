# Gen115 Plan: Multi-Target Optimization

## Current State
- **Score**: 85.50 (Gen114)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

## Priority 1: Fix n=7 with CMA-ES

The Gen114 CMA-ES found n=7: 1.894 → 1.795 (5.2% improvement) but it had overlaps.

**Approach**:
- Run CMA-ES with stricter overlap penalty (10x current)
- Try multiple random starting configurations
- Use repair step with tighter tolerance
- Validate strictly before accepting

**Expected gain**: ~0.02-0.03 points if we can achieve even 3% improvement

## Priority 2: CMA-ES on n=3-6

These haven't been optimized as aggressively. Current values:
- n=3: 1.142 (from Gen113)
- n=4: 1.295 (already good)
- n=5: 1.501 (from Gen112)
- n=6: 1.715 (room for improvement?)

**Approach**:
- Run CMA-ES with 10k evaluations per n
- Multiple restarts with different initial configurations
- Focus on n=6 which seems to have most slack

## Priority 3: Medium n Optimization (11-30)

This range is UNTOUCHED - only using Rust's Best-of-20 output.

**Approach**:
- Extract current solutions from submission_best.csv
- Run CMA-ES or multi-start SA on n=11-20 first
- These contribute ~20% of total score

**Expected gain**: If we can improve by 1% average, that's ~0.2 points

## Priority 4: Continuous Angles (Paradigm Shift)

Current algorithm uses discrete 45° angles. Top solutions use continuous.

**Approach**:
- Modify CMA-ES to optimize angles as continuous variables
- Start with small n (3-10) where we can validate quickly
- This could unlock significant improvements

## Score Impact Analysis

| Target | Current | Potential | Points Saved |
|--------|---------|-----------|--------------|
| n=7 | 1.894 | 1.80 | 0.025 |
| n=6 | 1.715 | 1.68 | 0.014 |
| n=11-20 | ~2.3-3.0 | -1% each | ~0.05 |
| n=21-30 | ~3.2-4.0 | -1% each | ~0.04 |

**Total potential**: 0.10-0.15 points (85.50 → 85.35-85.40)

## Files to Create/Modify

- `python/gen115_runner.py` - Multi-target CMA-ES optimizer
- `python/gen115_optimized.json` - Results
- `submission_gen115.csv` - New submission

## Validation Checklist

Before any submission:
1. Run `python3 python/validate_submission.py submission_best.csv`
2. Check for overlaps in improved n values
3. Verify score is actually better than 85.50
