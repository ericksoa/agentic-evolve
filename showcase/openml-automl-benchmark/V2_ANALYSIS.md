# v2 Improvements Analysis

## What Worked

### 1. Skip Strategy (Low Overlap <20%)
**Success!** On confident datasets, v2 correctly skips optimization:

| Dataset | Overlap | v1 Gain | v2 Gain | Result |
|---------|---------|---------|---------|--------|
| breast-w | 2.0% | +0.4% | +0.0% | Correct skip |
| banknote-auth | 3.2% | +0.0% | +0.0% | Correct skip |
| spambase | 11.0% | -0.4% | +0.0% | **Fixed v1 harm** |
| qsar-biodeg | 19.4% | +0.0% | +0.2% | Correct skip |

**Impact**: v2 avoids the -0.4% harm that v1 caused on spambase. Saves compute on 4 datasets.

### 2. credit-g Detection
v2 correctly identified credit-g as "aggressive" (85.5% overlap) and used wide threshold range.
- Optimal threshold found: **0.05** (way below default 0.5)
- Gain: +18.3%

## What Didn't Work

### 1. Aggressive Strategy Sometimes Hurts
High overlap doesn't always mean threshold optimization helps:

| Dataset | Overlap | Optimal Thresh | v2 Gain | Problem |
|---------|---------|----------------|---------|---------|
| credit-g | 85.5% | 0.05 | +18.3% | Works! |
| blood-transfusion | 64.6% | 0.54 | -2.2% | Optimal near 0.5 |
| ilpd | 60.0% | 0.48 | -0.6% | Optimal near 0.5 |

**Insight**: High overlap + optimal threshold near 0.5 = no benefit from optimization.

### 2. Calibration Hurts
The `calibrate=True` option made things worse on most datasets:
- blood-transfusion: -11.3%
- ilpd: -28.6%
- pc1: -55.3%

**Conclusion**: Remove calibration as a default option.

## Key Discovery

**Overlap alone doesn't predict success.** The real pattern is:

```
Threshold optimization helps when:
  1. High overlap (>40%) - model is uncertain
  AND
  2. Optimal threshold is FAR from 0.5 - shifting threshold matters
```

credit-g has both: 85.5% overlap + optimal at 0.05
blood-transfusion has only #1: 64.6% overlap but optimal at 0.54

## Proposed v2.1 Improvement

Add a "threshold sensitivity" check during uncertainty analysis:
1. Compute F1 at multiple thresholds on CV data
2. If F1 is relatively flat (low variance), skip optimization
3. If F1 has a clear peak far from 0.5, use aggressive search

```python
# Pseudocode for improved detection
def should_optimize(probs, true_labels):
    f1_scores = [compute_f1_at_threshold(t) for t in [0.3, 0.4, 0.5, 0.6]]
    variance = np.var(f1_scores)

    if variance < 0.001:  # Flat curve
        return False, 0.5

    # Find if peak is far from 0.5
    best_t = find_best_threshold(...)
    if abs(best_t - 0.5) > 0.15:
        return True, 'aggressive'
    else:
        return True, 'normal'
```

## Summary

| Strategy | Datasets | v1 Avg | v2 Avg | Notes |
|----------|----------|--------|--------|-------|
| skip | 4 | +0.0% | +0.05% | **Working correctly** |
| normal | 5 | +1.3% | +1.3% | Unchanged |
| aggressive | 3 | +5.7% | +5.2% | Needs refinement |

**Net result**: v2 is slightly more conservative (-0.12% overall) but correctly avoids wasting compute on confident datasets.

## Next Steps

1. Add "threshold sensitivity" check to avoid over-optimizing on flat F1 curves
2. Remove calibration option (hurts more than helps)
3. Consider: If optimal threshold is near 0.5, don't change from default
