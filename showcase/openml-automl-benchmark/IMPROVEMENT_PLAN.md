# Adaptive Ensemble v4 Improvement Plan

## Checklist

- [x] **1. Optimize for other metrics** - Add `optimize_for` parameter (recall/precision/f1/f2)
- [x] **2. Auto-select base model** - LightGBM/XGBoost for large datasets
- [x] **3. Speed optimization** - Combine analysis + optimization CV passes
- [x] **4. Confidence intervals** - Bootstrap CI for benchmark results
- [x] **5. Deep dive kc2 vs diabetes** - Why similar profiles, different outcomes?
- [x] **6. Test on 50+ datasets** - Validate detection generalizes (CC-18 suite script)
- [ ] **7. Multiclass support** - Extend beyond binary classification

---

## 1. Optimize for Other Metrics

**Goal**: Allow users to optimize for recall, precision, or F-beta scores.

```python
clf = ThresholdOptimizedClassifier(optimize_for='recall')
clf = ThresholdOptimizedClassifier(optimize_for='f2')  # Emphasize recall
```

**Implementation**:
- Add `optimize_for` parameter: 'f1', 'f2', 'f0.5', 'recall', 'precision'
- Update `_optimize_threshold()` to use specified metric
- Update sensitivity analysis to use same metric

---

## 2. Auto-Select Base Model

**Goal**: Use better models for larger datasets.

| Dataset Size | Model |
|-------------|-------|
| < 2000 | LogisticRegression |
| 2000-10000 | LightGBM (if available) |
| > 10000 | LightGBM with more trees |

**Implementation**:
- Add `base_estimator='auto'` option
- Detect dataset size in fit()
- Fall back to LogReg if LightGBM not installed

---

## 3. Speed Optimization

**Goal**: Reduce from 2 CV passes to 1.

**Current**:
1. `_analyze_uncertainty()` - CV to get probabilities
2. `_optimize_threshold()` - CV to find best threshold

**Improved**:
1. Single CV pass that collects both probabilities AND evaluates thresholds

---

## 4. Confidence Intervals

**Goal**: Add statistical rigor to benchmarks.

**Implementation**:
- Bootstrap resampling (100 iterations)
- Report mean ± 95% CI
- Significance test vs baseline

---

## 5. Deep Dive: kc2 vs diabetes

**Question**: Both have ~40% overlap, but kc2 improves +1.8% while diabetes is neutral.

**Analysis needed**:
- Compare probability distributions
- Compare F1 curves
- Identify distinguishing features

---

## 6. Test on 50+ Datasets

**Goal**: Validate detection logic generalizes.

**Sources**:
- OpenML CC-18 suite (72 datasets)
- Filter to binary classification
- Run benchmark, analyze failures

---

## 7. Multiclass Support

**Goal**: Extend to multiclass classification.

**Options**:
- One-vs-Rest with per-class thresholds
- Macro/micro averaging
- Class-specific optimization

---

## Progress Log

| Step | Status | Commit | Notes |
|------|--------|--------|-------|
| 1 | Done | 1112e30 | Added optimize_for param (f1/f2/f0.5/recall/precision) |
| 2 | Done | 9f2c096 | Auto-select model: LogReg < 2k, LightGBM >= 2k |
| 3 | Done | ec01f21 | Single CV pass: reuse probs from analysis |
| 4 | Done | c6afc9b | benchmark_v4.py with bootstrap CIs + sig tests |
| 5 | Done | 12944b7 | kc2 gains (thresh 0.15 from 0.5), diabetes skipped (thresh 0.10) |
| 6 | Done | TBD | benchmark_cc18.py for CC-18 suite (binary datasets) |
| 7 | Pending | - | - |
