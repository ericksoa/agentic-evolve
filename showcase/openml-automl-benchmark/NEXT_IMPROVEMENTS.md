# Next Improvements for Adaptive Ensemble

## Analysis of Current Limitations

Based on benchmark results, the classifier:
- Achieves significant gains on 2/12 datasets (credit-g +18.3%, mozilla4 +8.9%)
- Correctly skips 9/12 datasets (avoids harm)
- Has marginal gains on 1/12 (kc2 +1.8%, not statistically significant)

**Key question**: How do we find MORE datasets that benefit?

---

## High-Impact Improvements

### 1. Hyperparameter Tuning for Base Model
**Potential gain**: +2-5% on many datasets

Currently we use fixed LogReg(C=0.5) or default LightGBM. Tuning these could:
- Improve base predictions → better probability estimates → better threshold optimization
- Find datasets where tuned model + threshold beats default

```python
clf = ThresholdOptimizedClassifier(
    base_estimator='auto',
    tune_base_model=True,  # NEW: auto-tune C, max_depth, etc.
)
```

### 2. Ensemble Thresholds
**Potential gain**: +1-3% stability

Instead of single optimal threshold, use multiple and vote:
- Train on bootstrap samples, get threshold for each
- Final prediction = majority vote across thresholds
- More robust to threshold variance

```python
clf = ThresholdOptimizedClassifier(
    ensemble_thresholds=5,  # NEW: use 5 thresholds
)
```

### 3. Cost-Sensitive Optimization
**Potential gain**: Domain-specific improvements

Allow users to specify misclassification costs:
- Medical: FN cost >> FP cost (missing disease is worse)
- Fraud: FP cost might be acceptable to catch more fraud

```python
clf = ThresholdOptimizedClassifier(
    cost_matrix={'fp': 1, 'fn': 10},  # NEW: FN costs 10x more
)
```

### 4. Probability Calibration Options
**Potential gain**: +1-2% on poorly calibrated models

Current `calibrate=True` uses isotonic. Add options:
- Platt scaling (sigmoid)
- Beta calibration
- Temperature scaling

```python
clf = ThresholdOptimizedClassifier(
    calibrate='platt',  # NEW: calibration method
)
```

---

## Medium-Impact Improvements

### 5. Feature Selection Integration
**Potential gain**: +1-3% on high-dimensional data

Auto-select features before threshold optimization:
- RFE (Recursive Feature Elimination)
- L1 regularization feature selection
- Mutual information

```python
clf = ThresholdOptimizedClassifier(
    feature_selection='auto',  # NEW
    max_features=20,
)
```

### 6. Adaptive CV Strategy
**Potential gain**: Better estimates, faster on large data

- Small datasets (<500): Use more folds (5-10)
- Large datasets (>5000): Use fewer folds (3) or holdout
- Imbalanced: Use stratified with proper minority sampling

```python
clf = ThresholdOptimizedClassifier(
    cv='auto',  # NEW: auto-select CV strategy
)
```

### 7. Meta-Learning for Detection
**Potential gain**: Find more datasets that benefit

Train a meta-model to predict "will threshold optimization help?" based on:
- Dataset meta-features (n_samples, n_features, imbalance_ratio)
- Model meta-features (probability distribution stats)
- Replace hand-tuned heuristics with learned model

---

## Lower-Impact but Useful

### 8. Threshold Regularization
Prevent overfitting to specific threshold values:
- Add penalty for thresholds far from 0.5
- Smooth threshold curve before finding optimum

### 9. Early Stopping for Large Datasets
For n > 10000, stop threshold search if:
- Gain plateaus
- Time budget exceeded

### 10. Multi-Objective Optimization
Optimize for multiple metrics simultaneously:
- Pareto front of recall vs precision
- Return threshold that balances user preferences

---

## Recommended Priority

| Priority | Feature | Effort | Expected Gain | Status |
|----------|---------|--------|---------------|--------|
| 1 | Hyperparameter tuning | Medium | +2-5% | ✅ Done |
| 2 | Cost-sensitive optimization | Low | Domain-specific | Pending |
| 3 | Probability calibration options | Low | +1-2% | Pending |
| 4 | Ensemble thresholds | Medium | +1-3% stability | Pending |
| 5 | Meta-learning detection | High | Find more winners | - |
| 6 | Feature selection | Medium | +1-3% on high-dim | - |

---

## Quick Wins (Low Effort)

1. ~~**Add XGBoost to auto model selection**~~ ✅ Done - Priority over LightGBM when available
2. **Stratified threshold search** - Sample thresholds more densely near 0.5
3. **Verbose mode improvements** - Show probability histograms
4. **Pickle/joblib serialization** - Save fitted models easily

---

## Implementation Notes

### XGBoost Support (Completed)
- XGBoost now has priority over LightGBM when both are available
- Auto model selection: LogReg < 2k samples, XGBoost/LightGBM >= 2k samples
- Benchmarks show XGBoost maintains the same +2.38% average gain

### Hyperparameter Tuning (Completed)
- New `tune_base_model=True` parameter enables GridSearchCV tuning
- LogisticRegression: Tunes C from [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
- XGBoost/LightGBM: Tunes n_estimators, max_depth, learning_rate
- Results are mixed: +0.42% on credit-g, -3.06% on kc2
- Conclusion: Useful option but not always beneficial; can cause overfitting

---

## Research Directions

1. **Why do only 2/12 datasets benefit significantly?**
   - Is it the dataset characteristics or the detection logic?
   - Can we find patterns in "winner" datasets?

2. **Threshold optimization vs class weight optimization**
   - Are they equivalent? When does each work better?
   - Could we optimize both jointly?

3. **Comparison with other threshold methods**
   - Youden's J statistic
   - Cost curves
   - ROC-based methods
