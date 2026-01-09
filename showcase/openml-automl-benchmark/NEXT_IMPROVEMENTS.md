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
| 2 | Cost-sensitive optimization | Low | Domain-specific | ✅ Done |
| 3 | Probability calibration options | Low | +1-2% | ✅ Done |
| 4 | Ensemble thresholds | Medium | +1-3% stability | ✅ Done |
| 5 | Meta-learning detection | High | Find more winners | ✅ Done |
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

### Cost-Sensitive Optimization (Completed)
- New `cost_matrix={'fp': 1, 'fn': 10}` parameter for cost-sensitive threshold search
- Minimizes total cost: cost = fp_cost * FP + fn_cost * FN
- High FN cost → lower threshold (catch more positives, accept more FP)
- High FP cost → higher threshold (be more selective, accept more FN)
- Use cases: medical diagnosis (FN worse), fraud detection (FN worse), spam (FP worse)
- When cost_matrix is set, optimize_for is ignored (cost takes precedence)

### Probability Calibration Options (Completed)
- Extended `calibrate` parameter to accept method names
- Options: False, True/'isotonic', 'sigmoid'/'platt'
- Isotonic: Non-parametric, flexible, can overfit on small data
- Sigmoid (Platt): Logistic regression, stable, assumes sigmoid shape
- True is backward-compatible alias for 'isotonic'
- Raises ValueError for unknown methods

### Ensemble Thresholds (Completed)
- New `ensemble_thresholds=5` parameter for bootstrap-based voting
- Each bootstrap sample produces its own optimal threshold
- Final predictions use majority vote across all thresholds
- More robust to threshold variance, especially on small datasets
- Stores threshold_ensemble_, threshold_ensemble_mean, threshold_ensemble_std
- Skipped when optimization is skipped (skip_if_confident)
- 5 new tests (51 total passing)

### Meta-Learning Detection (Completed)
- New `use_meta_detector=True` parameter to use learned predictor instead of heuristics
- New `meta_detector_threshold=0.5` parameter to control decision threshold
- Creates `adaptive_ensemble/meta_learning/` module with:
  - `MetaFeatureExtractor`: Extracts ~35 meta-features from (X, y) in ~0.5-2s
  - `MetaLearningDetector`: Trained predictor for P(will_help)
  - Training utilities for collecting data from OpenML
- Features extracted include:
  - Dataset features: n_samples, n_features, imbalance_ratio, etc.
  - Probability distribution: overlap_pct, class_separation, prob_mean, etc.
  - Threshold sensitivity: f1_range, potential_gain, threshold_distance, etc.
  - Derived interactions: imbalance_x_overlap, distance_x_gain, etc.
- Strategies: 'meta_aggressive', 'meta_normal', 'meta_skip'
- Fallback to heuristic rules if no pretrained model available
- Training scripts in `scripts/collect_training_data.py` and `scripts/train_meta_detector.py`
- 17 new tests (68 total passing)

---

## v7: Trust & Transparency Features

Goal: Give users confidence in threshold optimization decisions through confidence intervals, explanations, and visualizations.

### v7.1 Confidence Intervals (Completed)
- New `compute_confidence=True` parameter (default enabled)
- New `confidence_samples=100` parameter for bootstrap iterations
- Uses bootstrap resampling on CV probability predictions
- Stored in `threshold_confidence_` attribute with:
  - `point_estimate`: Best threshold estimate
  - `ci_low`, `ci_high`: 95% confidence interval bounds
  - `std`: Standard deviation of bootstrap estimates
  - `confidence`: 0-1 score (distance from uncertain 0.5)
  - `bootstrap_thresholds`: All bootstrap samples
- When optimization skipped, CI is [0.5, 0.5] with 100% confidence
- 5 new tests (60 total passing)

### v7.2 `.explain()` Method (Completed)
- New `explain()` method returns human-readable threshold analysis report
- Includes:
  - **Decision summary**: OPTIMIZE or SKIP with strategy name
  - **Confidence level**: High/Medium/Low based on CI
  - **Why this decision**: Strategy-specific explanations
  - **Recommendation**: Threshold to use and expected performance
  - **Cautions**: Risk warnings (small data, low confidence, etc.)
- Handles multiclass (explains why threshold NA)
- Returns "not fitted" message if called before fit()
- 4 new tests (64 total passing)

### v7.3 `.plot()` Visualization (Completed)
- New `plot(figsize=(10,6), show=True)` method for matplotlib visualization
- Shows F1 vs threshold curve with:
  - Blue curve of actual F1 scores across thresholds
  - Shaded confidence interval band
  - Red vertical line at optimal threshold
  - Gray dashed line at default threshold (0.5)
  - Green annotation showing % improvement
- Uses stored sensitivity data (`metric_scores`, `test_thresholds`)
- `show=False` returns Figure for saving: `fig.savefig('threshold.png')`
- Raises ValueError for multiclass (threshold NA)
- Raises RuntimeError if not fitted
- Requires matplotlib: `pip install matplotlib`
- 4 new tests (64 total passing)

### v7.4 Safety Mode (Completed)
- New `safety_mode=True` parameter enables holdout validation
- New `safety_margin=0.02` parameter controls rejection sensitivity
- When enabled:
  - Splits data 80/20 (stratified) before threshold optimization
  - Runs all CV analysis on 80% training split
  - Validates optimal threshold on held-out 20%
  - Compares holdout performance at optimal vs default (0.5)
- Rejection criteria:
  - Criterion 1: `holdout @ optimal < holdout @ default - safety_margin`
  - Criterion 2: CV-holdout gap > 10% (suggests overfitting)
- If rejected:
  - Reverts to default threshold (0.5)
  - Sets `optimization_skipped_ = True`
  - Records rejection reason in diagnostics
- Final model is fit on ALL data (train + holdout) with chosen threshold
- Stored in `safety_validation_` attribute with:
  - `holdout_at_optimal`, `holdout_at_default`: Scores on holdout
  - `cv_holdout_gap`: Difference between CV and holdout performance
  - `rejected`: Whether threshold was rejected
  - `rejection_reason`: Human-readable explanation
- 5 new tests (69 total passing)

### v7.5 sklearn Pipeline Integration (Completed)
- Verified full sklearn compatibility:
  - `clone()`: All parameters preserved through cloning
  - `Pipeline`: Works as final estimator with preprocessing steps
  - `cross_val_score()`: Compatible with sklearn CV utilities
  - `GridSearchCV`: Can search over classifier hyperparameters
  - `get_params()`/`set_params()`: Full roundtrip compatibility
- Key usage patterns tested:
  ```python
  # Pipeline example
  pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', ThresholdOptimizedClassifier(scale_features=False))
  ])

  # GridSearchCV example
  grid = GridSearchCV(clf, {'optimize_for': ['f1', 'f2'], 'cv': [2, 3]})
  ```
- Note: Set `scale_features=False` when using with external scaler
- 5 new tests (74 total passing)

---

## v8: Operating Point Selection (Pareto Frontier)

Goal: Transform from "find optimal threshold" to "explore the precision-recall tradeoff space and pick your operating point."

### v8.1 Core Infrastructure (Completed)
- New `_compute_operating_points()` method computes metrics at 50 thresholds
- New `_find_pareto_frontier()` method identifies non-dominated points
- Stores `operating_points_` attribute with:
  - `thresholds`: Array of 50 threshold values (0.01 to 0.99)
  - `precisions`, `recalls`, `f1_scores`, `f2_scores`: Metrics at each threshold
  - `fprs`, `specificities`: Additional metrics for ROC-style analysis
  - `pareto_mask`: Boolean mask of Pareto-optimal points
  - `selected_index`: Currently selected operating point
  - `selection_method`: How current point was selected

### v8.2 Selection Methods (Completed)
- New `set_operating_point(**constraints)` method with constraint options:
  - `min_recall=0.95`: Best precision where recall >= 95%
  - `min_precision=0.90`: Best recall where precision >= 90%
  - `max_fpr=0.05`: Best recall where FPR <= 5%
  - `target_f1=0.80`: Closest to F1=0.80
  - `target_f2=0.85`: Closest to F2=0.85
  - `threshold=0.35`: Direct threshold setting
- New `get_operating_point()` returns current point details
- New `list_operating_points(pareto_only=False)` returns DataFrame

### v8.3 Visualization (Completed)
- New `plot_operating_points()` method creates Pareto frontier visualization:
  - Precision-recall scatter plot
  - Blue line/points for Pareto frontier
  - Gray points for dominated operating points
  - Red star for current selected point with annotation
  - Iso-F1 curves for reference (F1=0.2, 0.4, 0.6, 0.8)

### Usage Example
```python
clf = ThresholdOptimizedClassifier()
clf.fit(X, y)

# Explore the tradeoff space
clf.plot_operating_points()              # Visual Pareto frontier
df = clf.list_operating_points()         # DataFrame of all points

# Select based on business constraints
clf.set_operating_point(min_recall=0.95)     # "Can't miss positives"
clf.set_operating_point(min_precision=0.90)  # "Can't have false alarms"

# See what you get
print(clf.get_operating_point())
# {'threshold': 0.28, 'precision': 0.72, 'recall': 0.96, 'f1': 0.82, ...}
```

- 12 new tests (86 total passing)

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
