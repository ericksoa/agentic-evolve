# MALLORN TDE Classification - Evolution Log

This document captures learnings from each generation of the `/evolve-ml` optimization process. Use this as a reference for future ML evolution tasks, especially on imbalanced classification problems.

---

## Problem Context

- **Task**: Binary classification (TDE vs non-TDE)
- **Challenge**: Extreme class imbalance (~5% positive class)
- **Data**: Small training sets (100-170 objects per split, 3-12 TDEs each)
- **Metric**: F1 Score (optimizes precision-recall tradeoff)

---

## Generation 1: Baseline Establishment

**F1 Score: 0.276**

### What We Tried
- Logistic Regression with basic band statistics
- Random Forest with default parameters
- XGBoost with default parameters
- Gradient Boosting

### Results
| Model | F1 Score |
|-------|----------|
| Logistic Regression | 0.276 |
| XGBoost | 0.180 |
| LightGBM | 0.100 |
| Random Forest | 0.000 |

### Key Learnings

1. **Simple models win on tiny data**: Logistic Regression outperformed all tree-based methods. With only 12 TDEs per split, complex models overfit immediately.

2. **Random Forest fails catastrophically**: With balanced class weights, RF still predicts all negatives. The random sampling in RF loses the few positive examples.

3. **High variance is expected**: With only ~2 TDEs per CV fold, results swing wildly between 0.0 and 0.5 F1.

### Recommendations for Similar Problems
- Start with Logistic Regression as your baseline
- Don't trust tree methods on datasets with <50 positive examples
- Use stratified CV, but expect high variance

---

## Generation 2: Physics-Based Feature Engineering

**F1 Score: 0.368 (+33% improvement)**

### What We Added

1. **Power-law decay fitting**
   - TDEs have characteristic t^(-5/3) decay
   - Fit decay constant (t0) and exponent (alpha)
   - Computed `tde_alpha_diff = |alpha - (-5/3)|`

2. **Rise/Decay asymmetry**
   - TDEs: fast rise, slow decay
   - Feature: `rise_time / decay_time`

3. **Color evolution**
   - Color at peak (g-r, r-i)
   - Color slope over time
   - TDEs are blue and evolve predictably

4. **Smoothness metrics**
   - Reduced chi-squared from polynomial fit
   - Scatter relative to trend
   - TDEs are smooth; SNe have bumps

5. **Global TDE indicators**
   - Mean decay alpha across bands
   - Blue excess ratio (g/r flux)
   - Peak-to-baseline brightness ratio

### Key Learnings

1. **Domain knowledge beats data**: Physics-informed features provided +33% improvement. The power-law decay and color evolution are exactly what astronomers use.

2. **Feature quality > quantity**: We went from 85 to 122 features. The new 37 features carried most of the signal.

3. **Robustness through redundancy**: Computing the same physics (e.g., decay alpha) across multiple bands provides robustness when some bands have missing data.

### Recommendations for Similar Problems
- Invest time in domain research before feature engineering
- Physics/domain-based features often outperform generic statistical ones
- Compute key features across multiple views (bands/channels) for robustness

---

## Generation 3: Threshold & Class Weight Optimization

### What We Changed

1. **Lower threshold**: 0.5 → 0.35
   - Default 0.5 threshold assumes balanced classes
   - Lower threshold improves recall for rare class

2. **Increased class weights**: scale_pos_weight = 15
   - Penalize missing a TDE 15x more than a false alarm

3. **Post-hoc threshold optimization**
   - Find optimal threshold on validation probabilities
   - Typically lands around 0.30-0.40

### Key Learnings

1. **Threshold is a hyperparameter**: For imbalanced problems, the threshold matters as much as model choice. Always tune it.

2. **Optimal threshold varies by split**: With tiny samples, the optimal threshold ranges from 0.25 to 0.45 depending on the data.

3. **Don't optimize threshold on training data**: This leads to overfitting. Use validation data or nested CV.

### Recommendations for Similar Problems
- Never use 0.5 threshold for imbalanced classification
- Grid search threshold on validation data
- Consider asymmetric costs (is FN worse than FP?)

---

## Generation 4: Ensemble Strategy

**F1 Score: 0.415 (+50% total improvement)**

### What We Built

Soft-voting ensemble combining:
1. **Logistic Regression**: Stable, works well on small data
2. **XGBoost**: Powerful, can capture nonlinearities

With:
- Adaptive weight learning (tested weights 0.3-0.7)
- Combined probability threshold optimization
- Shared preprocessing (imputation + scaling)

### Architecture
```python
ensemble_proba = lr_weight * lr_proba + xgb_weight * xgb_proba
prediction = (ensemble_proba >= threshold).astype(int)
```

### Key Learnings

1. **Ensembles reduce variance**: The ensemble had lower std across splits than either model alone.

2. **Complementary models work best**: LR provides stable baseline; XGB adds nonlinear power. Neither is strictly better.

3. **Weight optimization matters**: Optimal weights varied (0.4-0.6 for LR), showing splits have different characteristics.

4. **Keep it simple**: Soft voting with 2 models beat complex stacking attempts.

### Recommendations for Similar Problems
- Combine a linear model (stable) with a tree model (powerful)
- Use soft voting (probabilities) not hard voting (predictions)
- Optimize ensemble weights on validation data
- Start with 2 models; more isn't always better on small data

---

## Generation 5: Feature Selection

**F1 Score: 0.552 (+33% from Gen 4)**

### What We Discovered

Feature importance analysis revealed that only 20 of 121 features carry most signal:

| Rank | Feature | Type |
|------|---------|------|
| 1 | g_skew | Statistical |
| 2 | r_scatter | Smoothness |
| 3 | r_skew | Statistical |
| 4 | i_skew | Statistical |
| 5 | i_kurtosis | Statistical |
| 6 | r_kurtosis | Statistical |
| 7 | u_max | Band stat |
| 8 | g_min | Band stat |
| 9 | u_range | Band stat |
| 10 | y_mean | Band stat |

### Feature Count vs F1 Score

| Features | F1 Score |
|----------|----------|
| 10 | 0.454 |
| **20** | **0.552** |
| 30 | 0.546 |
| 50 | 0.522 |
| 75 | 0.531 |
| 121 (all) | 0.492 |

### Key Learnings

1. **Less is more**: 20 features outperform 121 features by 12%
2. **Statistical > Physics on tiny data**: Skewness and kurtosis beat power-law fits
3. **Simpler features generalize better**: Complex physics fits overfit on small samples
4. **Feature selection is critical**: Reduces noise, improves generalization

### Surprising Finding

The physics-based features (decay_alpha, tde_alpha_diff) ranked LOW in importance. On tiny data, the fitting errors dominate the signal. Simple statistics (skewness, scatter) are more robust.

### Recommendations

- Always do feature selection on small datasets
- Start with simple statistics before complex domain features
- The best features may not be the most intuitive ones

---

## Generation 6: LightGBM Replaces XGBoost

**F1 Score: 0.575 (+4% from Gen 5, +108% total)**

### What We Tested

| Ensemble | F1 Score |
|----------|----------|
| LR only | 0.358 |
| XGB only | 0.546 |
| LGB only | 0.567 |
| LR + XGB (Gen 5) | 0.552 |
| **LR + LGB** | **0.575** |
| XGB + LGB | 0.568 |
| LR + XGB + LGB | 0.570 |

### Key Learnings

1. **LightGBM > XGBoost on small data**: 0.567 vs 0.546 (solo performance)
2. **2 models > 3 models**: Adding XGBoost to LR+LGB hurt performance
3. **Simpler ensembles generalize better**: More models add noise on tiny samples
4. **Optimal weights**: LR=0.3, LGB=0.7

### Why LightGBM Wins

- Better leaf-wise growth (vs XGBoost's level-wise)
- More efficient handling of categorical-like features
- Less prone to overfitting on small data
- Faster training (bonus for iteration)

---

## Summary: Evolution Trajectory

```
Gen 1: Baseline LogReg           F1 = 0.276
       ↓ +33% (physics features)
Gen 2: + TDE features            F1 = 0.368
       ↓ (threshold tuning)
Gen 3: + Threshold optimization  F1 ~ 0.38
       ↓ +12% (ensemble)
Gen 4: + Ensemble (LR + XGB)     F1 = 0.415
       ↓ +33% (feature selection)
Gen 5: + Feature selection       F1 = 0.552
       ↓ +4% (LightGBM)
Gen 6: + LightGBM replaces XGB   F1 = 0.575
                                 ─────────
                         Total: +108% improvement
```

---

## Anti-Patterns Discovered

1. **Don't use Random Forest on tiny imbalanced data** - It fails completely

2. **Don't optimize threshold on training data** - Overfits badly

3. **Don't trust single-fold results** - Variance is huge with <10 positive examples

4. **Don't add complexity without validation** - Deeper trees and more estimators hurt on small data

5. **Don't ignore domain knowledge** - Generic features underperform physics-based ones

---

## Transferable Strategies

These strategies likely transfer to other imbalanced classification problems:

| Strategy | When to Use |
|----------|-------------|
| Start with LogReg | Always, as baseline |
| Physics/domain features | When domain knowledge exists |
| Threshold optimization | Any imbalanced problem |
| LR + XGB ensemble | Small data, <1000 samples |
| Multiple seeds/splits | When positive class <50 |

---

## Files Modified

- `python/src/features.py`: Added TDE physics features (Gen 2)
- `python/src/classifier.py`: Ensemble classifier (Gen 4)
- `README.md`: Results summary
- `EVOLUTION_LOG.md`: This document

---

## Next Steps (Future Generations)

If continuing evolution:

1. **Gen 5**: Try LightGBM in ensemble (often better than XGB on small data)
2. **Gen 6**: Feature selection (remove noisy features)
3. **Gen 7**: Calibrated probabilities (Platt scaling)
4. **Gen 8**: Semi-supervised learning (use unlabeled test data)
