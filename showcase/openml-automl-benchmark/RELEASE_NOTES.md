# Release Notes - Adaptive Ensemble

## v9.0: ThresholdOptimizer (Wrapper Mode)

**Release significance: VERY HIGH**

### Why This Matters

The biggest adoption barrier for threshold optimization was: "I already have a model I've tuned." Users had to switch to `ThresholdOptimizedClassifier` and lose control over their base model.

**v9 solves this with `ThresholdOptimizer` - a simple wrapper that adds threshold optimization to ANY sklearn-compatible classifier.**

```python
from adaptive_ensemble import ThresholdOptimizer

# Wrap ANY classifier in one line
clf = ThresholdOptimizer(YourModel())
clf.fit(X, y)

# All v7/v8 features work
clf.explain()                             # Human-readable report
clf.plot()                                # Threshold curve
clf.set_operating_point(min_recall=0.95)  # Business constraint
clf.plot_operating_points()               # Pareto frontier
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Universal compatibility** | Works with LogisticRegression, RandomForest, XGBoost, LightGBM, any model with `predict_proba()` |
| **Full sklearn integration** | clone(), Pipeline, GridSearchCV, cross_val_score all work |
| **Strategy modes** | `'auto'` (smart detection), `'always'` (force optimize), `'never'` (baseline) |
| **v7 features** | Confidence intervals, explain(), plot(), safety_mode |
| **v8 features** | Operating points, Pareto frontier, set_operating_point() |
| **Probability calibration** | Isotonic and sigmoid (Platt) calibration built-in |

### Usage Examples

```python
# Basic usage
from sklearn.ensemble import RandomForestClassifier
from adaptive_ensemble import ThresholdOptimizer

clf = ThresholdOptimizer(
    RandomForestClassifier(n_estimators=100),
    optimize_for='f1',
    strategy='auto'
)
clf.fit(X_train, y_train)
print(clf.explain())

# In a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ThresholdOptimizer(LogisticRegression()))
])
pipe.fit(X_train, y_train)

# With operating point selection
clf.fit(X_train, y_train)
clf.set_operating_point(min_recall=0.95)  # Business constraint
print(clf.get_operating_point())           # See threshold, precision, recall
clf.plot_operating_points()                # Visualize Pareto frontier
```

### Comparison

| Feature | ThresholdOptimizedClassifier | ThresholdOptimizer |
|---------|------------------------------|-------------------|
| **Base model** | Built-in (auto-select) | User-provided |
| **Model tuning** | `tune_base_model=True` | User's responsibility |
| **Feature scaling** | `scale_features=True` | User's responsibility |
| **Meta-detector** | `use_meta_detector=True` | Simpler heuristics |
| **v7/v8 features** | Full | Full |
| **Best for** | Quick start, auto-everything | Integration with existing pipelines |

### Test Coverage

35 new tests covering:
- Core wrapper functionality (11 tests)
- Strategy detection (6 tests)
- v7 features: confidence, explain, plot (8 tests)
- v8 features: operating points, Pareto (10 tests)

**Total: 121 tests passing**

---

## v8.0: Operating Point Selection (Pareto Frontier)

**Release significance: HIGH**

### Why This Matters

Most ML classifiers give you a single answer: "here's the best threshold." But in production, "best" depends entirely on your business context:

| Domain | Priority | Constraint |
|--------|----------|------------|
| Cancer screening | Don't miss cancer | `min_recall=0.99` |
| Spam filtering | Don't block real email | `min_precision=0.95` |
| Fraud detection | Limit false alarms | `max_fpr=0.05` |
| Credit scoring | Balance risk/reward | `target_f1=0.80` |

**v8 transforms threshold optimization from a black box into a decision-support tool.** Instead of trusting an algorithm to pick "optimal," you can:

1. **See the tradeoff space** - Visualize all possible operating points
2. **Understand what you're giving up** - Pareto frontier shows efficient tradeoffs
3. **Set business constraints** - Express requirements in domain terms
4. **Make informed decisions** - Pick the point that fits your risk tolerance

### The Pareto Frontier Concept

Borrowed from economics/portfolio theory: the Pareto frontier contains all "non-dominated" points - operating points where you cannot improve precision without sacrificing recall (or vice versa). Points below the frontier are strictly worse; points on the frontier represent genuine tradeoffs.

```
Precision
    │      ╭──────╮
1.0 │     ╱        ╲  ← Pareto Frontier
    │    ╱    ★     ╲    (efficient tradeoffs)
0.8 │   ╱  selected  ╲
    │  ╱              ╲
0.6 │ ╱    ○ ○         ╲
    │╱   ○   ○ ○        ╲
0.4 │  ○  dominated     ╲
    │     points         ╲
    └────────────────────────
    0.4   0.6   0.8   1.0  Recall
```

### New API

```python
clf = ThresholdOptimizedClassifier()
clf.fit(X, y)

# Explore the tradeoff space
clf.plot_operating_points()              # Visual Pareto frontier
df = clf.list_operating_points()         # DataFrame of all points

# Select based on business constraints
clf.set_operating_point(min_recall=0.95)     # "Can't miss positives"
clf.set_operating_point(min_precision=0.90)  # "Can't have false alarms"
clf.set_operating_point(max_fpr=0.05)        # "Limit false positive rate"
clf.set_operating_point(target_f1=0.80)      # "Get closest to F1=0.80"

# See what you get
point = clf.get_operating_point()
# {'threshold': 0.28, 'precision': 0.72, 'recall': 0.96, 'f1': 0.82, ...}
```

### What This Enables

1. **Stakeholder conversations**: Show the plot to business stakeholders. "If we catch 95% of fraud, we'll have 20% false alarms. If we accept 90% catch rate, false alarms drop to 8%. Which do you prefer?"

2. **Regulatory compliance**: Some domains require specific recall levels (e.g., "must detect 99% of X"). Now you can set that constraint directly.

3. **A/B testing guidance**: Pick two operating points and test which performs better in production.

4. **Cost-benefit analysis**: Combine with cost_matrix to find the operating point that minimizes total business cost.

---

## v7.0: Trust & Transparency

**Release significance: MEDIUM-HIGH**

### Features

| Feature | What It Does |
|---------|--------------|
| **Confidence Intervals** | Bootstrap-based 95% CI for threshold estimates |
| **`.explain()` method** | Human-readable analysis: why optimize/skip, confidence level, recommendations |
| **`.plot()` method** | F1 vs threshold curve with CI band and improvement annotation |
| **Safety Mode** | Holdout validation to catch overfitting before it hurts |
| **sklearn Integration** | Verified: clone(), Pipeline, cross_val_score(), GridSearchCV |

### Why Trust Matters

Threshold optimization can overfit, especially on small datasets. v7 gives you tools to:

1. **Know when to trust the threshold** - CI tells you if 0.35 really means 0.35 or could be anywhere from 0.25-0.45
2. **Understand the decision** - `.explain()` tells you why in plain English
3. **Validate before deploying** - Safety mode catches overfitting with held-out data
4. **Integrate confidently** - Full sklearn compatibility means it works in your existing pipelines

---

## v6.0: Meta-Learning Detection

**Release significance: MEDIUM**

Replaced hard-coded heuristics with a trained meta-model that predicts "will threshold optimization help for this dataset?" based on 35+ meta-features.

---

## v5.0: Advanced Optimization Options

**Release significance: MEDIUM**

- Hyperparameter tuning for base models
- Cost-sensitive optimization (custom FP/FN costs)
- Multiple probability calibration methods
- Ensemble thresholds (bootstrap voting)

---

## Version History

| Version | Focus | Tests |
|---------|-------|-------|
| v9.0 | ThresholdOptimizer (Wrapper Mode) | 121 |
| v8.0 | Operating Point Selection | 86 |
| v7.0 | Trust & Transparency | 74 |
| v6.0 | Meta-Learning Detection | 68 |
| v5.0 | Advanced Optimization | 51 |
| v4.0 | Core Threshold Optimization | ~40 |

---

## What's Next?

See `NEXT_IMPROVEMENTS.md` for the roadmap. Key candidates for v10:

1. **Segment-specific thresholds** - Different thresholds for different customer segments
2. **Production deployment** - Model export (ONNX), monitoring, drift detection
3. **Multi-label support** - Extend to multi-label classification
4. **Online threshold adaptation** - Update thresholds as data distribution shifts
