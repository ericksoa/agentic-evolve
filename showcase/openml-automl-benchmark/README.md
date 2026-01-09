# Adaptive Ensemble

**A drop-in sklearn classifier that intelligently optimizes decision thresholds - only when it helps.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![sklearn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)

---

## The Problem

Most classifiers use a default 0.5 decision threshold. On imbalanced datasets, this often means:
- Poor recall on the minority class
- Suboptimal F1 scores
- Missed predictions that matter

But blindly optimizing thresholds can also **hurt** performance on some datasets.

## The Solution

**Adaptive Ensemble** uses smart detection to decide **when** threshold optimization will help:

```python
from adaptive_ensemble import ThresholdOptimizedClassifier

clf = ThresholdOptimizedClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# See what the classifier decided
print(clf.summary())
# Strategy: aggressive (high potential gain detected)
# Optimal threshold: 0.05
# Potential gain: +18.1%
```

## Installation

```bash
# From GitHub
pip install git+https://github.com/ericksoa/agentic-evolve.git#subdirectory=showcase/openml-automl-benchmark

# Or clone and install locally
git clone https://github.com/ericksoa/agentic-evolve.git
cd agentic-evolve/showcase/openml-automl-benchmark
pip install -e .
```

## Results

### Benchmark Results (12 datasets)

| Metric | v4 (Current) | v1 (Always Optimize) |
|--------|--------------|----------------------|
| **Datasets Harmed** | **0** | 4 |
| **Datasets Improved** | 3 | 3 |
| **Avg Improvement** | **+2.38%** | +1.96% |
| **Best Improvement** | **+18.3%** (credit-g) | +18.6% |
| **Significant Improvements** | 2 (p<0.05) | - |

**Key insight**: Smart detection eliminates harm while preserving gains.

### Detailed Results

| Dataset | Strategy | Gain | 95% CI | Notes |
|---------|----------|------|--------|-------|
| credit-g | aggressive | **+18.3%*** | [+17.4%, +19.2%] | High overlap, optimal far from 0.5 |
| mozilla4 | aggressive | **+8.9%*** | [+8.5%, +9.3%] | Detected as high-gain candidate |
| kc2 | normal | **+1.8%** | [+0.9%, +2.7%] | Moderate optimization |
| diabetes | skip_low_gain | 0.0% | - | <1% potential gain, correctly skipped |
| blood-transfusion | skip_low_gain | 0.0% | - | Was -0.6% in v1 |
| ilpd | skip_low_gain | 0.0% | - | Was -0.9% in v1 |
| pc1 | skip_near_default | 0.0% | - | Was -3.3% in v1 |
| phoneme | skip_low_gain | 0.0% | - | Was -0.3% in v1 |
| 4 others | skip | 0.0% | - | Model already confident |

*\* Statistically significant (p < 0.05)*

### v4 Performance Improvements

v4 adds several performance optimizations:

| Feature | Benefit |
|---------|---------|
| **Single CV pass** | ~50% faster fitting (reuses probabilities from analysis) |
| **Auto model selection** | Better accuracy on large datasets (XGBoost/LightGBM for n≥2000) |
| **XGBoost support** | Priority over LightGBM when available (often better accuracy) |
| **Hyperparameter tuning** | Optional auto-tune base model with `tune_base_model=True` |
| **Metric selection** | Optimize for F1, F2, recall, precision, or F0.5 |
| **Bootstrap CIs** | Statistical confidence in benchmark results |
| **Multiclass support** | Graceful fallback (no crashes on multiclass data) |

**Speed comparison** (synthetic 1000-sample dataset):
- v3: ~2.1s (2 CV passes)
- v4: ~1.1s (1 CV pass)

**Auto model selection** (with `base_estimator='auto'`):
- Small datasets (<2000 samples): LogisticRegression (fast, avoids overfitting)
- Large datasets (≥2000 samples): XGBoost > LightGBM > LogReg (priority order)
- Falls back gracefully if XGBoost/LightGBM not installed

**Hyperparameter tuning** (with `tune_base_model=True`):
- LogisticRegression: Tunes C (regularization strength)
- XGBoost/LightGBM: Tunes n_estimators, max_depth, learning_rate
- Uses GridSearchCV with F1 scoring
- Note: May not always improve results; can cause overfitting on small datasets

## How It Works

The classifier analyzes your data before deciding whether to optimize:

```
1. Compute "overlap zone" - % of samples with uncertain predictions (0.3-0.7)
2. Test F1 at multiple thresholds to estimate potential gain
3. Check if optimal threshold is far enough from default 0.5

Strategy Selection:
├── overlap < 20%           → skip (model already confident)
├── F1 range < 0.02         → skip_flat (F1 doesn't vary with threshold)
├── potential_gain < 1%     → skip_low_gain (not worth optimizing)
├── thresh_distance < 0.10  → skip_near_default (optimal too close to 0.5)
├── gain > 5% AND dist > 0.15 → aggressive (wide search 0.05-0.60)
└── else                    → normal (standard search 0.20-0.55)
```

**The key discovery**: High uncertainty alone doesn't mean optimization helps. We need BOTH:
1. High overlap (model is uncertain)
2. Optimal threshold FAR from 0.5 (shifting threshold matters)

## Quick Start

### Basic Usage

```python
from adaptive_ensemble import ThresholdOptimizedClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Create imbalanced data
X, y = make_classification(n_samples=1000, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit with smart threshold optimization
clf = ThresholdOptimizedClassifier()
clf.fit(X_train, y_train)

# Check what strategy was used
print(f"Strategy: {clf.diagnostics_['strategy']}")
print(f"Overlap zone: {clf.overlap_pct_:.1f}%")
print(f"Potential gain: {clf.diagnostics_['potential_gain']*100:+.1f}%")
print(f"Optimal threshold: {clf.optimal_threshold_:.2f}")

# Predict
y_pred = clf.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
```

### Auto Model Selection

```python
# Automatically pick the best base model for your dataset size
clf = ThresholdOptimizedClassifier(base_estimator='auto')
clf.fit(X_train, y_train)

# The model selection is based on dataset size:
# < 2000 samples: LogisticRegression (fast, good for small data)
# 2000-10000 samples: LightGBM (if installed)
# > 10000 samples: LightGBM with more trees

print(clf.diagnostics_['auto_model'])
# e.g., 'lightgbm (2000 <= n < 10000)'
```

### Use Your Own Base Model

```python
from sklearn.ensemble import RandomForestClassifier

clf = ThresholdOptimizedClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    threshold_range='auto',  # Smart range selection
    cv=5
)
clf.fit(X_train, y_train)
```

### Optimize for Different Metrics

```python
# Emphasize recall (catch more positives, accept lower precision)
clf = ThresholdOptimizedClassifier(optimize_for='f2')

# Emphasize precision (fewer false positives)
clf = ThresholdOptimizedClassifier(optimize_for='f0.5')

# Pure recall optimization
clf = ThresholdOptimizedClassifier(optimize_for='recall')

# Pure precision optimization
clf = ThresholdOptimizedClassifier(optimize_for='precision')
```

### Force Optimization (Skip Smart Detection)

```python
# Always optimize, even when detection says skip
clf = ThresholdOptimizedClassifier(
    skip_if_confident=False,
    threshold_range=(0.10, 0.60)
)
```

### Hyperparameter Tuning

```python
# Auto-tune base model hyperparameters (uses GridSearchCV)
clf = ThresholdOptimizedClassifier(
    tune_base_model=True,  # Tune C for LogReg, or n_estimators/max_depth/lr for boosting
)
clf.fit(X_train, y_train)

# See tuning results
print(clf.diagnostics_['tuning'])
# {'status': 'completed', 'best_params': {'C': 0.1}, 'best_score': 0.82, 'model_type': 'LogisticRegression'}

# Combine with auto model selection
clf = ThresholdOptimizedClassifier(
    base_estimator='auto',
    tune_base_model=True,
)
```

### Get Detailed Diagnostics

```python
clf = ThresholdOptimizedClassifier()
clf.fit(X_train, y_train)

print(clf.summary())
# ThresholdOptimizedClassifier Summary
# ========================================
#
# Uncertainty Analysis:
#   Overlap zone: 85.5%
#   Class separation: 0.062
#   F1 range across thresholds: 0.670
#   Potential gain: +18.1%
#   Strategy: aggressive
#
# Optimization:
#   Skipped: False
#   Threshold range: (0.05, 0.6)
#   Optimal threshold: 0.050
#   CV F1: 0.824
```

## API Reference

### ThresholdOptimizedClassifier

The recommended classifier for most use cases.

```python
ThresholdOptimizedClassifier(
    base_estimator=None,      # None, 'auto', or sklearn estimator
    optimize_for='f1',        # 'f1', 'f2', 'f0.5', 'recall', 'precision'
    threshold_range='auto',   # 'auto' or tuple like (0.20, 0.55)
    threshold_steps=20,       # Number of candidates to try
    cv=3,                     # CV folds for optimization
    scale_features=True,      # Standardize features
    skip_if_confident=True,   # Skip when detection says it won't help
    tune_base_model=False,    # Auto-tune hyperparameters with GridSearchCV
    calibrate=False,          # Probability calibration (usually not needed)
    random_state=42
)
```

**base_estimator options:**
- `None`: Default LogisticRegression with balanced weights
- `'auto'`: Auto-select based on dataset size (LogReg < 2k, XGBoost/LightGBM >= 2k)
- Custom estimator: Any sklearn-compatible classifier with `predict_proba`

**tune_base_model tuning parameters:**
- LogisticRegression: `C` (regularization strength) from [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
- XGBoost/LightGBM: `n_estimators`, `max_depth`, `learning_rate`

**Attributes after fitting:**
- `optimal_threshold_`: The learned optimal threshold
- `overlap_pct_`: % of samples in uncertain zone (0.3-0.7)
- `class_separation_`: Difference in mean prob between classes
- `optimization_skipped_`: Whether optimization was skipped
- `diagnostics_`: Full dict of detection metrics (includes `tuning` when enabled)
- `imbalance_ratio_`: Class imbalance in training data

### AdaptiveEnsembleClassifier

Full adaptive ensemble with RFE feature selection. More powerful but can overfit on small datasets.

```python
AdaptiveEnsembleClassifier(
    threshold='auto',         # 'auto' or float
    n_features='auto',        # Feature selection
    ensemble_size='auto',     # Number of models
    verbose=False,
    random_state=42
)
```

### DatasetAnalyzer

Analyze dataset characteristics and get strategy recommendations.

```python
from adaptive_ensemble import DatasetAnalyzer

analyzer = DatasetAnalyzer()
profile = analyzer.analyze(X_train, y_train)
print(analyzer.summary())
```

## When to Use This

**Good fit:**
- Binary classification with class imbalance (1.5x - 5x)
- Small to medium datasets (<50k samples)
- When F1 score matters more than accuracy
- Quick experiments without hyperparameter tuning

**Consider alternatives:**
- Extremely imbalanced data (>10x) - try SMOTE or specialized methods
- Very large datasets - XGBoost/LightGBM with custom thresholds

**Multiclass support:**
- The classifier gracefully handles multiclass problems
- Falls back to standard argmax prediction (no threshold optimization)
- A warning is issued when multiclass is detected

## Origin Story

This library emerged from **LLM-guided evolution experiments** on the OpenML benchmark:

1. Started with standard LogisticRegression baseline
2. LLM proposed 27 variations across 9 generations
3. **Threshold optimization emerged as the #1 technique**
4. Complex approaches (XGBoost, domain features) often hurt
5. Validated on 16 diverse datasets
6. **Deep dive revealed WHY some datasets benefit** (high overlap + optimal far from 0.5)
7. **v3 implements smart detection** to only optimize when it helps

See [EVOLUTION_RESULTS.md](./EVOLUTION_RESULTS.md) and [V2_ANALYSIS.md](./V2_ANALYSIS.md) for the full story.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (20 tests)
pytest tests/ -v

# Run benchmark (requires openml)
pip install -e ".[benchmark]"
python benchmark_v3.py
```

## License

MIT License - see [LICENSE](./LICENSE)

## Citation

If you use this in research:

```bibtex
@software{adaptive_ensemble,
  title = {Adaptive Ensemble: Smart Threshold Optimization for Imbalanced Classification},
  author = {Agentic Evolve Project},
  year = {2025},
  url = {https://github.com/ericksoa/agentic-evolve}
}
```
