# Adaptive Ensemble

**A drop-in sklearn classifier that automatically handles imbalanced data.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![sklearn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)

---

## The Problem

Most classifiers use a default 0.5 decision threshold. On imbalanced datasets, this often means:
- Poor recall on the minority class
- Suboptimal F1 scores
- Missed predictions that matter

## The Solution

**Adaptive Ensemble** automatically optimizes the decision threshold for your data. No tuning required.

```python
from adaptive_ensemble import ThresholdOptimizedClassifier

clf = ThresholdOptimizedClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(f"Learned threshold: {clf.optimal_threshold_}")  # e.g., 0.35 instead of 0.50
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

Tested on **16 OpenML datasets** with varying imbalance ratios:

| Metric | ThresholdOptimized | LogReg Baseline |
|--------|-------------------|-----------------|
| **Win Rate** | **44%** | 38% |
| **Avg Improvement** | **+1.7%** | - |
| **Best Improvement** | **+18.5%** (credit-g) | - |

Key findings:
- **Biggest wins on moderately imbalanced data** (2-3x ratio)
- **Neutral on highly imbalanced data** (>5x) - doesn't hurt, rarely helps
- **Simple and fast** - adds <1 second to training time

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

# Fit with automatic threshold optimization
clf = ThresholdOptimizedClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"Optimal Threshold: {clf.optimal_threshold_:.3f}")
```

### Use Your Own Base Model

```python
from sklearn.ensemble import RandomForestClassifier

clf = ThresholdOptimizedClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    threshold_range=(0.20, 0.60),
    cv=5
)
clf.fit(X_train, y_train)
```

### Analyze Your Dataset

```python
from adaptive_ensemble import DatasetAnalyzer

analyzer = DatasetAnalyzer()
profile = analyzer.analyze(X_train, y_train)
print(analyzer.summary())

# Output:
# Dataset Profile:
#   Samples: 750 (small)
#   Features: 20 (many)
#   Imbalance ratio: 2.33 (imbalanced)
#
# Recommended Strategy:
#   Threshold: 0.35
#   Feature selection: 8
#   Complex models: No (use LogReg)
```

## API Reference

### ThresholdOptimizedClassifier

The recommended classifier for most use cases.

```python
ThresholdOptimizedClassifier(
    base_estimator=None,      # Base classifier (default: LogisticRegression)
    threshold_range=(0.20, 0.55),  # Range to search
    threshold_steps=15,       # Number of candidates
    cv=3,                     # CV folds for optimization
    scale_features=True,      # Standardize features
    random_state=42
)
```

**Attributes after fitting:**
- `optimal_threshold_`: The learned optimal threshold
- `imbalance_ratio_`: Class imbalance in training data
- `classes_`: Unique class labels

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
DatasetAnalyzer()
    .analyze(X, y)  # Returns DatasetProfile
    .summary()      # Returns human-readable string
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
- Multiclass problems - this library focuses on binary classification

## How It Works

1. **Detects imbalance** - Computes class ratio automatically
2. **Searches thresholds** - Tests 15 candidates via cross-validation
3. **Optimizes for F1** - Finds threshold that maximizes F1 score
4. **Fits final model** - Trains on full data with optimal threshold

The key insight: **lowering the threshold from 0.5 to ~0.35 often improves F1 by 5-20% on imbalanced data**, with zero additional complexity.

## Origin Story

This library emerged from **LLM-guided evolution experiments** on the OpenML benchmark:

1. Started with standard LogisticRegression baseline
2. LLM proposed 27 variations across 9 generations
3. **Threshold optimization emerged as the #1 technique**
4. Complex approaches (XGBoost, domain features) often hurt
5. Validated on 16 diverse datasets

See [EVOLUTION_RESULTS.md](./EVOLUTION_RESULTS.md) for the full story.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmark (requires openml)
pip install -e ".[benchmark]"
python -m adaptive_ensemble.benchmark
```

## License

MIT License - see [LICENSE](./LICENSE)

## Citation

If you use this in research:

```bibtex
@software{adaptive_ensemble,
  title = {Adaptive Ensemble: Automatic Threshold Optimization for Imbalanced Classification},
  author = {Agentic Evolve Project},
  year = {2025},
  url = {https://github.com/ericksoa/agentic-evolve}
}
```
