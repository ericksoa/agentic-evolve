# MALLORN Astronomical Classification Challenge

Competing in the [Kaggle MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge) using evolutionary optimization to discover novel feature extraction and classification strategies for identifying Tidal Disruption Events (TDEs).

## Problem Overview

**MALLORN** = Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients

### The Task
Classify astronomical transients from simulated LSST light curves to identify **Tidal Disruption Events (TDEs)** - rare cosmic events where stars are torn apart by supermassive black holes.

### Why This Matters
- The Vera C. Rubin Observatory's LSST will produce a **100x increase** in observed transients
- Insufficient spectroscopic resources to follow up all targets
- Must prioritize objects based purely on photometric data
- TDEs are scientifically valuable but rare (~0.6% of dataset)

### Dataset
- **10,178** simulated light curves from real ZTF observations
- **Classes**: 64 TDEs, 727 nuclear supernovae, 1,407 AGN
- **Training**: 30% with spectral type + true redshift
- **Testing**: 70% with only photometric redshift
- **Bands**: LSST ugrizy (6 photometric bands)

### Metric
**F1 Score** - harmonic mean of precision and recall, ideal for imbalanced classification.

## Evolution Strategy

### What We're Evolving

1. **Feature Extraction Functions**
   - Light curve shape descriptors (rise time, decay rate, peak brightness)
   - Color evolution features (band-to-band ratios over time)
   - Variability metrics (amplitude, timescales)

2. **Classification Thresholds**
   - Decision boundaries for TDE vs non-TDE
   - Confidence weighting schemes

3. **Ensemble Strategies**
   - Feature combination methods
   - Classifier voting mechanisms

### Fitness Function
```python
def fitness(predictions, ground_truth):
    """F1 score for TDE class (binary: TDE vs non-TDE)"""
    return f1_score(ground_truth, predictions, pos_label='TDE')
```

## Quick Start

### Prerequisites
- Python 3.10+
- Kaggle API credentials configured

### Setup
```bash
cd showcase/mallorn-astro-classification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
kaggle competitions download -c mallorn-astronomical-classification-challenge
unzip mallorn-astronomical-classification-challenge.zip -d data/
```

### Run Baseline
```bash
python python/src/benchmark.py
```

### Run Evolution
```bash
# Use the /evolve skill
/evolve-ml mallorn-astro-classification
```

## File Structure

```
mallorn-astro-classification/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Kaggle competition data
│   ├── train.csv            # Training light curves
│   ├── test.csv             # Test light curves
│   └── sample_submission.csv
├── python/
│   └── src/
│       ├── __init__.py
│       ├── features.py      # Feature extraction (evolved)
│       ├── classifier.py    # Classification logic (evolved)
│       ├── baselines.py     # Baseline algorithms
│       ├── benchmark.py     # Evaluation harness
│       └── submit.py        # Generate Kaggle submission
└── mutations/               # Evolution history
    └── gen*.py              # Each generation's candidates
```

## Results Summary

| Generation | Algorithm | Holdout F1 | Public F1 | Status |
|------------|-----------|------------|-----------|--------|
| **Gen 10** | **Pure LogReg (t=0.40)** | **0.5025** | Pending | **+21% vs best** |
| Gen 9 | Pure LogReg (t=0.35) | 0.4611 | Pending | +11% vs best |
| Gen 4 | LR + XGBoost ensemble | ~0.41 | **0.4154** | Best Public |
| Gen 5-6 | Feature sel + LightGBM | ~0.32 | 0.32 | OVERFIT |
| Gen 1 | Baseline LogReg | 0.276 | - | Baseline |

**Key Finding**: Gen 5-6 showed that higher CV score ≠ better public score. Gen 9-10 use holdout validation to prevent overfitting.

## Evolution Journey

### Generation 1: Baseline Establishment (F1 = 0.276)
- [x] Logistic Regression with basic statistics: **F1 = 0.276**
- [x] XGBoost baseline: F1 = 0.180
- [x] Random Forest: F1 = 0.000 (fails on imbalanced data)

### Generation 2: Physics Feature Evolution (F1 = 0.368)
- [x] Power-law decay fitting (TDE signature: α ≈ -5/3)
- [x] Rise time / decay time asymmetry
- [x] Color evolution (g-r, r-i at peak and slope)
- [x] Smoothness metrics (reduced χ², scatter)
- [x] Blue excess indicator
- **Improvement: +33% over baseline**

### Generation 3: Threshold & Class Weight Optimization
- [x] Lower threshold (0.5 → 0.35) for better recall
- [x] Increased class weights (scale_pos_weight = 15)
- [x] Post-hoc threshold optimization on validation

### Generation 4: Ensemble Strategy (Public F1 = 0.4154)
- [x] Soft voting ensemble: LogReg (stable) + XGBoost (powerful)
- [x] Adaptive weight learning (optimized per-fold)
- [x] Combined threshold optimization
- **Best public leaderboard score: 0.4154**

### Generations 5-6: Overfitting Lesson (CV=0.55, Public=0.32)
- [x] Feature selection (20 of 121 features) - OVERFIT
- [x] LightGBM replaces XGBoost - MORE OVERFIT
- **Key lesson**: Higher CV score ≠ better public score

### Generation 9: Back to Basics (Holdout F1 = 0.4611)
- [x] Pure Logistic Regression (removed XGBoost)
- [x] Strong regularization (C=0.05)
- [x] Holdout validation protocol (splits 14, 17, 20)
- **+11% improvement over best public**

### Generation 10: Threshold Optimization (Holdout F1 = 0.5025)
- [x] Optimized threshold: 0.35 → 0.40
- [x] Fewer false positives, better precision
- **+21% improvement over best public (pending submission)**

### Key Insights
1. **Simpler is better**: Pure LogReg beats LR+XGB ensemble on tiny data
2. **Holdout validation is critical**: CV score can be misleading
3. **Threshold matters**: 0.40 > 0.35 for this imbalanced dataset
4. **Strong regularization prevents overfitting**: C=0.05 (not 1.0)

## References

- [MALLORN Paper (arXiv:2512.04946)](https://arxiv.org/abs/2512.04946)
- [Kaggle Competition](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
- [PLAsTiCC Challenge](https://www.kaggle.com/c/PLAsTiCC-2018/) - Similar prior challenge
