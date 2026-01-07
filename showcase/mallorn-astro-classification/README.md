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

| Algorithm | F1 Score | Notes |
|-----------|----------|-------|
| **Evolved (ours)** | TBD | - |
| Random Forest Baseline | TBD | Standard features |
| XGBoost Baseline | TBD | Standard features |

## Evolution Journey

*To be documented as we evolve solutions*

### Generation 1: Baseline Establishment
- [ ] Random Forest with hand-crafted features
- [ ] XGBoost with hand-crafted features
- [ ] Simple neural network

### Generation 2+: Feature Evolution
- [ ] Evolve feature extraction functions
- [ ] Evolve feature combinations
- [ ] Evolve classifier architectures

## References

- [MALLORN Paper (arXiv:2512.04946)](https://arxiv.org/abs/2512.04946)
- [Kaggle Competition](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
- [PLAsTiCC Challenge](https://www.kaggle.com/c/PLAsTiCC-2018/) - Similar prior challenge
