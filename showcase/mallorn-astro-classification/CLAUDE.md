# MALLORN Astronomical Classification - Claude Workflow

## Quick Reference

### Competition Details
- **Goal**: Identify TDEs (Tidal Disruption Events) from LSST light curves
- **Metric**: F1 Score (higher is better)
- **Classes**: TDE (~0.6%), Nuclear SN (~7%), AGN (~14%)
- **Data**: 10,178 light curves in 20 train/test splits

### Data Location
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/mallorn-astro-classification
```

## Setup Instructions

### 1. Download Data (User Must Do Manually)

The competition data cannot be downloaded programmatically until the user accepts the competition rules on Kaggle.

```bash
# User must first:
# 1. Go to https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge
# 2. Click "Join Competition" and accept the rules
# 3. Then run:

kaggle competitions download -c mallorn-astronomical-classification-challenge -p data/
cd data && unzip mallorn-astronomical-classification-challenge.zip
```

### 2. Set Up Python Environment

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/mallorn-astro-classification

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Baseline Benchmark

```bash
cd python/src
python benchmark.py --quick  # Fast test with 20% data
python benchmark.py          # Full benchmark
```

## Evolution Workflow

### Using /evolve Skill

The `/evolve-ml` skill can optimize:
1. **Feature extraction** in `features.py`
2. **Classifier parameters** in `classifier.py`
3. **Ensemble strategies**

### Evolution Targets

**Primary target**: `features.py::extract_evolved_features()`
- This function extracts features from light curves
- Evolve to discover TDE-specific patterns
- Key insight: TDEs have -5/3 power law decay

**Secondary target**: `classifier.py::EvolvedTDEClassifier`
- Hyperparameter optimization
- Threshold tuning
- Ensemble methods

### Fitness Function

Located in `benchmark.py::fitness()`:
```python
def fitness(...) -> float:
    """Returns F1 score (0-1, higher is better)"""
```

## Data Format

### Light Curves (split_XX/train_full_lightcurves.csv)
| Column | Description |
|--------|-------------|
| object_id | Unique object identifier |
| mjd | Modified Julian Date |
| flux | Flux in microjanskys |
| flux_err | Flux error |
| band | Filter (u, g, r, i, z, y) |

### Metadata
| Column | Train | Test |
|--------|-------|------|
| object_id | Yes | Yes |
| redshift | Yes (spectroscopic) | No |
| photo_z | No | Yes |
| ebv | Yes | Yes |
| target | Yes | No (predict this!) |

### Target Classes
- **TDE**: Tidal Disruption Event (positive class)
- **SN**: Nuclear Supernova
- **AGN**: Active Galactic Nucleus

## Key Files

```
mallorn-astro-classification/
├── CLAUDE.md                 # This file
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
├── .gitignore               # Excludes data/
├── data/                    # Competition data (NOT committed)
│   └── split_XX/            # 20 train/test splits
└── python/src/
    ├── features.py          # Feature extraction (EVOLVE THIS)
    ├── classifier.py        # Classification (EVOLVE THIS)
    ├── baselines.py         # Baseline comparisons
    ├── benchmark.py         # Evaluation harness
    ├── data_loader.py       # Data loading utilities
    └── submit.py            # Kaggle submission generator
```

## TDE Science Background

**What is a TDE?**
A Tidal Disruption Event occurs when a star passes too close to a supermassive black hole and gets torn apart by tidal forces. The stellar debris forms an accretion disk, producing a characteristic bright flare.

**TDE Light Curve Signatures:**
1. **Rapid rise** (days to weeks)
2. **Power-law decay** with exponent ~-5/3
3. **Blue colors** (high UV flux)
4. **Smooth evolution** (unlike SNe)
5. **No prior AGN activity**

**Key features to evolve:**
- Decay timescale fitting
- Color evolution (g-r, r-i over time)
- Rise time vs decay time ratio
- Variability smoothness metrics
- Pre-flare baseline comparison

## Submission Workflow

```bash
# Generate submission
python python/src/submit.py --model xgboost --output submission.csv

# Submit to Kaggle
kaggle competitions submit \
  -c mallorn-astronomical-classification-challenge \
  -f submission.csv \
  -m "Description of changes"
```

## Performance Tracking

Track evolution progress in mutations/ directory:
- `mutations/gen01_baseline.py` - Initial features
- `mutations/gen02_decay_fit.py` - Added decay fitting
- etc.

Update README.md with results after each significant improvement.

## Important Notes

1. **Data Privacy**: Competition data cannot be committed (see .gitignore)
2. **Class Imbalance**: TDEs are only ~0.6% of data - use class weights
3. **Multi-Split**: Data has 20 splits - can use for cross-validation
4. **F1 Focus**: Optimize for F1, not accuracy (imbalanced classes)
