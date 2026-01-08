# MALLORN Astronomical Classification - Claude Workflow

## ⚠️ CRITICAL: Save Discoveries Immediately

**MANDATORY WORKFLOW RULE**: When you make ANY discovery, insight, or observation during experimentation:

1. **IMMEDIATELY** append it to `DISCOVERIES.md` (create if doesn't exist)
2. Include: timestamp, what you tried, what you found, why it matters
3. Do this BEFORE moving on to the next experiment
4. Do this EVEN IF the discovery seems minor

**Format for DISCOVERIES.md:**
```markdown
## [Date] Discovery: [Short Title]

**What I tried:** [Experiment description]
**What I found:** [Results/observations]
**Why it matters:** [Implications for next steps]
**Action:** [What to do with this knowledge]
```

**Why this matters:** Context can be lost at any time (crashes, session limits, reboots). Undocumented discoveries are LOST discoveries. The 30 seconds to write it down saves hours of re-discovery.

**Current discoveries file:** `DISCOVERIES.md` (check this first when resuming work!)

---

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
├── CLAUDE.md                 # This file (workflow instructions)
├── EVOLUTION_LOG.md          # Learnings from each generation (READ THIS!)
├── README.md                 # Project overview and results
├── requirements.txt          # Dependencies
├── .gitignore               # Excludes data/
├── data/                    # Competition data (NOT committed)
│   └── split_XX/            # 20 train/test splits
└── python/src/
    ├── features.py          # Feature extraction (EVOLVED - Gen 2)
    ├── classifier.py        # Ensemble classifier (EVOLVED - Gen 4)
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

## Evolution Results (Current Best Public: F1 = 0.4154, Pending: 0.50)

| Gen | Strategy | CV F1 | Holdout F1 | Public F1 | Status |
|-----|----------|-------|------------|-----------|--------|
| 4 | LR+XGB ensemble | 0.415 | ~0.41 | **0.4154** | Best Public |
| 5-6 | Feature sel + LGB | 0.55-0.58 | ~0.32 | 0.32 | OVERFIT |
| 9 | Pure LogReg (t=0.35) | 0.227 | 0.4611 | Pending | +11% |
| **10** | **Pure LogReg (t=0.40)** | 0.235 | **0.5025** | Pending | **+21%** |

**CRITICAL**: Gen 5-6 showed that higher CV score != better public score. Gen 9-10 focus on holdout validation.

**Read `EVOLUTION_LOG.md` for detailed learnings from each generation.**

## Critical Learnings for This Problem

1. **Use Logistic Regression as baseline** - Tree methods fail on <50 positive examples
2. **Physics features are essential** - Power-law decay (α ≈ -5/3) is the TDE signature
3. **Tune the threshold** - Optimal is ~0.35, not 0.5
4. **Ensemble LR + XGB** - Combines stability with power
5. **Expect high variance** - Only 3-12 TDEs per split

## Anti-Patterns (Don't Do These)

- Don't use Random Forest on this data (fails completely)
- Don't optimize threshold on training data (overfits)
- Don't trust single-fold results (variance is huge)
- Don't add model complexity without validation

## Continuing Evolution

If resuming evolution, read `EVOLUTION_LOG.md` first, then try:
- Gen 5: Add LightGBM to ensemble
- Gen 6: Feature selection (remove noisy features)
- Gen 7: Calibrated probabilities
- Gen 8: Semi-supervised learning

## Important Notes

1. **Data Privacy**: Competition data cannot be committed (see .gitignore)
2. **Class Imbalance**: TDEs are only ~5% of training data - use class weights
3. **Multi-Split**: Data has 20 splits - use for cross-validation
4. **F1 Focus**: Optimize for F1, not accuracy (imbalanced classes)
5. **Small Data**: Only 100-170 training objects per split with 3-12 TDEs
