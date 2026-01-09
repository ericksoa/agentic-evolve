# OpenML AutoML Benchmark - Claude Workflow

## Goal

Demonstrate that `evolve-sdk` (with ML mode) can match or beat established AutoML tools (Auto-sklearn, TPOT, FLAML) on the OpenML-CC18 benchmark suite of 72 classification datasets.

## Why This Matters

- **Direct comparison**: OpenML-CC18 is THE standard benchmark for AutoML systems
- **Published baselines**: Auto-sklearn, TPOT, FLAML all have published scores
- **Reproducible**: Standardized 10-fold CV with pre-defined splits
- **Credibility**: "Beats Auto-sklearn on X/72 datasets" is a headline result

## Quick Start

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/openml-automl-benchmark

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Also install evolve-sdk
pip install -e ../../sdk/

# Run pilot test (5 small datasets) - baselines only
python python/src/run_benchmark.py --pilot

# Run evolution on a single dataset
python python/src/run_benchmark.py --dataset-id 31 --evolve

# Run full benchmark with evolution (warning: many hours)
python python/src/run_benchmark.py --suite cc18 --evolve
```

## Using evolve-sdk Directly

For more control, run evolve-sdk directly with a config:

```bash
# Generate config for a dataset
python python/src/run_benchmark.py --dataset-id 31

# Run evolve-sdk with the generated config
python -m evolve_sdk --config evolve_config_31.json --mode=ml --max-generations=15

# Resume interrupted evolution
python -m evolve_sdk --resume
```

## Cloud Execution (Recommended for Full Suite)

For heavy parallelism, use Lightning AI:

```bash
# Configure lightning.ai credentials
lightning login

# Run benchmark on cloud (much faster)
python python/src/run_benchmark.py --suite cc18 --evolve --cloud
```

## Benchmark Strategy

### Phase 1: Pilot (5 datasets)
Test on 5 small, representative datasets to validate the approach:
1. `credit-g` (1000 samples, 20 features, binary) - classic imbalanced
2. `diabetes` (768 samples, 8 features, binary) - small/simple
3. `vehicle` (846 samples, 18 features, 4-class) - multiclass
4. `segment` (2310 samples, 19 features, 7-class) - multiclass
5. `kc1` (2109 samples, 21 features, binary) - software defect

### Phase 2: Full Suite
Run on all 72 OpenML-CC18 datasets with:
- 10-fold CV (using OpenML's pre-defined splits)
- 1-hour time budget per dataset (matching AMLB protocol)
- Compare against published Auto-sklearn/TPOT/FLAML scores

## Evaluation Protocol

Following the official AMLB (AutoML Benchmark) protocol:

1. **Metric**: Accuracy for balanced, macro-F1 for imbalanced (>3:1 ratio)
2. **CV**: 10-fold using OpenML task splits
3. **Time budget**: 1 hour per dataset (optional: 10min for quick runs)
4. **Seeds**: 3 random seeds, report mean +/- std

## Directory Structure

```
openml-automl-benchmark/
├── CLAUDE.md                 # This file
├── RESULTS.md               # Benchmark results and analysis
├── requirements.txt          # Dependencies
├── python/src/
│   ├── openml_loader.py     # OpenML dataset loading
│   ├── baseline_runner.py   # Default sklearn baselines
│   ├── evolve_runner.py     # evolve-ml integration
│   ├── run_benchmark.py     # Main benchmark script
│   └── compare_automl.py    # Compare with published results
└── results/
    ├── baselines/           # Baseline model results
    ├── evolved/             # evolve-ml results
    └── comparison/          # Side-by-side comparisons
```

## Key Files

### openml_loader.py
- Loads datasets from OpenML API
- Handles preprocessing (missing values, categorical encoding)
- Returns standardized train/test splits

### baseline_runner.py
- Runs default sklearn models (LogReg, RF, XGB, LightGBM)
- Provides baseline scores to compare against

### evolve_runner.py
- Integrates with evolve-ml for each dataset
- Runs evolution with proper holdout validation
- Saves evolved solutions

## Comparison Targets

Published AMLB results (1-hour budget):

| Framework | Mean Rank | Win Rate |
|-----------|-----------|----------|
| AutoGluon | 1.8 | 45% |
| Auto-sklearn | 2.3 | 25% |
| FLAML | 2.9 | 15% |
| TPOT | 3.5 | 10% |

Our target: **Beat Auto-sklearn on 40+ datasets** (>55% win rate)

## Critical Learnings from MALLORN

Apply these lessons to OpenML:

1. **Holdout validation**: Always reserve 1-2 folds for overfitting detection
2. **Simpler is better on small data**: Many OpenML datasets are <1000 samples
3. **Fixed thresholds**: Don't over-optimize, use default or fixed thresholds
4. **Track CV-holdout gap**: Reject candidates that overfit

## Evolution Strategy per Dataset

For each dataset:

1. **Gen 0**: Run baselines (LogReg, RF, XGB, LightGBM) with defaults
2. **Gen 1-2**: Hyperparameter evolution on best baseline
3. **Gen 3-4**: Feature preprocessing evolution (scaling, encoding)
4. **Gen 5+**: Ensemble strategies if time permits

## Resource Limits

Per CLAUDE.md global rules:
- Max 4 parallel processes locally
- For heavy runs, use cloud (lightning.ai)
- Sequential is fine for individual datasets

## Success Criteria

1. **Match Auto-sklearn**: Win rate >= 50% on OpenML-CC18
2. **Beat on small data**: Win rate >= 70% on datasets <1000 samples
3. **Publishable**: Clean results for blog post/paper

## References

- OpenML-CC18 Suite: https://www.openml.org/s/99
- AMLB Paper: https://arxiv.org/abs/2207.12560
- AMLB GitHub: https://github.com/openml/automlbenchmark
