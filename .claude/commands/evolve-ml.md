---
description: ML subskill for /evolve - optimizes model accuracy and performance
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, WebSearch, WebFetch, AskUserQuestion
argument-hint: <problem description>
---

# /evolve-ml - Machine Learning Evolution Subskill

This is the **ML optimization subskill** for the `/evolve` command. It evolves ML components for **classification metrics** (F1, accuracy, precision, recall, AUC).

**Note**: This subskill is invoked by the master `/evolve` skill when ML mode is detected. You can also invoke it directly with `/evolve-ml`.

---

## What We Evolve

Unlike `/evolve-perf` which optimizes runtime speed, `/evolve-ml` optimizes **prediction quality**:

| Component | Description | Example |
|-----------|-------------|---------|
| **Feature extraction** | Functions that transform raw data into features | Light curve statistics, decay fitting |
| **Hyperparameters** | Model configuration | learning_rate, max_depth, n_estimators |
| **Thresholds** | Decision boundaries | Classification threshold for binary |
| **Ensemble strategies** | How to combine models | Voting, stacking, blending |
| **Preprocessing** | Data transformations | Scaling, imputation strategies |

---

## Core Differences from /evolve-perf

| Aspect | /evolve-perf | /evolve-ml |
|--------|--------------|------------|
| Metric | ops/sec, latency | F1, accuracy, AUC |
| Direction | Lower is better (time) | Higher is better (score) |
| Evaluation | Benchmark execution | Cross-validation |
| Code focus | Algorithm optimization | Feature/model optimization |
| Determinism | Exact reproducibility | Seed-controlled randomness |

---

## Evaluation Contract (Hard Requirements)

### Three-Way Split (MANDATORY)

Every ML candidate MUST be evaluated with proper data splits:

1. **TRAIN**: Used for model fitting (typically 60-70% of labeled data)
2. **VALID**: Used for selection decisions (typically 15-20%)
3. **TEST/HOLDOUT**: Never used for selection; reported only for final analysis (15-20%)

For competitions with pre-defined splits (like MALLORN with 20 splits), use:
- Multiple splits for cross-validation
- Report mean ± std across splits

### Acceptance Criteria

A candidate is accepted only if ALL of the following hold:

1. **VALID improvement**: Candidate improves mean VALID metric by at least ε
   - Default ε = 0.005 (0.5% absolute improvement for F1)

2. **No overfitting signal**: TRAIN-VALID gap must not increase significantly
   - If TRAIN improves >10% but VALID regresses, flag as overfit

3. **Statistical significance** (for stochastic models):
   - Run with multiple seeds (default 3)
   - Require improvement > 1σ for acceptance

4. **Correctness**: Model must produce valid predictions (no NaN, correct format)

---

## Metrics Supported

### Classification (Binary)

| Metric | Formula | Use When |
|--------|---------|----------|
| **F1** | 2 × (P × R) / (P + R) | Imbalanced classes (default) |
| **Accuracy** | (TP + TN) / N | Balanced classes |
| **Precision** | TP / (TP + FP) | False positives costly |
| **Recall** | TP / (TP + FN) | False negatives costly |
| **AUC-ROC** | Area under ROC | Ranking quality |

### Classification (Multiclass)

| Metric | Averaging | Use When |
|--------|-----------|----------|
| **Macro F1** | Average per-class F1 | Class balance matters |
| **Weighted F1** | Class-size weighted | Prevalence matters |
| **Accuracy** | Overall correct | Simple baseline |

### Regression

| Metric | Formula | Use When |
|--------|---------|----------|
| **RMSE** | √(mean(error²)) | Large errors costly |
| **MAE** | mean(|error|) | Robust to outliers |
| **R²** | 1 - SS_res/SS_tot | Explained variance |

---

## Problem Analysis (Step 1)

Before evolution, analyze the ML problem:

### Data Characteristics

```python
analysis = {
    "task_type": "binary_classification",  # or multiclass, regression
    "n_samples": 3043,
    "n_features": 50,  # after feature extraction
    "class_distribution": {"TDE": 148, "non-TDE": 2895},
    "imbalance_ratio": 19.6,  # severe imbalance
    "missing_values": "10% in flux measurements",
    "data_splits": 20,  # pre-defined CV splits
}
```

### Algorithm Families

For classification, consider these families:

| Family | Examples | Strengths |
|--------|----------|-----------|
| **Tree ensembles** | XGBoost, LightGBM, RF | Handles imbalance, feature importance |
| **Linear** | LogReg, SVM | Fast, interpretable |
| **Neural** | MLP, LSTM | Complex patterns |
| **Naive Bayes** | GaussianNB | Fast baseline |

### Feature Strategies

For time-series classification (like MALLORN):

| Strategy | Description |
|----------|-------------|
| **Statistical** | Mean, std, min, max per band |
| **Temporal** | Rise time, decay rate, peak timing |
| **Physics-based** | Power-law fits, color evolution |
| **Engineered** | Domain-specific (TDE signatures) |

---

## Evolution Loop

### Generation 1: Baseline Exploration

Spawn N parallel agents (N = 8-16 for ML problems):

```python
gen1_strategies = [
    # Model variants
    "xgboost_default",
    "lightgbm_default",
    "random_forest_default",
    "logistic_regression",

    # Feature variants
    "statistical_features_only",
    "temporal_features_only",
    "physics_features_only",
    "all_features_combined",

    # Threshold variants
    "threshold_0.3",
    "threshold_0.5",
    "threshold_0.7",

    # Class weight variants
    "balanced_weights",
    "custom_weights_10x",
    "smote_oversampling",
]
```

### Generation 2+: Crossover + Mutation

**Crossover examples** (combining successful parents):

```python
crossover_ideas = [
    # Combine feature sets from two parents
    "parent_a_features + parent_b_model",

    # Combine hyperparameters
    "parent_a_tree_depth + parent_b_learning_rate",

    # Ensemble parents
    "voting_ensemble(parent_a, parent_b)",
    "stacking(parent_a, parent_b)",
]
```

**Mutation examples**:

```python
mutation_strategies = [
    "tune_threshold",           # Optimize decision threshold
    "add_feature",              # Add new engineered feature
    "remove_feature",           # Feature selection
    "tune_hyperparameter",      # Adjust one hyperparameter
    "change_class_weights",     # Adjust imbalance handling
    "change_preprocessing",     # Different scaling/imputation
]
```

---

## Directory Structure

```
.evolve/<problem>/
├── python/
│   ├── features.py          # Feature extraction (evolved)
│   ├── classifier.py        # Model definition (evolved)
│   ├── baselines.py         # Known algorithms
│   ├── benchmark.py         # Evaluation harness
│   └── evolved.py           # Current champion
├── data/
│   ├── train/               # Training data (if local)
│   ├── valid/               # Validation data
│   └── test/                # Holdout data
├── mutations/               # All tested mutations
│   ├── gen1_xgboost.py
│   ├── gen1_lightgbm.py
│   └── ...
├── evolution.json           # Full state
├── champion.json            # Best solution manifest
└── generations.jsonl        # Per-generation log
```

---

## Fitness Function Interface

The benchmark must expose this interface:

```python
def fitness(
    feature_extractor: Callable,
    classifier: Any,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    metric: str = "f1",
    n_seeds: int = 3
) -> Dict[str, float]:
    """
    Evaluate a candidate on the ML task.

    Returns:
        {
            "train_score": 0.85,
            "valid_score": 0.72,
            "train_valid_gap": 0.13,
            "std": 0.02,  # across seeds
            "threshold": 0.45,  # optimal threshold found
        }
    """
```

---

## Feature Evolution

### Feature Extraction Template

```python
def extract_evolved_features(light_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Evolved feature extraction function.

    This function is the primary evolution target.

    Args:
        light_curve: DataFrame with columns [mjd, band, flux, flux_err]

    Returns:
        Dictionary of feature_name -> value
    """
    features = {}

    # === BASELINE FEATURES ===
    # Per-band statistics
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        band_data = light_curve[light_curve['band'] == band]
        if len(band_data) >= 2:
            features[f'{band}_mean'] = band_data['flux'].mean()
            features[f'{band}_std'] = band_data['flux'].std()
            # ... more statistics

    # === EVOLVED FEATURES ===
    # (This section will be modified by evolution)

    # Example: Power-law decay fitting for TDE detection
    # TDEs have characteristic t^(-5/3) decay
    # ... evolved code here

    return features
```

### Feature Mutation Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **add_statistic** | Add new statistical feature | Skewness, kurtosis |
| **add_ratio** | Add ratio between features | g_mean / r_mean |
| **add_temporal** | Add time-based feature | Time to peak |
| **add_physics** | Add domain-specific feature | Power-law alpha |
| **remove_weak** | Remove low-importance feature | Based on RF importance |

---

## Hyperparameter Evolution

### Hyperparameter Chromosome

```python
hyperparameters = {
    # XGBoost example
    "n_estimators": 200,        # [50, 1000]
    "max_depth": 6,             # [3, 12]
    "learning_rate": 0.1,       # [0.01, 0.3]
    "subsample": 0.8,           # [0.5, 1.0]
    "colsample_bytree": 0.8,    # [0.5, 1.0]
    "scale_pos_weight": 10,     # [1, 50] for imbalance
    "min_child_weight": 1,      # [1, 10]
}
```

### Mutation Operators

```python
def mutate_hyperparameter(params: Dict, strategy: str) -> Dict:
    """Mutate one hyperparameter."""
    new_params = params.copy()

    if strategy == "increase_depth":
        new_params["max_depth"] = min(params["max_depth"] + 1, 12)

    elif strategy == "decrease_lr":
        new_params["learning_rate"] = params["learning_rate"] * 0.8

    elif strategy == "more_trees":
        new_params["n_estimators"] = int(params["n_estimators"] * 1.5)

    elif strategy == "increase_class_weight":
        new_params["scale_pos_weight"] = params["scale_pos_weight"] * 1.5

    return new_params
```

---

## Threshold Optimization

For binary classification, the threshold matters significantly:

```python
def optimize_threshold(y_true, y_proba, metric="f1"):
    """Find optimal classification threshold."""
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

**Key insight**: For imbalanced classes, the optimal threshold is often NOT 0.5.

---

## Class Imbalance Handling

### Strategies to Evolve

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **class_weight** | Weight minority class higher | Most cases |
| **SMOTE** | Synthetic minority oversampling | Medium imbalance |
| **undersampling** | Remove majority samples | Severe imbalance |
| **threshold_tuning** | Adjust decision boundary | All cases |
| **cost_sensitive** | Custom loss function | When costs known |

### Evolution Targets

```python
imbalance_config = {
    "method": "class_weight",  # or "smote", "undersample"
    "weight_ratio": 10,        # minority weight multiplier
    "threshold": 0.35,         # lowered for high recall
}
```

---

## Cross-Validation Strategy

### For Pre-defined Splits (MALLORN style)

```python
def evaluate_across_splits(
    candidate,
    data_dir: str,
    n_splits: int = 5  # Use 5 of 20 for speed
) -> Dict[str, float]:
    """Evaluate candidate across multiple pre-defined splits."""

    scores = []
    for split_num in range(1, n_splits + 1):
        train_lc, test_lc, train_meta, test_meta = load_single_split(
            data_dir, split_num
        )

        # Extract features
        X_train = extract_features(train_lc)
        X_test = extract_features(test_lc)
        y_train = train_meta['target']
        y_test = test_meta['target']

        # Fit and evaluate
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)

        scores.append(f1_score(y_test, y_pred))

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "scores": scores
    }
```

---

## Logging & Artifacts

### Per-Generation Log Entry

```json
{
  "generation": 3,
  "timestamp": "2024-01-15T10:30:00Z",
  "candidates": [
    {
      "id": "gen3_xgboost_tuned",
      "parent_ids": ["gen2_xgboost_default"],
      "mutation_type": "hyperparameter_tune",
      "changes": {"max_depth": "6 -> 8", "n_estimators": "200 -> 300"},
      "metrics": {
        "train_f1": 0.89,
        "valid_f1": 0.74,
        "test_f1": 0.72,
        "threshold": 0.38,
        "train_valid_gap": 0.15
      },
      "acceptance": {
        "result": "KEEP",
        "improvement": 0.02,
        "reason": "Valid F1 improved from 0.72 to 0.74"
      }
    }
  ],
  "champion_id": "gen3_xgboost_tuned",
  "best_valid_f1": 0.74
}
```

### Champion Manifest

```json
{
  "id": "gen5_ensemble_xgb_lgb",
  "generation": 5,
  "discovered_at": "2024-01-15T11:45:00Z",
  "code_path": "python/evolved.py",
  "metrics": {
    "train_f1": 0.85,
    "valid_f1": 0.76,
    "test_f1": 0.74,
    "threshold": 0.42
  },
  "components": {
    "features": "extract_evolved_features_v5",
    "model": "VotingClassifier(xgb, lgb)",
    "threshold": 0.42
  },
  "key_innovations": [
    "Power-law decay fitting for TDE detection",
    "Ensemble of XGBoost and LightGBM",
    "Optimized threshold at 0.42"
  ]
}
```

---

## Stopping Criteria

### Plateau Detection

```python
def is_plateau(history, patience=3, min_improvement=0.005):
    """Check if evolution has plateaued."""
    if len(history) < patience:
        return False

    recent = history[-patience:]
    best_recent = max(h['best_valid_f1'] for h in recent)
    best_before = max(h['best_valid_f1'] for h in history[:-patience])

    improvement = best_recent - best_before
    return improvement < min_improvement
```

### Early Stopping Triggers

1. **Budget exhausted**: Token/generation limit reached
2. **Plateau**: No improvement for 3+ generations
3. **Target achieved**: Reached specified F1/accuracy target
4. **Overfitting**: Train-valid gap exceeds threshold

---

## Usage Examples

### Basic Usage

```bash
# Optimize TDE classifier with default settings
/evolve-ml improve F1 score for TDE classification

# With budget
/evolve-ml improve MALLORN classifier --budget 50k

# Resume previous evolution
/evolve-ml --resume
```

### With Specific Targets

```bash
# Focus on feature evolution
/evolve-ml evolve features for light curve classification

# Focus on hyperparameter tuning
/evolve-ml tune XGBoost hyperparameters for imbalanced classification

# Focus on ensemble
/evolve-ml create ensemble classifier for MALLORN
```

---

## MALLORN-Specific Guidance

For the MALLORN TDE classification challenge:

### Key Insights

1. **Severe imbalance**: Only ~5% TDE - must handle carefully
2. **Time-series data**: Light curves require temporal features
3. **Physics signals**: TDEs have -5/3 power-law decay
4. **Multi-band**: 6 LSST bands provide color information
5. **20 splits**: Use for robust cross-validation

### Recommended Evolution Path

1. **Gen 1**: Establish baselines (XGBoost, LightGBM, RF with basic features)
2. **Gen 2-3**: Feature evolution (add physics-based features)
3. **Gen 4-5**: Hyperparameter tuning (focus on class weights, threshold)
4. **Gen 6+**: Ensemble strategies (combine best models)

### TDE-Specific Features to Evolve

```python
tde_features = [
    # Power-law decay (TDE signature)
    "decay_alpha",              # Should be close to -5/3
    "decay_timescale",          # Characteristic time

    # Color evolution
    "g_minus_r_peak",           # Color at peak
    "color_evolution_rate",     # How fast color changes

    # Light curve shape
    "rise_time",                # Time to peak
    "decay_time",               # Time from peak to half-max
    "asymmetry",                # rise_time / decay_time

    # Variability
    "smoothness",               # Residuals from fit
    "chi_squared",              # Fit quality
]
```

---

## Key Principles

1. **F1 focus**: For imbalanced problems, F1 beats accuracy
2. **Threshold matters**: Evolve the threshold, don't assume 0.5
3. **Features first**: Good features often beat model tuning
4. **Cross-validate**: Use all available splits for robust estimates
5. **Watch overfitting**: Track train-valid gap throughout
6. **Domain knowledge**: Physics-based features often win
7. **Ensemble late**: Combine models after individual optimization
