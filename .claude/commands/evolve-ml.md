---
description: ML subskill for /evolve - optimizes model accuracy and performance
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite
argument-hint: <problem description>
---

# /evolve-ml - Machine Learning Evolution Subskill

This is the **ML optimization subskill** for the `/evolve` command. It evolves ML components for **classification metrics** (F1, accuracy, precision, recall, AUC).

**Implementation**: Uses the evolve-sdk Python package for hierarchical agent orchestration.

---

## Usage

```bash
/evolve-ml <problem description>
/evolve-ml <problem description> --budget <tokens|generations>
/evolve-ml --resume
```

**Examples**:
```
/evolve-ml improve F1 score for TDE classification
/evolve-ml tune XGBoost hyperparameters for imbalanced classification
/evolve-ml evolve features for light curve classification
/evolve-ml --resume
```

---

## Execution Instructions

When this skill is invoked:

### Step 1: Verify SDK Installation

```bash
python3 -c "import evolve_sdk" 2>/dev/null && echo "SDK ready" || echo "SDK missing"
```

If missing, inform user:
```
The evolve-sdk package is not installed. Install with:
  cd <repo-root>/sdk && pip install -e .
```

### Step 2: Check for evolve_config.json

Look for a config file in the current directory:
```bash
ls evolve_config.json 2>/dev/null
```

If found, use `--config` flag. If not, pass problem description directly.

### Step 3: Run Evolution

```bash
# With config file:
python3 -m evolve_sdk --config=evolve_config.json --mode=ml

# Without config file:
python3 -m evolve_sdk "<problem description>" --mode=ml

# Resume previous:
python3 -m evolve_sdk --resume --mode=ml
```

**Available flags**:
| Flag | Default | Description |
|------|---------|-------------|
| `--max-generations=N` | 50 | Maximum generations |
| `--population-size=N` | 10 | Population size |
| `--plateau=N` | 5 | Stop after N gens without improvement |
| `--no-parallel` | false | Run mutations sequentially |
| `--model=X` | claude-opus-4-6 | Model for subagents |

### Step 4: Report Results

After completion:
1. Read `.evolve-sdk/<problem>/champion.json`
2. Summarize: generations, final F1/accuracy, champion path
3. Show key innovations and metric progression

---

## Mode-Specific Guidance

The following sections define **ML mode requirements** that the SDK's subagents follow.

---

## What We Evolve

| Component | Description | Example |
|-----------|-------------|---------|
| **Feature extraction** | Functions that transform raw data into features | Light curve statistics, decay fitting |
| **Hyperparameters** | Model configuration | learning_rate, max_depth, n_estimators |
| **Thresholds** | Decision boundaries | Classification threshold for binary |
| **Ensemble strategies** | How to combine models | Voting, stacking, blending |
| **Preprocessing** | Data transformations | Scaling, imputation strategies |

---

## Evaluation Contract (Hard Requirements)

### Three-Way Split (MANDATORY)

Every ML candidate MUST be evaluated with proper data splits:

1. **TRAIN**: Used for model fitting (typically 60-70% of labeled data)
2. **VALID**: Used for selection decisions (typically 15-20%)
3. **TEST/HOLDOUT**: Never used for selection; reported only for final analysis (15-20%)

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

---

## Mutation Strategies (ML Mode)

For ML optimization, prioritize these mutation strategies:

1. **Feature addition**: Add new engineered features
2. **Feature removal**: Feature selection based on importance
3. **Hyperparameter tuning**: Adjust model parameters
4. **Threshold optimization**: Tune decision boundary
5. **Class weight adjustment**: Handle imbalance differently
6. **Model swap**: Try different model family
7. **Ensemble creation**: Combine multiple models
8. **Preprocessing change**: Different scaling/imputation

---

## Crossover Guidelines

When combining two parent solutions:

1. **Feature combination**: Merge feature sets from both parents
2. **Hyperparameter blending**: Average or select best hyperparameters
3. **Ensemble parents**: Create voting/stacking ensemble
4. **Best component selection**: Use best model from A, best features from B

---

## CRITICAL: Overfitting Detection

**This is the most important section.** Higher CV scores do NOT guarantee better generalization.

### Mandatory Holdout Validation

**BEFORE accepting any improvement**, validate on a true holdout:

```python
def validate_with_holdout(candidate, cv_score, holdout_data):
    holdout_score = candidate.evaluate(holdout_data)
    gap = cv_score - holdout_score

    if gap > 0.10:  # 10% gap threshold
        print(f"OVERFITTING DETECTED!")
        print(f"   CV: {cv_score:.3f}, Holdout: {holdout_score:.3f}")
        return False, "overfit"

    return True, "accepted"
```

### What Causes Overfitting

1. **Aggressive feature selection** - Use 50+ features on tiny data
2. **Complex models** - XGBoost often safer than LightGBM
3. **Optimizing weights on training data** - Use fixed weights
4. **Threshold optimization on training data** - Use fixed thresholds

---

## Class Imbalance Handling

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **class_weight** | Weight minority class higher | Most cases |
| **SMOTE** | Synthetic minority oversampling | Medium imbalance |
| **threshold_tuning** | Adjust decision boundary | All cases |

---

## Logging & Artifacts

All evolution artifacts are stored in `.evolve-sdk/<problem>/`:

```
.evolve-sdk/<problem>/
├── evolution.json       # Full state (mode, population, history)
├── champion.json        # Best solution manifest
├── generations.jsonl    # Per-generation log (append-only)
├── mutations/           # All tested mutations
└── python/              # ML code (features.py, classifier.py)
```

---

## Key Principles

1. **Holdout FIRST**: Always validate on holdout before accepting CV improvements
2. **F1 focus**: For imbalanced problems, F1 beats accuracy
3. **Threshold matters**: Evolve the threshold, don't assume 0.5
4. **Features first**: Good features often beat model tuning
5. **Watch overfitting**: Track CV-holdout gap throughout (MOST IMPORTANT)
6. **Domain knowledge**: Physics-based features often win
7. **Simpler is better**: On tiny data, prefer simpler models
8. **Fixed > Optimized**: Fixed weights/thresholds generalize better
