# Evolution Improvement Plan

## Current State

| Approach | ROC-AUC | Gap to Target |
|----------|---------|---------------|
| Manual ensemble (evolved_ensemble.py) | 0.8711 | -2.9% to 0.90 |
| Auto-evolved best (gen0_b.py) | 0.8585 | -4.2% to 0.90 |
| Target | 0.90 | - |

**Key insight**: Manual design found fingerprint+descriptor hybrid. Auto-evolution didn't discover this - it tried MACCS+pharmacophore+MLP instead.

---

## Phase 1: Seed Evolution with Best Known Solution

**Goal**: Start from 0.8711 instead of 0.84, give evolution a head start.

```bash
# Add evolved_ensemble.py as primary starter (already in config)
# Run evolution with it as the seed
```

**Changes needed**:
- Update `starter_solutions` to prioritize `evolved_ensemble.py`
- Add hint in config: "The hybrid FP+descriptor approach works well"

---

## Phase 2: Expand Feature Search Space

**Goal**: Help evolution discover feature combinations it missed.

**New optimization strategies to add**:
1. "Combine Morgan FP (2048 bits) with RDKit descriptors in same model"
2. "Try different FP radii: radius=1 for local, radius=3 for extended"
3. "Add MACCS keys (167 bits) to existing fingerprint features"
4. "Experiment with feature concatenation order and scaling"

**New starter solutions**:
- `hybrid_fp_maccs.py` - Morgan + MACCS fingerprints
- `hybrid_fp_desc_light.py` - FP + only top-10 descriptors
- `triple_ensemble.py` - RF + GBM + SVM (different from current RF+GBM+ET)

---

## Phase 3: Multi-Objective Evolution

**Goal**: Optimize for both discrimination AND calibration.

**Current fitness**: `roc_auc` only

**Proposed fitness**:
```python
fitness = 0.7 * roc_auc + 0.3 * (1 - ece)  # Penalize miscalibration
```

Or Pareto frontier approach:
- Track both metrics
- Keep solutions that are best on either dimension
- Final selection balances both

**Changes needed**:
- Update `evaluate.py` to return composite fitness
- Add `--multi-objective` flag to evolution

---

## Phase 4: External Validation Gate

**Goal**: Prevent overfitting to TDC benchmark.

**Data sources**:
1. ChEMBL hERG bioactivity data (~2000 compounds)
2. PubChem hERG assay data
3. Literature curated sets (Czodrowski 2013)

**Implementation**:
```python
# In evaluation:
tdc_auc = evaluate_on_tdc(model)
chembl_auc = evaluate_on_chembl(model)  # External

# Only accept if both are good
valid = tdc_auc > 0.80 and chembl_auc > 0.75
fitness = (tdc_auc + chembl_auc) / 2
```

---

## Phase 5: Ensemble of Evolved Solutions

**Goal**: Combine diverse evolved solutions for robustness.

**Approach**:
1. Run evolution 5x with different random seeds
2. Keep top solution from each run
3. Create meta-ensemble of all 5
4. Weight by validation performance

**Expected benefit**: Reduce variance, improve generalization.

---

## Phase 6: Neural Architecture Search (if needed)

**Goal**: If tree ensembles plateau, try neural approaches.

**Options**:
1. Graph Neural Networks (if torch-geometric available)
2. Transformer on SMILES strings
3. Message Passing Neural Network

**Constraint**: Keep inference < 100ms (may need distillation)

---

## Implementation Priority

| Phase | Effort | Expected Gain | Priority |
|-------|--------|---------------|----------|
| 1. Seed with best | Low | +0.5-1% | **HIGH** |
| 2. Feature search | Medium | +1-2% | **HIGH** |
| 3. Multi-objective | Medium | +0.5% (calibration) | MEDIUM |
| 4. External validation | High | Robustness | MEDIUM |
| 5. Meta-ensemble | Low | +0.3-0.5% | LOW |
| 6. Neural search | High | Unknown | LOW |

---

## Quick Start: Phase 1+2 Combined

```bash
# 1. Create hybrid starter solutions
# 2. Update config with new strategies
# 3. Run evolution with larger population

cd showcase/molecular-admet-prediction
PYTHONPATH=../../sdk python3 -m evolve_sdk \
    --config evolve_config.json \
    --max-generations 10 \
    --population-size 6 \
    --plateau 4 \
    --no-parallel
```

---

## Success Criteria

- [ ] Auto-evolved solution beats 0.8711 (manual best)
- [ ] Reach 0.88+ ROC-AUC on TDC test set
- [ ] ECE < 0.10 (improved calibration)
- [ ] Validated on external dataset (ChEMBL)
- [ ] Inference still < 100ms
