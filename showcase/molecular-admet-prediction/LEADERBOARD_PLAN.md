# Plan: Reaching #1 on TDC hERG Leaderboard

**Current Position**: #4 (0.874 ± 0.008 AUROC)
**Target**: #1 (MapLight + GNN: 0.880 AUROC)
**Gap**: ~0.006 AUROC (not statistically significant, but we want to beat it clearly)

## Validation Protocol (CRITICAL)

All experiments MUST be validated using:
1. **Official TDC benchmark**: 5 seeds, scaffold split
2. **External ChEMBL validation**: Within-domain test
3. **No cherry-picking**: Report mean ± std across all seeds

```bash
# Validation command
./venv/bin/python external_validation.py <mutation.py> --full
```

## Execution Plan (Ordered by Expected Impact)

### Phase 1: GNN Ensemble Component ⬅️ START HERE
**Expected Impact**: +0.5-1.5% AUROC
**Rationale**: Top 3 models (MapLight, AttentiveFP, AttrMasking) all use GNNs

Implementation:
- [ ] Add PyTorch Geometric / DGL dependency
- [ ] Implement simple GCN or GAT model
- [ ] Create gen22_gnn_ensemble.py
- [ ] Benchmark on TDC (5-seed)
- [ ] If improved, tune ensemble weights

### Phase 2: Pre-trained Molecular Transformers
**Expected Impact**: +0.3-1.0% AUROC
**Rationale**: Transfer learning from large chemical datasets

Implementation:
- [ ] Test ChemBERTa embeddings (HuggingFace)
- [ ] Test MolBERT or Grover if available
- [ ] Add as features to ensemble
- [ ] Benchmark on TDC

### Phase 3: ChEMBL Pre-training
**Expected Impact**: +0.2-0.8% AUROC
**Rationale**: More diverse training data

Implementation:
- [ ] Download full ChEMBL hERG dataset (~300k)
- [ ] Pre-train models on ChEMBL
- [ ] Fine-tune on TDC training set
- [ ] Benchmark on TDC

### Phase 4: hERG-Specific Features
**Expected Impact**: +0.1-0.5% AUROC
**Rationale**: Domain knowledge about ion channel binding

Implementation:
- [ ] Add cationic nitrogen count at pH 7.4
- [ ] Add aromatic surface area
- [ ] Add hydrophobic moment descriptors
- [ ] Benchmark on TDC

### Phase 5: Hyperparameter Optimization
**Expected Impact**: +0.1-0.3% AUROC
**Rationale**: Systematic search vs manual tuning

Implementation:
- [ ] Set up Optuna study
- [ ] Optimize ensemble weights
- [ ] Optimize individual model hyperparameters
- [ ] Benchmark on TDC

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| TDC AUROC | 0.874 | 0.880 | 0.890 |
| TDC Rank | #4 | #1 | #1 |
| ChEMBL Within-Domain | 0.809 | 0.820 | 0.850 |

## Anti-Overfitting Measures

1. **Never tune on test set** - Only use validation for model selection
2. **Report all experiments** - No hiding failed attempts
3. **Cross-validate everything** - 5-fold CV on training data
4. **External validation required** - ChEMBL must also improve
