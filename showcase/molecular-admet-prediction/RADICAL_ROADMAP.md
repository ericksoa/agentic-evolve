# Radical Roadmap: Reaching #1 on TDC hERG

**Current Status:** 0.869 ± 0.005 AUROC (Rank #5)
**Target:** 0.880 ± 0.002 AUROC (MapLight+GNN, Rank #1)
**Gap:** 1.1% (0.011 AUROC)

## What We've Tried (Gen1-33)
- ✅ Ensemble tuning (RF, XGB, ET, SVM)
- ✅ GNN hybrids (GCN, GAT, AttentiveFP) - high variance
- ✅ ChemBERTa transformer embeddings - no improvement
- ✅ ChEMBL pre-training - **invalidated by 93% data leakage**
- ✅ CatBoost + Avalon fingerprints - no improvement
- ✅ SMILES augmentation - **best approach, 0.869**

## Why The Gap Exists

MapLight+GNN uses **pre-trained GIN fingerprints** from molfeat that:
1. Are trained on millions of molecules with supervised masking
2. Capture transferable molecular representations
3. Provide features unavailable through fixed fingerprints

Our classical approach hits a ceiling because Morgan/MACCS fingerprints
can only capture predefined structural patterns.

---

## RADICAL IDEAS FOR FUTURE WORK

### 1. 🧬 Train Custom Pre-trained GNN (HIGH IMPACT)

**Idea:** Train our own GIN/GCN on large unlabeled molecular data using
self-supervised objectives, then use as feature extractor.

**Steps:**
1. Download PubChem/ZINC (100M+ molecules)
2. Train GNN with:
   - Masked atom prediction
   - Contrastive learning between augmented views
   - Node-level and graph-level tasks
3. Extract embeddings for TDC molecules
4. Combine with our fingerprint ensemble

**Why it might work:** This is essentially what MapLight does. We'd be
creating our own version without their proprietary pre-training.

**Estimated effort:** 2-4 GPU days for pre-training

---

### 2. 🔬 Protein-Ligand Interaction Features (NOVEL)

**Idea:** Use hERG protein structure (PDB: 5VA1) to compute
interaction-based features.

**Steps:**
1. Dock all TDC molecules to hERG binding site
2. Extract interaction fingerprints (PLIP, IFP)
3. Add binding pose features (RMSD to known blockers)
4. Include binding affinity estimates

**Why it might work:** hERG blocking is fundamentally about binding.
Structure-based features capture the actual mechanism.

**Challenges:** Docking is slow (~minutes per molecule), conformational
flexibility, binding site uncertainty.

---

### 3. 🎯 Active Learning on Test Distribution (RADICAL)

**Idea:** The TDC test set has a different scaffold distribution than
training. Use uncertainty-guided pseudo-labeling.

**Steps:**
1. Identify high-uncertainty predictions on test scaffolds
2. Find similar molecules in ChEMBL (excluding TDC overlap!)
3. Add confident pseudo-labels to training
4. Iterate with improved model

**Why it might work:** Closes the scaffold generalization gap.

**Risk:** Need very careful overlap checking to avoid data leakage.

---

### 4. 🧪 Quantum Mechanical Descriptors (EXPERIMENTAL)

**Idea:** Add computed QM properties that correlate with hERG binding.

**Features to compute:**
- HOMO/LUMO energies and gap
- Partial charges (ESP-derived)
- Molecular polarizability
- Dipole moment
- Electron density at key positions

**Why it might work:** QM features capture electronic properties
that influence ion channel binding.

**Challenges:** Expensive computation (~5-10 min per molecule with DFT).

---

### 5. 🔄 Test-Time Adaptation (META-LEARNING)

**Idea:** Fine-tune model on test batch using pseudo-labels before
final prediction.

**Steps:**
1. For each test batch, generate pseudo-labels with current model
2. Fine-tune ensemble on high-confidence predictions
3. Re-predict with adapted model

**Why it might work:** Adapts to test distribution shift.

**Risk:** Could overfit to incorrect pseudo-labels.

---

### 6. 📊 Scaffold-Aware Ensemble (TARGETED)

**Idea:** Train separate specialists for different scaffold families,
combine predictions based on scaffold similarity.

**Steps:**
1. Cluster training molecules by scaffold
2. Train specialized model for each cluster
3. At inference, weight specialists by scaffold similarity
4. Combine with global model

**Why it might work:** Different scaffolds may have different SAR.

---

### 7. 🌐 External Data Fusion (CAREFUL)

**Idea:** Incorporate additional hERG data sources with proper
deduplication.

**Data sources:**
- ChEMBL hERG (with STRICT overlap removal)
- BindingDB hERG assays
- PubChem BioAssay hERG screens
- Patent literature extractions

**Critical:** Must verify NO overlap with TDC test molecules!

---

## PRIORITY RANKING

| Priority | Approach | Expected Gain | Effort | Risk |
|----------|----------|---------------|--------|------|
| 1 | Custom Pre-trained GNN | +0.01-0.02 | High | Medium |
| 2 | Scaffold-Aware Ensemble | +0.005-0.01 | Medium | Low |
| 3 | External Data Fusion | +0.005-0.01 | Medium | High (leakage) |
| 4 | Protein-Ligand Features | +0.005-0.015 | High | Medium |
| 5 | Active Learning | +0.003-0.008 | Medium | High |
| 6 | QM Descriptors | +0.002-0.005 | Very High | Low |
| 7 | Test-Time Adaptation | +0.001-0.005 | Low | High |

---

## RECOMMENDED NEXT SESSION

**Start with:** Custom Pre-trained GNN (Approach #1)

1. Set up GNN pre-training pipeline on lightning.ai
2. Use ZINC-250K for initial experiments
3. Train with masked atom prediction
4. Extract embeddings and combine with gen32
5. Evaluate on TDC benchmark

This has the highest probability of closing the gap because it directly
addresses what makes MapLight+GNN successful.

---

## METRICS TO TRACK

- AUROC (primary)
- Variance across seeds (stability)
- Performance by scaffold family
- Training/inference time
- Model complexity

---

*Last updated: 2026-01-14*
*Current champion: gen32_aug_predict_avg.py (0.869 ± 0.005)*
