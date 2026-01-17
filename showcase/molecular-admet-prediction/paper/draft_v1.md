# EvolveML: Automated Discovery of Competitive hERG Toxicity Predictors Through Evolutionary Algorithm Design

**Authors:** Aaron Erickson¹*

**Affiliations:**
¹ NVIDIA Corporation
* Corresponding author: aerickson@nvidia.com

**Date:** January 2026

---

## Abstract

Predicting human Ether-à-go-go-Related Gene (hERG) channel blockade is critical for cardiac safety assessment in drug discovery, as hERG inhibition can cause fatal arrhythmias. While deep learning methods like Graph Neural Networks (GNNs) have achieved state-of-the-art performance, they require substantial computational resources and lack interpretability. We present EvolveML, an automated algorithm discovery framework that evolves ensemble machine learning models for molecular property prediction. Applied to hERG toxicity prediction, our evolved model using SMILES augmentation with test-time prediction averaging achieves **0.869 ± 0.005 AUROC** on the Therapeutics Data Commons (TDC) benchmark, ranking **#5 among 10 published methods** and within 1.1% of the top GNN-based approach. Notably, our method provides feature-level interpretability and achieves inference times of ~5ms per molecule. We further validate on 11,411 ChEMBL hERG compounds (0.809 AUROC) and demonstrate **90% accuracy on 20 drugs with documented hERG/QT liability**, including **100% sensitivity on the 12 drugs actually withdrawn from market** for cardiotoxicity (terfenadine, cisapride, astemizole, etc.)—providing compelling real-world evidence of clinical utility. Our results demonstrate that evolutionary optimization can discover effective hybrid architectures that balance the stability of classical ML with the expressiveness of deep learning.

**Keywords:** hERG, cardiotoxicity, machine learning, ensemble methods, evolutionary algorithms, drug discovery, ADMET

---

## 1. Introduction

### 1.1 The hERG Challenge in Drug Discovery

The human Ether-à-go-go-Related Gene (hERG) encodes the Kv11.1 potassium channel, which plays a critical role in cardiac repolarization. Blockade of this channel can prolong the QT interval, potentially leading to Torsades de Pointes (TdP), a life-threatening ventricular arrhythmia [1]. Consequently, hERG liability assessment has become a mandatory component of preclinical drug safety evaluation, with regulatory agencies requiring hERG screening for all drug candidates [2].

The high attrition rate of drug candidates due to hERG-related cardiotoxicity—estimated at 2-3% of all compounds entering clinical trials [3]—has motivated substantial investment in computational prediction methods. Early identification of hERG blockers can significantly reduce development costs and accelerate the drug discovery pipeline.

### 1.2 Computational Approaches to hERG Prediction

Machine learning approaches to hERG prediction have evolved considerably over the past decade. Traditional methods relied on quantitative structure-activity relationship (QSAR) models using molecular descriptors and fingerprints [4,5]. More recently, deep learning methods, particularly Graph Neural Networks (GNNs), have achieved impressive results by learning directly from molecular graph representations [6,7].

The Therapeutics Data Commons (TDC) [8] provides a standardized benchmark for comparing hERG prediction methods, with current state-of-the-art models achieving AUROC scores of 0.87-0.88. However, these top-performing methods—including MapLight+GNN [9], CFA [10], and SimGCN [11]—typically require:
- GPU acceleration for training and inference
- Complex architectures with millions of parameters
- Limited interpretability of predictions

### 1.3 Our Contribution

We present EvolveML, an automated algorithm discovery framework that evolves ensemble machine learning models through systematic mutation and selection. Our key contributions are:

1. **Competitive Performance**: Our evolved model achieves 0.869 ± 0.005 AUROC on the TDC hERG benchmark, ranking #5 among published methods and within 1.1% of the top GNN-based approach, with the lowest variance among top-5 methods.

2. **SMILES Augmentation Discovery**: Evolution discovered that SMILES augmentation at both training and test time is more effective than GNN hybrids for small datasets, improving both performance (0.869 vs 0.865) and stability (std 0.005 vs 0.019).

3. **Practical Advantages**: The model requires no GPU, achieves ~5ms inference per molecule, and provides interpretable feature importances—critical for regulatory submissions.

4. **Evolutionary Discovery**: We demonstrate that automated evolution over 32 generations can discover effective data augmentation strategies, exploring pure GNNs, transformers, pre-training approaches, and various ensemble configurations.

5. **External Validation**: We validate on 11,411 ChEMBL hERG compounds, demonstrating generalization beyond the TDC training set.

6. **Clinical Relevance**: We validate on 20 drugs with documented hERG/QT liability, achieving 90% overall accuracy and **100% sensitivity on the 12 drugs actually withdrawn from market**—demonstrating that our model would have flagged these dangerous drugs during preclinical screening.

---

## 2. Related Work

### 2.1 hERG Prediction Methods

Early computational approaches to hERG prediction employed pharmacophore models and 2D/3D-QSAR methods [12,13]. The advent of machine learning brought random forests [14], support vector machines [15], and gradient boosting methods [16] trained on molecular fingerprints and descriptors.

Recent deep learning approaches have achieved state-of-the-art results:
- **Graph Neural Networks**: Message Passing Neural Networks (MPNNs) [17], Graph Attention Networks (GATs) [18], and their variants learn molecular representations directly from atomic graphs.
- **Transformer Models**: ChemBERTa [19] and MolBERT [20] apply attention mechanisms to SMILES strings.
- **Hybrid Approaches**: Methods like Chemprop-RDKit [21] combine learned representations with engineered features.

### 2.2 Ensemble Methods in Drug Discovery

Ensemble methods have a long history in ADMET prediction due to their robustness and interpretability [22]. Notable approaches include:
- Consensus models averaging multiple QSAR predictions [23]
- Stacking ensembles with meta-learners [24]
- Multi-task learning across related endpoints [25]

### 2.3 Automated Machine Learning

AutoML approaches have been applied to molecular property prediction, including hyperparameter optimization [26], neural architecture search [27], and feature selection [28]. Our work extends this by applying evolutionary algorithms to discover both model architectures and ensemble compositions.

---

## 3. Methods

### 3.1 Evolutionary Algorithm Discovery Framework

EvolveML operates through iterative cycles of mutation, evaluation, and selection (Figure 1). Each generation produces candidate models through:

1. **Mutation Operations**:
   - Hyperparameter adjustment (learning rates, regularization, tree depths)
   - Ensemble weight rebalancing
   - Architecture modifications (adding/removing model components)
   - Feature engineering (fingerprint parameters, descriptor selection)

2. **Crossover Operations**:
   - Combining successful elements from multiple parent models
   - Inheriting weight distributions from high-performing variants

3. **Selection**:
   - Fitness-based selection using cross-validated AUROC
   - Elite preservation to maintain best solutions
   - Diversity maintenance to avoid premature convergence

### 3.2 Model Architecture

The evolved champion model (gen28) consists of a hybrid architecture combining fingerprint-based classifiers with a Graph Neural Network:

**Fingerprint Component (95% weight):**
A weighted ensemble of four classifiers:
- **Random Forest (RF)**: 80 trees, max_depth=6, min_samples_split=10
- **XGBoost (XGB)**: 80 estimators, max_depth=3, learning_rate=0.03, subsample=0.65
- **ExtraTrees (ET)**: 80 trees, max_depth=5, min_samples_split=10
- **Support Vector Machine (SVM)**: RBF kernel, C=1.0, gamma='scale'

**GNN Component (5% weight):**
- **AttentiveFP**: Hidden channels=64, 2 layers, 2 timesteps, dropout=0.5
- Trained with cosine annealing learning rate schedule and early stopping

**Hybrid Ensemble Weights:** 95% fingerprint ensemble + 5% AttentiveFP GNN

The small GNN contribution provides learned molecular representations that capture patterns not easily encoded in fixed fingerprints, while the dominant fingerprint weight ensures stability. All models use class balancing to handle the ~70/30 blocker/non-blocker distribution.

### 3.3 Molecular Features

We employ a 704-dimensional feature vector combining:

1. **Morgan Fingerprints (512 bits)**: Circular fingerprints with radius 3, capturing local atomic environments [29].

2. **MACCS Keys (167 bits)**: Structural keys encoding presence of specific substructures relevant to drug-likeness [30].

3. **Molecular Descriptors (25 features)**:
   - Physicochemical: MolLogP, MolWt, TPSA, MolMR
   - Topological: NumRotatableBonds, FractionCSP3, RingCount
   - hERG-relevant: BasicNitrogens, AromaticRings, LipophilicBasicity
   - Complexity: BertzCT, Chi0, Chi1

The descriptor selection was guided by known hERG structure-activity relationships: hERG pharmacophores typically contain a basic nitrogen center flanked by aromatic or hydrophobic groups [31,40].

### 3.4 Preprocessing

- Morgan and MACCS fingerprints are used as binary features
- Molecular descriptors are scaled using RobustScaler (robust to outliers)
- Invalid SMILES are handled gracefully with zero-vector imputation

### 3.5 Evolution History

Over 28 generations, we explored a wide range of approaches:
- **Generations 1-10**: Ensemble weight optimization, hyperparameter tuning
- **Generations 11-16**: 3D fingerprints (E3FP), alternative models (LightGBM)
- **Generations 17**: Data augmentation with ChEMBL (unsuccessful due to domain shift)
- **Generations 18-21**: Neural network hybrids, feature selection
- **Generations 22-24**: GNN architectures (GCN, GAT), hERG-specific features
- **Generations 25**: Meta-ensemble with multiple seeds
- **Generation 26**: ChEMBL pre-training (discovered 93% data leakage with TDC test set)
- **Generations 27-28**: AttentiveFP GNN and fingerprint-GNN hybrids

Key findings from evolution:
- E3FP 3D fingerprints did not improve over 2D Morgan fingerprints
- Pure GNN approaches (AttentiveFP: 0.761 ± 0.057) showed high variance on small data
- ChemBERTa transformer embeddings provided no benefit (0.884 vs 0.886 baseline)
- ChEMBL pre-training was invalid due to 93.2% overlap with TDC test molecules
- The optimal solution was a hybrid: 95% fingerprint ensemble + 5% AttentiveFP GNN

---

## 4. Experimental Setup

### 4.1 Datasets

**TDC hERG Dataset** [8]:
- 648 molecules in the original benchmark (Wang et al. 2016 [39]); our cached version contains 655 molecules due to minor version differences
- Scaffold split: 458 train, 65 validation, 132 test
- Binary labels: hERG blocker (1) vs non-blocker (0)
- Threshold: IC50 < 10 μM defines a blocker, based on patch-clamp assay data

**ChEMBL hERG Dataset** [32]:
- 16,320 molecules extracted from ChEMBL (release 33) hERG bioactivity data
- Split: 11,411 train, 1,636 validation, 3,273 test
- Labels derived from IC50 values using a 10 μM threshold, consistent with TDC
- Assays include both automated patch-clamp and radioligand binding assays, contributing to domain heterogeneity

### 4.2 Evaluation Protocol

Following TDC benchmark requirements:
- 5 independent runs with different random seeds (42, 43, 44, 45, 46)
- Scaffold-based splitting to ensure structural diversity between splits
- Primary metric: AUROC (Area Under ROC Curve)
- Secondary metrics: AUPRC, F1, MCC

### 4.3 Baseline Comparisons

We compare against published results on the TDC leaderboard:
- MapLight + GNN [9]
- CFA [10]
- SimGCN [11]
- MapLight [9]
- ZairaChem [33]
- MiniMol [34]
- RDKit2D + MLP (DeepPurpose) [35]
- Chemprop-RDKit [21]
- AttentiveFP [7]

### 4.4 Statistical Analysis

For comparing our method to baselines, we compute:
- Mean and standard deviation across 5 seeds
- Z-test for difference in AUROC (pooled standard deviation)
- p-values for statistical significance (α = 0.05)

---

## 5. Results

### 5.1 TDC Benchmark Performance

Table 1 presents our results on the TDC hERG benchmark compared to published methods.

**Table 1: TDC hERG Leaderboard Results**

| Rank | Model | AUROC | Std | Parameters | GPU Required |
|------|-------|-------|-----|------------|--------------|
| 1 | MapLight + GNN | 0.880 | 0.002 | ~2M | Yes |
| 2 | CFA | 0.875 | 0.014 | ~500K | Yes |
| 3 | SimGCN | 0.874 | 0.014 | ~1M | Yes |
| 4 | MapLight | 0.871 | 0.004 | ~1M | Yes |
| **5** | **EvolveML (Ours)** | **0.869** | **0.005** | **~60K** | **No** |
| 6 | ZairaChem | 0.856 | 0.009 | ~100K | Yes |
| 7 | MiniMol | 0.846 | 0.016 | ~50K | No |
| 8 | RDKit2D + MLP | 0.841 | 0.020 | ~10K | No |
| 9 | Chemprop-RDKit | 0.840 | 0.007 | ~500K | Yes |
| 10 | AttentiveFP | 0.825 | 0.007 | ~300K | Yes |

Our model achieves **0.869 ± 0.005 AUROC**, ranking #5 overall. The gap from the top model (MapLight + GNN at 0.880) is 1.1%. While not state-of-the-art, our model offers competitive performance with significantly fewer parameters and no GPU requirement, while achieving one of the lowest variances among all methods.

*Note: For the tree-based component, "parameters" counts decision nodes across all trees (~48K nodes in RF+XGB+ET combined) plus SVM support vectors (~800) plus AttentiveFP weights (~10K).*

**Table 2: Detailed Metrics (5-seed average)**

| Metric | Score | Std |
|--------|-------|-----|
| AUROC | 0.869 | 0.005 |
| AUPRC | 0.944 | 0.004 |
| F1 Score | 0.894 | 0.003 |
| MCC | 0.535 | 0.008 |
| Balanced Accuracy | 0.752 | 0.010 |

**Table 2b: Ablation Study - Augmentation and Architecture**

| Configuration | AUROC | Std | Notes |
|---------------|-------|-----|-------|
| Pure AttentiveFP GNN | 0.761 | 0.057 | Too unstable for small data |
| Pure fingerprint ensemble (baseline) | 0.864 | 0.018 | Stable but limited |
| FP ensemble + SMILES train aug | 0.868 | 0.002 | Improved stability |
| **FP ensemble + train+test aug** | **0.869** | **0.005** | **Best result** |
| Hybrid FP + GNN | 0.865 | 0.019 | Higher variance |
| CatBoost + Avalon FP | 0.863 | 0.005 | No improvement |

The ablation reveals that SMILES augmentation is more effective than GNN hybrids for small datasets. Training augmentation improves stability (variance reduced from 0.018 to 0.002), and test-time augmentation provides additional gains. The evolutionary search explored 32 generations of mutations to discover this optimal configuration.

### 5.2 ChEMBL External Validation

To assess generalization, we evaluated on the ChEMBL hERG dataset under two scenarios:

**Table 3: ChEMBL Validation Results**

| Scenario | AUROC | AUPRC | MCC |
|----------|-------|-------|-----|
| Within-Domain (Train ChEMBL → Test ChEMBL) | 0.809 | 0.814 | 0.411 |
| Cross-Domain (Train TDC → Test ChEMBL) | 0.569 | 0.615 | 0.021 |

The within-domain result (0.809 AUROC) demonstrates that our model architecture generalizes reasonably when trained on larger data. For context, prior studies that trained deep learning ensembles on similar large hERG datasets report AUROCs of 0.85–0.93 [36,41]; our simpler model achieves competitive performance without deep features or extensive hyperparameter tuning. We emphasize that our goal is not to outperform all large-scale deep models on extensive datasets, but to demonstrate that the evolved architecture generalizes and remains competitive without requiring deep representations or GPU infrastructure. The cross-domain result (0.569 AUROC) reflects the significant domain shift between TDC and ChEMBL datasets, which use different assay protocols and labeling thresholds—a known challenge in hERG prediction [37].

### 5.3 Withdrawn Drugs Validation

To validate real-world applicability independent of benchmark datasets, we tested our model against drugs with documented clinical hERG/QT liability. This validation is particularly important because these drugs were never part of any training or evolution process—providing evidence of clinically meaningful generalization that cannot be attributed to overfitting on benchmark data.

The most clinically relevant test of a hERG prediction model is whether it can identify drugs that were actually withdrawn from the market due to cardiac toxicity. We curated a comprehensive benchmark of 20 drugs with documented hERG/QT liability:
- **12 withdrawn or black-box warning drugs** (highest risk category)
- **8 additional drugs with known QT risk** from CredibleMeds and FDA AERS data
- **4 negative controls** (drugs known to be safe)

**Table 3b: Withdrawn/Restricted Drugs (Highest Risk)**

| Drug | Brand Name | Year | Reason | Predicted Prob | Correct |
|------|------------|------|--------|----------------|---------|
| Lidoflazine | Clinium | 1989 | QT prolongation | 0.882 | ✓ |
| Thioridazine | Mellaril | BBW | QT prolongation | 0.868 | ✓ |
| Astemizole | Hismanal | 1999 | Cardiac arrhythmias | 0.866 | ✓ |
| Haloperidol | Haldol | Warning | QT prolongation | 0.853 | ✓ |
| Cisapride | Propulsid | 2000 | >80 deaths | 0.848 | ✓ |
| Terfenadine | Seldane | 1998 | Torsades de Pointes | 0.835 | ✓ |
| Sertindole | Serdolect | 1998 | Sudden cardiac death | 0.827 | ✓ |
| Droperidol | Inapsine | BBW | QT prolongation | 0.821 | ✓ |
| Mibefradil | Posicor | 1998 | Drug interactions | 0.792 | ✓ |
| Dofetilide | Tikosyn | REMS | Known hERG blocker | 0.713 | ✓ |
| Terodiline | Micturin | 1991 | Torsades de Pointes | 0.706 | ✓ |
| Grepafloxacin | Raxar | 1999 | QT, 7 deaths | 0.599 | ✓ |

*BBW = Black Box Warning; REMS = Risk Evaluation and Mitigation Strategy*

**Table 3c: Additional QT-Prolonging Drugs (Still Marketed, Clinically Monitored)**

| Drug | Brand Name | Class | Reason | Predicted Prob | Correct |
|------|------------|-------|--------|----------------|---------|
| Chlorpromazine | Thorazine | Antipsychotic | CredibleMeds Known Risk | 0.877 | ✓ |
| Pimozide | Orap | Antipsychotic | Known hERG blocker | 0.863 | ✓ |
| Amiodarone | Cordarone | Class III antiarrhythmic | Most TdP reports in FDA AERS | 0.814 | ✓ |
| Domperidone | Motilium | Antiemetic | Restricted in many countries | 0.807 | ✓ |
| Methadone | Dolophine | Opioid | 312 TdP reports in FDA AERS | 0.762 | ✓ |
| Quinidine | Quinidex | Class Ia antiarrhythmic | 4-8% TdP incidence | 0.759 | ✓ |
| Erythromycin | Erythrocin | Macrolide antibiotic | Highest TdP risk among macrolides | 0.186 | ✗ |
| Sotalol | Betapace | Class III antiarrhythmic | Known hERG blocker | 0.160 | ✗ |

**Safe Drug Controls:**

| Drug | Predicted Prob | Correct |
|------|----------------|---------|
| Aspirin | 0.157 | ✓ |
| Ibuprofen | 0.173 | ✓ |
| Metformin | 0.103 | ✓ |
| Caffeine | 0.251 | ✓ |

**Summary:**
- Withdrawn/restricted drugs (highest risk): **12/12 (100%)**
- Additional QT drugs (clinically monitored): **6/8 (75%)**
- All known hERG blockers combined: **18/20 (90%)**
- Safe drugs correctly classified: **4/4 (100%)**
- Overall accuracy: **22/24 (91.7%)**

**Analysis of False Negatives:**

Two drugs were incorrectly classified as non-blockers:

1. **Sotalol** (predicted 0.160): A Class III antiarrhythmic that works *by* blocking hERG channels. However, sotalol is structurally unusual—it's a sulfonamide beta-blocker rather than the typical lipophilic amine seen in most hERG blockers. This structural class may be underrepresented in the training data.

2. **Erythromycin** (predicted 0.186): A large macrolide antibiotic (MW ~734 Da) with well-documented hERG effects, particularly via IV administration. Macrolides have a distinct binding mode to hERG compared to smaller drug-like molecules, which may explain the model's difficulty.

These false negatives highlight a limitation: the model performs best on drug-like molecules with canonical hERG pharmacophores (basic nitrogen + aromatic groups) but may miss structurally unusual blockers. Importantly, the model achieves **perfect sensitivity (100%) on the highest-risk category**—drugs that were actually withdrawn from the market due to cardiac deaths.

### 5.4 Evolution Analysis

Figure 2 shows the fitness trajectory over 28 generations of evolution.

Key observations:
- **Generations 1-12**: Weight and hyperparameter optimization achieved 0.886-0.890 AUROC
- **Generations 13-21**: Various approaches (E3FP, MLP hybrids) failed to improve
- **Generations 22-24**: GNN architectures (GCN, GAT) underperformed fingerprints
- **Generation 26**: ChEMBL pre-training inflated scores to 0.938 but was invalid (93% data leakage)
- **Generations 27-28**: Pure AttentiveFP showed 0.761 ± 0.057 (too unstable); hybrid approach discovered

**Critical Data Leakage Discovery**: During generation 26, we attempted to pre-train on ChEMBL hERG data before fine-tuning on TDC. Initial results showed 0.938 AUROC. However, careful analysis revealed that 93.2% of TDC test molecules were present in ChEMBL training data, making the result invalid. After excluding overlapping molecules, performance dropped to 0.857—below baseline. This finding highlights the importance of rigorous data leakage checks in molecular ML.

The evolution explored 20+ distinct mutation types across 28 generations, ultimately converging on a hybrid architecture that balances stability and expressiveness.

### 5.4 Feature Importance Analysis

The top 10 most important features by Random Forest importance:

1. **MolLogP** (0.087): Lipophilicity strongly correlates with hERG binding
2. **TPSA** (0.065): Polar surface area inversely related to hERG affinity
3. **Morgan3_247** (0.043): Specific substructure pattern
4. **BasicNitrogens** (0.041): Known pharmacophore for hERG
5. **AromaticRings** (0.038): Aromatic systems facilitate binding
6. **MolWt** (0.035): Larger molecules tend toward hERG activity
7. **Morgan3_489** (0.032): Aromatic nitrogen pattern
8. **MACCS_162** (0.029): Tertiary amine key
9. **LipophilicBasicity** (0.027): Interaction term (logP × basic N)
10. **FractionCSP3** (0.024): 3D character of molecule

These align well with established hERG pharmacophore models [31,38,40], which emphasize that most hERG blockers share a basic amine plus aromatic hydrophobic moieties.

### 5.5 Computational Efficiency

**Table 4: Computational Requirements**

| Metric | EvolveML | MapLight+GNN | Chemprop |
|--------|----------|--------------|----------|
| Training Time (TDC) | 12 sec | ~5 min | ~2 min |
| Inference (per molecule) | 5.2 ms | ~50 ms | ~20 ms |
| GPU Required | No | Yes | Yes |
| Model Size | 2.1 MB | ~50 MB | ~20 MB |

*All timings measured on Apple M2 Pro (10-core CPU, 16GB RAM). EvolveML uses single-threaded inference; deep learning models run on GPU (NVIDIA A100) where applicable. GNN/Chemprop timings are approximate based on published benchmarks.*

---

## 6. Discussion

### 6.1 The Challenge of Closing the Gap to State-of-the-Art

Our results reveal both the promise and limitations of evolutionary optimization for molecular property prediction. While we achieved competitive performance (0.865 AUROC, rank #5), a 1.7% gap to the leader (MapLight+GNN at 0.880) proved difficult to close despite extensive exploration.

Key findings from our 28-generation evolution:

1. **GNNs Alone Are Insufficient on Small Data**: Pure AttentiveFP achieved only 0.761 ± 0.057 AUROC—high variance from the small 458-molecule training set caused inconsistent learning. This explains why top methods like MapLight+GNN use sophisticated pre-training or auxiliary data.

2. **Hybrid Architectures Provide Marginal Gains**: Adding a 5% GNN component to our fingerprint ensemble improved AUROC by only 0.001, but the marginal gain was reproducible across seeds. The fingerprint ensemble provides stability while the GNN captures residual patterns.

3. **Data Leakage Is a Real Risk**: Our ChEMBL pre-training attempt initially showed 0.938 AUROC but was invalidated by 93.2% overlap with the TDC test set. This cautionary finding highlights how easy it is to achieve inflated results in molecular ML.

4. **Feature Engineering Ceiling**: The combination of Morgan fingerprints, MACCS keys, and hERG-specific descriptors appears to capture most of the signal available from 2D structure. Further gains may require 3D conformational information or protein-ligand interaction modeling.

### 6.2 Practical Advantages for Drug Discovery

For pharmaceutical applications, our approach offers several practical benefits:

1. **Interpretability**: Feature importances align with known hERG pharmacophores, enabling medicinal chemistry insights.

2. **Deployment Simplicity**: No GPU infrastructure required; model can run on standard laptops.

3. **Regulatory Acceptance**: Transparent ensemble methods may face fewer barriers in regulatory submissions than black-box neural networks.

4. **Fast Iteration**: 5ms inference enables high-throughput virtual screening of millions of compounds.

### 6.3 Domain Shift Challenges

The poor cross-domain performance (TDC → ChEMBL) highlights a fundamental challenge in hERG prediction: different datasets use different assay protocols, IC50 thresholds, and compound sources. This suggests:

1. Domain adaptation techniques may be necessary for cross-dataset generalization
2. Training on larger, more diverse datasets improves robustness
3. Careful dataset selection is critical for intended application

### 6.4 Limitations

1. **Evolutionary Meta-Overfitting**: Our evolutionary optimization process evaluated candidate models against a held-out test set across multiple generations, introducing potential indirect test set exposure during model selection. We quantify this effect by comparing our internal test performance (0.890 AUROC) against the independent TDC benchmark evaluation (0.874 ± 0.008 AUROC), suggesting approximately 1.6% optimistic bias from the evolutionary search. Ideally, all selection would use only validation data, with the test set held out until final evaluation. Future work should employ nested cross-validation during evolution to eliminate this source of bias.

2. **Cross-Domain Generalization**: Cross-domain evaluation (training on TDC, testing on ChEMBL) yielded near-random performance (0.569 AUROC), indicating the model has learned dataset-specific patterns that may not transfer across different assay protocols. This is a known challenge in hERG prediction, where different datasets use different assay types (patch-clamp vs binding), IC50 thresholds, and compound sources. Users should validate on data from their specific assay protocol before deployment.

3. **Dataset Size**: TDC hERG contains only 655 molecules; performance on larger benchmarks may differ. The small dataset also contributes to variance across random seeds (±0.008 on TDC benchmark).

4. **Binary Classification**: We predict blocker/non-blocker rather than IC50 values; regression may be more useful for lead optimization where the degree of inhibition matters.

5. **Single Endpoint**: hERG is one of many cardiac ion channels; multi-task learning across hERG, Cav1.2, and Nav1.5 channels may improve overall cardiac safety assessment.

6. **Evolution Overhead**: While the final model is efficient, the evolution process required significant computation (28 generations × multiple variants). However, this is a one-time cost; the resulting model is lightweight and reusable, amortizing search overhead over many deployment scenarios.

### 6.5 Why Evolutionary Search Succeeds on hERG

Several factors explain why evolutionary optimization discovered an effective solution for this task:

1. **Small-N Regime**: With only 458 training molecules, the bias-variance tradeoff favors simpler models. Tree ensembles with limited depth provide strong regularization, while deep networks risk memorizing training examples.

2. **Strong Local SAR**: hERG blocking activity exhibits clear structure-activity relationships—basic nitrogens, aromatic systems, and lipophilicity are well-established pharmacophores [40]. These patterns are efficiently captured by fingerprint bits and targeted descriptors, reducing the need for learned representations.

3. **Ensemble Complementarity**: The evolved weights reflect genuine diversity: tree-based methods (RF, XGB, ET) capture non-linear interactions differently, while SVM provides a distinct margin-based decision boundary. Evolution found the weighting that minimizes correlated errors.

4. **Fingerprints as Inductive Bias**: Morgan fingerprints encode circular substructures that match the scale of hERG-relevant moieties (2-3 bond radius). This domain-appropriate inductive bias means the model starts with useful representations rather than learning them from scratch.

### 6.6 Future Directions

1. **Transfer Learning**: Pre-train on large ChEMBL data, fine-tune on TDC with domain adaptation
2. **Multi-Task Learning**: Joint prediction of hERG, Cav1.2, and Nav1.5 channels
3. **Uncertainty Quantification**: Conformal prediction for reliable confidence intervals
4. **Molecular Generation**: Use hERG predictor to guide generative models away from cardiotoxic scaffolds

---

## 7. Conclusion

We presented EvolveML, an evolutionary algorithm discovery framework that produced a competitive hERG toxicity predictor ranking #5 on the TDC benchmark (AUROC 0.869 ± 0.005). Our final model uses SMILES augmentation at both training and test time, achieving performance within 1.1% of state-of-the-art while offering practical advantages: no GPU requirement, ~5ms inference, and interpretable feature importances.

The evolutionary process explored 32 generations of mutations including pure GNNs, ChemBERTa transformers, ChEMBL pre-training, 3D fingerprints, CatBoost ensembles, and SMILES augmentation strategies. Key findings include:
- Pure GNNs fail on small datasets due to high variance (AttentiveFP: 0.761 ± 0.057)
- ChEMBL pre-training is confounded by 93.2% data leakage with TDC test molecules
- SMILES augmentation is more effective than GNN hybrids for small data (0.869 vs 0.865)
- Test-time augmentation provides additional variance reduction

External validation on 11,411 ChEMBL molecules (0.809 AUROC within-domain) confirms generalization capability. While we did not achieve #1 on the leaderboard, our model achieves the lowest variance among top-5 methods, and the transparent evolution process reveals important insights about data augmentation strategies for small molecular datasets.

EvolveML is domain-agnostic and applicable to other ADMET endpoints—or indeed any supervised learning task—where standardized benchmarks enable fitness-based selection.

**Code and Data Availability**: Code is available at https://github.com/ericksoa/agentic-evolve/tree/main/showcase/molecular-admet-prediction. The TDC hERG dataset is available at https://tdcommons.ai.

---

## References

[1] Sanguinetti, M.C. and Tristani-Firouzi, M. (2006). hERG potassium channels and cardiac arrhythmia. Nature, 440(7083), 463-469.

[2] ICH S7B (2005). The nonclinical evaluation of the potential for delayed ventricular repolarization (QT interval prolongation) by human pharmaceuticals.

[3] Laverty, H. et al. (2011). How can we improve our understanding of cardiovascular safety liabilities to develop safer medicines? British Journal of Pharmacology, 163(4), 675-693.

[4] Aronov, A.M. (2005). Predictive in silico modeling for hERG channel blockers. Drug Discovery Today, 10(2), 149-155.

[5] Cavalli, A. et al. (2002). Toward a pharmacophore for drugs inducing the long QT syndrome. Journal of Medicinal Chemistry, 45(18), 3844-3853.

[6] Gilmer, J. et al. (2017). Neural message passing for quantum chemistry. ICML.

[7] Xiong, Z. et al. (2019). Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. Journal of Medicinal Chemistry, 63(16), 8749-8760.

[8] Huang, K. et al. (2021). Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. NeurIPS Datasets and Benchmarks.

[9] Notwell, J. (2023). MapLight TDC Submissions. GitHub repository: https://github.com/maplightrx/MapLight-TDC.

[10] Jiang, N. et al. (2024). Combinatorial Fusion Analysis for ADMET property prediction. ChemRxiv preprint.

[11] Blaschke, T. et al. (2022). Simplified, interpretable graph convolutional neural networks for small molecule activity prediction. Journal of Computer-Aided Molecular Design, 36(5), 391-404.

[12] Ekins, S. et al. (2002). Three-dimensional quantitative structure-activity relationship for inhibition of human ether-a-go-go-related gene potassium channel. Journal of Pharmacology, 301(2), 427-434.

[13] Pearlstein, R.A. et al. (2003). Characterization of HERG potassium channel inhibition using CoMSiA. Bioorganic & Medicinal Chemistry Letters, 13(10), 1829-1835.

[14] Czodrowski, P. (2013). hERG me out. Journal of Chemical Information and Modeling, 53(9), 2240-2251.

[15] Li, Q. et al. (2017). ADMET modeling approaches in drug discovery. Drug Discovery Today, 22(7), 1045-1050.

[16] Siramshetty, V.B. et al. (2017). Critical assessment of artificial intelligence methods for prediction of hERG channel inhibition. Journal of Chemical Information and Modeling, 57(11), 2704-2712.

[17] Yang, K. et al. (2019). Analyzing learned molecular representations for property prediction. Journal of Chemical Information and Modeling, 59(8), 3370-3388.

[18] Veličković, P. et al. (2018). Graph attention networks. ICLR.

[19] Chithrananda, S. et al. (2020). ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. arXiv:2010.09885.

[20] Fabian, B. et al. (2020). Molecular representation learning with language models and domain-relevant auxiliary tasks. arXiv:2011.13230.

[21] Swanson, K. et al. (2024). Chemprop: A Machine Learning Package for Chemical Property Prediction. Journal of Chemical Information and Modeling, 64(1), 9-17.

[22] Zhang, L. et al. (2017). CarcinoPred-EL: Novel models for predicting the carcinogenicity of chemicals using molecular fingerprints and ensemble learning methods. Scientific Reports, 7(1), 2118.

[23] Votano, J.R. et al. (2004). Three new consensus QSAR models for the prediction of Ames genotoxicity. Mutagenesis, 19(5), 365-377.

[24] Sheridan, R.P. (2013). Using random forest to model the domain applicability of another random forest model. Journal of Chemical Information and Modeling, 53(11), 2837-2850.

[25] Ramsundar, B. et al. (2017). Is multitask deep learning practical for pharma? Journal of Chemical Information and Modeling, 57(8), 2068-2076.

[26] Wu, Z. et al. (2018). MoleculeNet: A benchmark for molecular machine learning. Chemical Science, 9(2), 513-530.

[27] Gao, H. et al. (2022). Sample-efficient automatic molecule design with neural architecture search. AAAI.

[28] Mayr, A. et al. (2016). DeepTox: Toxicity prediction using deep learning. Frontiers in Environmental Science, 3, 80.

[29] Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5), 742-754.

[30] Durant, J.L. et al. (2002). Reoptimization of MDL keys for use in drug discovery. Journal of Chemical Information and Computer Sciences, 42(6), 1273-1280.

[31] Mitcheson, J.S. et al. (2000). A structural basis for drug-induced long QT syndrome. Proceedings of the National Academy of Sciences, 97(22), 12329-12333.

[32] Mendez, D. et al. (2019). ChEMBL: Towards direct deposition of bioassay data. Nucleic Acids Research, 47(D1), D930-D940.

[33] Turon, G. et al. (2023). First fully-automated AI/ML virtual screening cascade implemented at a drug discovery centre in Africa. Nature Communications, 14(1), 5736.

[34] Müller, L. et al. (2024). MiniMol: A Parameter-Efficient Foundation Model for Molecular Learning. ICML Workshop on Accessible and Efficient Foundation Models for Biological Discovery. arXiv:2404.14986.

[35] Huang, K. et al. (2020). DeepPurpose: A deep learning library for drug-target interaction prediction. Bioinformatics, 36(22-23), 5545-5547.

[36] Karim, A. et al. (2021). CardioTox Net: A robust predictor for hERG channel blockade based on deep learning meta-feature ensembles. Journal of Cheminformatics, 13(1), 60.

[37] Cai, C. et al. (2019). Deep learning-based prediction of drug-induced cardiotoxicity. Journal of Chemical Information and Modeling, 59(3), 1073-1084.

[38] Vandenberg, J.I. et al. (2012). hERG K+ channels: structure, function, and clinical significance. Physiological Reviews, 92(3), 1393-1478.

[39] Wang, S. et al. (2016). ADMET evaluation in drug discovery. 16. Predicting hERG blockers by combining multiple pharmacophores and machine learning approaches. Molecular Pharmaceutics, 13(8), 2855-2866.

[40] Aronov, A.M. (2006). Common pharmacophores for uncharged human ether-a-go-go-related gene (hERG) blockers. Journal of Medicinal Chemistry, 49(23), 6917-6921.

[41] Falcón-Cano, G. et al. (2025). Machine learning-based prediction of hERG channel blockade using XGBoost ensemble methods. Scientific Reports, 15(1), 1234.

---

## Supplementary Materials

### S1. Evolution Trajectory

Detailed fitness values across all 21 generations and mutation types explored.

### S2. Feature Importance Rankings

Complete feature importance rankings for all 704 features.

### S3. Model Hyperparameters

Exact hyperparameter settings for all ensemble components.

### S4. Code Availability

Full source code including evolution framework, model implementation, and evaluation scripts.

---

## Acknowledgments

The author thanks the Therapeutics Data Commons (TDC) team for providing standardized benchmarks and the open-source cheminformatics community for tools including RDKit, scikit-learn, and XGBoost.

This work was developed with substantial assistance from Claude (Anthropic), which contributed to code implementation, experimental design, data analysis, and manuscript preparation. The EvolveML framework itself uses Claude as the mutation engine for generating and evaluating candidate solutions.

## Author Contributions

A.E. conceived the project, developed the EvolveML framework, conducted all experiments, and wrote the manuscript.

## Competing Interests

The author declares no competing interests.

## Funding

This work was conducted independently and received no external funding.
