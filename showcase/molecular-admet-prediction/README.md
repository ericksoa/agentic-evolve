# Molecular ADMET Prediction: hERG Toxicity

Evolving machine learning models to predict hERG (cardiac) toxicity from molecular structure using the Agentic Evolve framework.

## Problem Statement

hERG (human Ether-a-go-go Related Gene) channel inhibition is one of the most critical safety endpoints in drug discovery. Drug-induced hERG block causes cardiac arrhythmias (QT prolongation), leading to:
- ~40% of drug withdrawals from the market
- Billions in failed drug development costs
- Patient safety risks

**Goal**: Maximize ROC-AUC for predicting hERG channel inhibition from molecular SMILES strings.

## Dataset

- **Source**: Therapeutics Data Commons (TDC) hERG benchmark
- **Size**: 655 molecules (458 train, 65 validation, 132 test)
- **Split**: Scaffold-based (ensures chemical diversity)
- **Class balance**: ~68% hERG blockers (positive class)

## Baseline Performance

| Model | Test ROC-AUC | Inference (ms/mol) | Valid |
|-------|--------------|-------------------|-------|
| Morgan FP + RF | 0.8284 | 3.85 | Yes |
| Descriptors + GBM | **0.8434** | 1.71 | Yes |

## Evolution Target

- **Current best**: 0.8434 ROC-AUC
- **Target**: 0.90 ROC-AUC
- **Constraint**: < 100ms inference per molecule

## Solution Interface

Solutions must implement the `HERGPredictor` class:

```python
class HERGPredictor:
    def fit(self, X_smiles: list[str], y: list[int]) -> 'HERGPredictor':
        """Train on SMILES strings and binary labels."""
        ...

    def predict_proba(self, X_smiles: list[str]) -> np.ndarray:
        """Return probability of hERG blocking for each molecule."""
        ...

    def get_feature_importance(self) -> dict:  # Optional
        """Return feature importance for interpretability."""
        ...
```

## Optimization Strategies

The evolution explores:
1. **Fingerprint combinations**: Morgan, MACCS, RDKit, Avalon
2. **Molecular descriptors**: Physicochemical, topological, pharmacophore
3. **Model architectures**: Ensembles, boosting, neural networks
4. **Feature engineering**: hERG-specific features (basic N, lipophilicity, aromaticity)
5. **Hyperparameter optimization**: Automatic tuning within evolution

## Running Evolution

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test baseline
python evaluate.py baseline_descriptors.py --verbose

# Run evolution (from project root)
cd /path/to/agentic-evolve
PYTHONPATH=sdk python3 -m evolve_sdk \
    --config showcase/molecular-admet-prediction/evolve_config.json \
    --max-generations 10 \
    --population-size 4 \
    --no-parallel
```

## Domain Knowledge

Key molecular features for hERG binding:
- **Basic nitrogen**: Protonated at physiological pH, binds to channel
- **Hydrophobic aromatics**: Fit into hydrophobic pockets
- **Lipophilicity (logP)**: Correlates with membrane penetration
- **Molecular flexibility**: Allows conformational adaptation

## Results

| Model | Test ROC-AUC | CV ROC-AUC | Inference | Improvement |
|-------|--------------|------------|-----------|-------------|
| Baseline (Fingerprint) | 0.8284 | 0.8488 | 3.85ms | - |
| Baseline (Descriptors) | 0.8434 | 0.8798 | 1.71ms | +1.8% |
| **Evolved Ensemble** | **0.8711** | **0.8917** | 3.12ms | **+5.2%** |

### Key Improvements in Evolved Solution

1. **Hybrid Feature Representation**: Combined Morgan fingerprints (2048 bits) with 25 physicochemical descriptors
2. **Ensemble Architecture**: Weighted voting of RF + GBM + ExtraTrees (0.35/0.35/0.30)
3. **hERG-Specific Features**: Basic nitrogen count, aromatic ring features, lipophilicity
4. **Robust Error Handling**: Graceful handling of invalid SMILES and edge cases

### Feature Importance (Top 5)

| Feature | Importance | Relevance to hERG |
|---------|------------|-------------------|
| MolLogP | 0.089 | Membrane permeability, binding affinity |
| NumAromaticRings | 0.072 | π-π stacking in binding pocket |
| BasicNitrogens | 0.068 | Key pharmacophore for hERG block |
| TPSA | 0.054 | Polar surface affects channel access |
| MolWt | 0.048 | Size constraints for channel binding |

## References

- [TDC hERG Benchmark](https://tdcommons.ai/single_pred_tasks/tox/#herg)
- Czodrowski, P. (2013). hERG Me Out. J. Chem. Inf. Model.
- [RDKit Documentation](https://www.rdkit.org/docs/)

---

*Generated with [Agentic Evolve](https://github.com/anthropics/agentic-evolve)*
