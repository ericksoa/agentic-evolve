# Molecular ADMET Prediction: hERG Cardiac Toxicity

**TL;DR**: We predict if a drug molecule will cause heart problems, before it's ever tested on humans.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    THE PROBLEM                          │
                    │                                                         │
                    │   Drug Candidate  ──►  hERG Channel Block  ──►  💔      │
                    │   (new medicine)       (ion channel in heart)  (arrhythmia)│
                    │                                                         │
                    │   ~40% of drug withdrawals are due to hERG toxicity    │
                    │   Each failed drug = $1-2 billion wasted               │
                    └─────────────────────────────────────────────────────────┘
```

## What is hERG and Why Does It Matter?

**hERG** (human Ether-a-go-go Related Gene) is a protein that forms an ion channel in your heart. It controls the heartbeat rhythm by letting potassium ions flow in and out of heart cells.

```
    Normal Heart Cell                    Drug Blocks hERG Channel
    ================                     ========================

    K+ ions flow freely                  K+ ions can't exit
         ↓↓↓↓↓                                  ✖
    ┌─────────────┐                     ┌─────────────┐
    │  ═══════    │  ◄── hERG          │  ═══✖═══    │  ◄── Blocked!
    │   Heart     │      Channel        │   Heart     │
    │    Cell     │                     │    Cell     │
    └─────────────┘                     └─────────────┘
         │                                    │
         ▼                                    ▼
    Normal heartbeat                    QT prolongation → Arrhythmia → ☠️
```

**The Business Case**: Pharma companies spend $1-2 billion developing a single drug. If hERG toxicity is discovered late (in clinical trials or after market release), all that investment is lost. Early prediction saves lives AND billions of dollars.

## What Our Model Does

We take a molecule's structure (written as a "SMILES" string) and predict how likely it is to block hERG:

```
    INPUT                              OUTPUT
    =====                              ======

    "CC(=O)Oc1ccccc1C(=O)O"    ──►    0.23  (Low risk - probably safe)
         ↑
    This is Aspirin's
    molecular structure

    "CN1CCN(CC1)c2ccccc2"      ──►    0.87  (High risk - likely blocks hERG!)
         ↑
    A molecule with features
    known to block hERG
```

## How It Works (ELI5 Version)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE PREDICTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: MOLECULE IN                    Step 2: EXTRACT FEATURES
═══════════════════                    ════════════════════════

   H   O                               Fingerprints (2048 bits):
    \ //                               "Does it have this pattern?" → 0 or 1
 H-C-C-O-H         ──────►             [0,1,0,1,1,0,0,1,1,0,1,...]
    |
   H                                   Descriptors (25 numbers):
                                       • MolLogP = 1.2 (how greasy?)
 "CCO" (ethanol)                       • NumAromaticRings = 0
                                       • BasicNitrogens = 0
                                       • MolWeight = 46.07
                                       • ... 21 more properties


Step 3: ENSEMBLE PREDICTION            Step 4: OUTPUT
═══════════════════════════            ══════════════

┌─────────────────┐
│ Random Forest   │──┐
│  (200 trees)    │  │                     Risk Score
└─────────────────┘  │                     ══════════
                     │  Weighted
┌─────────────────┐  │  Average            0.0 ─── Safe
│ Gradient Boost  │──┼────────────►        0.5 ─── Uncertain
│  (150 trees)    │  │                     1.0 ─── Dangerous
└─────────────────┘  │
                     │
┌─────────────────┐  │
│ Extra Trees     │──┘
│  (200 trees)    │
└─────────────────┘
```

## Why These Features Matter for hERG

Our model focuses on molecular features known to affect hERG binding:

```
Feature               Why It Matters for hERG                    Example
═══════════════════════════════════════════════════════════════════════════════
MolLogP (lipophilicity)   Greasy molecules penetrate cell         Drugs need to
                          membranes to reach the channel          be "just right"

NumAromaticRings          Flat ring structures fit into           [benzene ring]
                          the hERG binding pocket                  fits like a key

BasicNitrogens            Positive charge at body pH binds        NH₂ groups are
                          to negatively charged channel            common culprits

MolWeight                 Bigger molecules may not fit;           Sweet spot:
                          too small won't bind strongly            300-500 Da

TPSA (polar surface)      Affects how molecule interacts          Balance needed
                          with water vs membrane                   for access
```

## Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL COMPARISON                               │
├─────────────────────────┬────────────┬────────────┬───────────┬────────────┤
│ Model                   │ Test AUC   │ CV AUC     │ Speed     │ Improvement│
├─────────────────────────┼────────────┼────────────┼───────────┼────────────┤
│ Baseline (Fingerprint)  │   0.8284   │   0.8488   │  3.85ms   │     -      │
│ Baseline (Descriptors)  │   0.8434   │   0.8798   │  1.71ms   │   +1.8%    │
│ ► Evolved Ensemble      │   0.8711   │   0.8917   │  3.12ms   │   +5.2%    │
└─────────────────────────┴────────────┴────────────┴───────────┴────────────┘

What do these numbers mean?
═══════════════════════════
• AUC = 0.87 means: if we pick a random safe molecule and a random
  dangerous molecule, our model ranks them correctly 87% of the time

• Speed = 3.12ms means: we can screen ~320 molecules per second
  (fast enough for virtual screening of millions of compounds)
```

## Dataset

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TDC hERG BENCHMARK DATASET                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Total: 655 molecules                                                      │
│   ├── Training:   458 molecules (70%)                                       │
│   ├── Validation:  65 molecules (10%)                                       │
│   └── Test:       132 molecules (20%)                                       │
│                                                                             │
│   Labels:                                                                   │
│   ├── hERG Blockers (dangerous):     68%  ████████████████████░░░░░░░░     │
│   └── Non-blockers (safe):           32%  ██████████░░░░░░░░░░░░░░░░░░     │
│                                                                             │
│   Split method: Scaffold-based (ensures chemically diverse test set)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Set up environment
cd showcase/molecular-admet-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Test the evolved model
python evaluate.py evolved_ensemble.py --verbose

# 3. Run on your own molecules (example)
python -c "
from evolved_ensemble import HERGPredictor
from tdc.single_pred import Tox

# Load model and data
predictor = HERGPredictor()
data = Tox(name='hERG')
split = data.get_split()

# Train and predict
predictor.fit(split['train']['Drug'].tolist(), split['train']['Y'].tolist())

# Test on some molecules
test_molecules = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
]
for smi, prob in zip(test_molecules, predictor.predict_proba(test_molecules)):
    risk = 'HIGH RISK' if prob > 0.5 else 'LOW RISK'
    print(f'{smi[:30]:30} -> {prob:.2%} ({risk})')
"
```

## File Structure

```
molecular-admet-prediction/
├── README.md                 # This file (you are here!)
├── requirements.txt          # Python dependencies
├── evaluate.py               # Evaluation harness
├── evolve_config.json        # Evolution configuration
│
├── baseline_fingerprint.py   # Baseline 1: Morgan fingerprints + Random Forest
├── baseline_descriptors.py   # Baseline 2: Molecular descriptors + GBM
└── evolved_ensemble.py       # Final: Hybrid ensemble (BEST)
```

## For the ML Engineer: Technical Details

**Feature Engineering**:
- Morgan fingerprints (radius=2, 2048 bits) - circular substructure encoding
- 25 RDKit descriptors selected for hERG relevance
- Robust scaling on descriptor features only

**Model Architecture**:
- Ensemble of 3 diverse tree-based models
- Weights optimized via cross-validation: RF(0.35) + GBM(0.35) + ET(0.30)
- Class balancing for imbalanced dataset (68% positive)

**Evaluation Protocol**:
- 5-fold stratified CV on training set
- Scaffold-based split (prevents data leakage from similar molecules)
- Primary metric: ROC-AUC (handles class imbalance)

## References

- [Therapeutics Data Commons - hERG](https://tdcommons.ai/single_pred_tasks/tox/#herg)
- [RDKit: Open-source cheminformatics](https://www.rdkit.org/)
- Czodrowski, P. (2013). "hERG Me Out" - J. Chem. Inf. Model.

---

*Built with [Agentic Evolve](https://github.com/anthropics/agentic-evolve) - evolutionary optimization for ML*
