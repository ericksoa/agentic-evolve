# Global Chess Challenge 2025 - Evolve SDK Project

Competing in the [AIcrowd Global Chess Challenge 2025](https://www.aicrowd.com/challenges/global-chess-challenge-2025) using evolutionary optimization of LLM training strategies.

## Competition Overview

| Metric | Value |
|--------|-------|
| **Goal** | Train <8B LLM to play chess from text |
| **Prize** | $25,000 ($10K first place) |
| **Deadline** | January 31, 2026 |
| **Metric** | Average Centipawn Loss (ACPL) |

## Current Status

| Generation | Champion | ACPL | vs Random | Key Innovation |
|------------|----------|------|-----------|----------------|
| Gen0 | gen0_elite_only | 113.2 | 8.7x better | High Elo (2200+) data |
| Gen1 | gen0_elite_only | 88.0 | 11.2x better | Refined params |
| Gen2 | gen0_elite_only | 92.6 | 10.7x better | (plateau) |
| Gen3 | **gen3_cross** | **85.6** | **11.5x better** | Crossover breakthrough |
| Gen4 | gen4_mutc | 87.1 | 11.3x better | Mutation variant |
| Gen5 | gen5_mutb | 96.9 | 10.2x better | (regression) |

**Baseline**: Random moves = 987 ACPL
**Champion**: gen3_cross with ACPL 85.6 (discovered Gen3)

## Evolution Trajectory

![Evolution Trajectory](visualizations/evolution_trajectory.svg)

*ACPL over generations - lower is better. Random baseline = 987.*

## Evolution Strategy

We're evolving **training strategies**, not the model architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION TARGETS                             │
├─────────────────────────────────────────────────────────────────┤
│  Data Selection    │  Prompt Format    │  Training Params       │
│  ─────────────────│──────────────────│──────────────────────── │
│  • Min/Max Elo    │  • FEN style      │  • Learning rate       │
│  • Game phases    │  • Move format    │  • Epochs              │
│  • Time controls  │  • Context info   │  • Batch size          │
│  • Tactical focus │  • Instructions   │  • LoRA rank/alpha     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Generation Workflow

Each generation follows this standardized workflow:

### Phase 1: Evaluate Current Population

```
┌────────────────────────────────────────────────────────────────┐
│  For each candidate in population:                              │
│                                                                 │
│  1. Generate training data using candidate's config             │
│  2. Fine-tune base model (Qwen2.5-3B)                          │
│  3. Evaluate on TEST positions (50 positions, Stockfish depth) │
│  4. Record ACPL, legal rate, best move rate                    │
│  5. Update evolution.json with fitness                          │
└────────────────────────────────────────────────────────────────┘
```

### Phase 2: Selection & Reproduction

```
┌────────────────────────────────────────────────────────────────┐
│  1. Rank candidates by fitness (lower ACPL = better)           │
│  2. Select top 2 as elites (survive to next gen)               │
│  3. Generate new candidates:                                    │
│     • 2 mutations of elite #1                                   │
│     • 1 mutation of elite #2                                    │
│     • 1 crossover of elite #1 × elite #2                       │
│  4. Update champion if new best found                           │
└────────────────────────────────────────────────────────────────┘
```

### Phase 3: Documentation & Learning

```
┌────────────────────────────────────────────────────────────────┐
│  1. Generate GEN{N}_RESULTS.md with:                           │
│     • Fitness table (all candidates)                           │
│     • What worked (winning strategies)                         │
│     • What didn't work (failed hypotheses)                     │
│     • Key learnings for next generation                        │
│                                                                 │
│  2. Generate visualizations:                                    │
│     • fitness_gen{N}.svg - Bar chart of candidate fitness      │
│     • evolution_trajectory.svg - ACPL over generations         │
│     • strategy_comparison.svg - Heatmap of config parameters   │
│                                                                 │
│  3. Update README.md with latest results                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Visualization Standards

### 1. Fitness Bar Chart (per generation)

Shows all candidates in a generation ranked by ACPL:

```
fitness_gen{N}.svg
┌──────────────────────────────────────────────────────────┐
│  Generation N Fitness                                     │
│                                                          │
│  gen{N}_elite    ████████████████████ 45.2 ACPL ★        │
│  gen{N}_tactical ██████████████████   52.1 ACPL          │
│  gen{N}_endgame  █████████████████    58.3 ACPL          │
│  gen{N}_baseline ████████████         89.7 ACPL          │
│  gen{N}_failed   ██                   450.2 ACPL ✗       │
│                                                          │
│  ★ = Champion   ✗ = Failed validation                    │
└──────────────────────────────────────────────────────────┘
```

### 2. Evolution Trajectory (cumulative)

Shows best ACPL over all generations:

```
evolution_trajectory.svg
┌──────────────────────────────────────────────────────────┐
│  ACPL                                                    │
│  1000 ─┬─────────────────────────────────────────────    │
│        │●                                                │
│   500 ─┼──●                                              │
│        │    ●                                            │
│   200 ─┼──────●──●                                       │
│        │          ●──●──●                                │
│   100 ─┼─────────────────●──●                            │
│        │                      ●──●──●                    │
│    50 ─┼─────────────────────────────●                   │
│        └─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──►                 │
│          0  1  2  3  4  5  6  7  8  9  Generation        │
└──────────────────────────────────────────────────────────┘
```

### 3. Strategy Comparison Heatmap

Shows which parameter combinations work best:

```
strategy_comparison.svg
┌──────────────────────────────────────────────────────────┐
│                min_elo  phase_focus  prompt  lr   epochs │
│  gen5_best       2200    midgame     rich   2e-5   4    │
│  gen4_good       2000    endgame     std    3e-5   3    │
│  gen3_decent     1800    mixed       std    2e-5   3    │
│  gen2_weak       1600    opening     min    5e-5   2    │
│                                                          │
│  Color: Green=good, Yellow=neutral, Red=bad              │
└──────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

The project includes a complete training pipeline for fine-tuning LLMs on chess data.

### Data Extraction

```bash
# Extract training data from PGN files
python python/extract_training_data.py \
    --pgn data/games/lichess_elite.pgn \
    --output data/training.jsonl \
    --min-elo 2200 \
    --max-examples 50000 \
    --depth 18
```

**Current Training Data**: `data/training_full.jsonl`
- 2,000 examples extracted
- Average Elo: 1,889
- Phase distribution: 32% opening, 68% middlegame
- File size: 1.4MB

### Local Training (Apple Silicon)

```bash
# Test pipeline components
python python/test_pipeline.py

# Train model locally (MPS backend)
python python/train_model.py \
    --data data/training_full.jsonl \
    --epochs 3 \
    --batch-size 2
```

### Cloud Training (Lightning.ai)

See [cloud/lightning/README.md](cloud/lightning/README.md) for detailed instructions.

```bash
# 1. Open a Lightning Studio with GPU (L4, A10G, or A100)
# 2. Clone repo and upload training data
# 3. Run setup and training:
cd global-chess-challenge-2025/cloud/lightning
./setup.sh
./train.sh --data ~/chess-training/data/training_full.jsonl
```

| GPU | VRAM | Cost/Hour | Training Time (50K examples) |
|-----|------|-----------|------------------------------|
| L4 | 24GB | ~$0.40 | ~3 hours |
| A10G | 24GB | ~$0.60 | ~2 hours |
| A100-40G | 40GB | ~$1.50 | ~1 hour |

---

## Directory Structure

```
global-chess-challenge-2025/
├── README.md                      # This file (updated each gen)
├── CLAUDE.md                      # Claude Code instructions
├── python/
│   ├── extract_training_data.py   # PGN → JSONL extraction
│   ├── train_model.py             # LoRA fine-tuning
│   ├── evaluate_model.py          # Model evaluation
│   ├── pipeline.py                # Full pipeline orchestrator
│   ├── test_pipeline.py           # Component testing
│   ├── evaluate.py                # ACPL evaluation
│   ├── generate_data.py           # Training data generation
│   ├── run_generation.py          # Evolution generation runner
│   └── visualize.py               # Generate SVG charts
├── cloud/
│   ├── lightning/                 # Lightning.ai training scripts
│   │   ├── setup.sh               # Environment setup
│   │   ├── train.sh               # Training script
│   │   └── README.md              # Lightning.ai guide
│   └── lambda/                    # Lambda Labs (alternative)
├── data/
│   ├── games/                     # PGN game databases
│   ├── training_full.jsonl        # 2K examples extracted
│   └── training_test.jsonl        # Small test set
├── starter-kit/                   # AIcrowd official starter
├── visualizations/                # Generated charts
│   ├── fitness_gen0.svg
│   ├── evolution_trajectory.svg
│   └── strategy_comparison.svg
├── results/                       # Per-generation documentation
│   ├── GEN0_RESULTS.md
│   ├── GEN1_RESULTS.md
│   └── ...
└── .evolve-sdk/
    └── evolve_chess_training_strategy/
        ├── evolution.json         # Master evolution state
        └── mutations/             # All candidate configs
```

---

## Generation Results

### Gen0: Initial Population (In Progress)

| Candidate | Strategy | Hypothesis | ACPL | Status |
|-----------|----------|------------|------|--------|
| baseline | Elo 1800+, mixed phases | Balanced starting point | - | Pending |
| elite_only | Elo 2200+, classical | Quality over quantity | - | Pending |
| endgame_focus | 60% endgame | Cleaner training signal | - | Pending |
| tactical | Middlegame tactics | Pattern recognition | - | Pending |
| curriculum | Easy→hard progression | Build foundations first | - | Pending |
| diverse_prompt | Rich position analysis | Better reasoning | - | Pending |

---

## Key Learnings

### What Works
- **High Elo threshold (2200+)**: Quality training data beats quantity
- **Balanced middlegame focus (60%)**: Best for pattern learning
- **Crossover strategy**: gen3_cross combined best traits from parents
- **Conservative learning rate (1e-5)**: Prevents overfitting
- **Higher LoRA rank (64+)**: More capacity helps

### What Doesn't Work
- **Endgame-heavy training**: Worse generalization
- **Tactical filtering**: Too narrow, missed important patterns
- **Low Elo data (1600-1800)**: Noisy training signal
- **Very high learning rates**: Caused instability

### Surprising Findings
- **Crossover outperformed mutations**: gen3_cross became champion
- **Prompt format had minimal impact**: FEN representation worked fine
- **Curriculum learning didn't help**: Direct training was better

---

## Running Evolution

```bash
# Activate environment
cd /path/to/global-chess-challenge-2025
source starter-kit/venv/bin/activate

# Run one generation
python python/run_generation.py

# Or use the evolve skill
/evolve-ml chess training strategy --resume
```

## Submission

```bash
# Submit best model to AIcrowd
./starter-kit/aicrowd_submit.sh
```

---

## References

- [Competition Page](https://www.aicrowd.com/challenges/global-chess-challenge-2025)
- [Starter Kit](https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit)
- [Stockfish](https://stockfishchess.org/)
- [python-chess](https://python-chess.readthedocs.io/)
