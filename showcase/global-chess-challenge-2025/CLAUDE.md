# Global Chess Challenge 2025 - Evolve SDK Project

## CRITICAL: Use Evolve SDK Memory System

**DO NOT create random .md files to store learnings, DRIs, roadmaps, or project state.**

Always use the evolve-sdk memory system:
- **Memory file**: `.evolve-sdk/evolve_chess_training_strategy/memory.json`
- **DRIs go in**: `memory.json` → `dont_repeat_incidents` array
- **Learnings go in**: `memory.json` → `learnings` object
- **Submissions go in**: `memory.json` → `submissions` array
- **Roadmaps go in**: `memory.json` → `comprehensive_roadmap` object
- **Credentials**: Always check `.env` file FIRST for API keys

Before any HuggingFace submission, CHECK the DRIs in memory.json to avoid repeating past mistakes.

**NEVER create these files** (use memory.json instead):
- IMPROVEMENT_PLAN.md
- WORK_IN_PROGRESS.md
- WINNING_STRATEGY.md
- Any other .md files for plans/state

## CRITICAL: Current Best Results

**Target: <100 ACPL for competitive submission**

| Model | ACPL | Status |
|-------|------|--------|
| V6 checkpoint-4000 (local) | **192.3** | Best local eval |
| Submission 307752 (early) | **~146** | Evaluating (3/100 matches) |
| V4 completed model | TBD | Ready to evaluate |

- Leader: 46.4 ACPL, Baseline: 71.9 ACPL
- We're ~3-4x behind leaders
- Current level: ~1400 Elo (weak amateur)

## Competition Overview
- **Platform**: AIcrowd
- **URL**: https://www.aicrowd.com/challenges/global-chess-challenge-2025
- **Prize**: $25,000 total ($10K first place)
- **Deadline**: January 31, 2026 (Round 2)
- **Goal**: Train a <8B parameter LLM to play chess from text

## Problem Specification

**Input**: Text prompt containing:
- FEN position string
- Side to move (white/black)
- List of legal moves in UCI format

**Output**:
- `<uci_move>e2e4</uci_move>` - The chosen move
- `<rationale>...</rationale>` - One sentence explanation

**Metric**: Average Centipawn Loss (ACPL) via Stockfish Level 20

## Evolution Strategy

This project uses `/evolve-ml` to evolve chess-playing LLM strategies.

### What We're Evolving

1. **Training Data Selection**
   - Which games to train on (high-Elo only? diverse Elo?)
   - Game phase focus (openings, middlegame, endgame)
   - Position diversity vs quality tradeoff

2. **Data Augmentation**
   - Mirror positions (a1-h8 reflection)
   - Swap colors
   - Multiple moves from same position

3. **Curriculum Learning**
   - Easy→hard position ordering
   - Puzzle difficulty progression
   - Opening→endgame or endgame→opening

4. **Prompt Engineering**
   - FEN representation (raw vs annotated)
   - Legal moves formatting
   - Context window usage

5. **Training Hyperparameters**
   - Learning rate schedule
   - Batch size
   - Number of epochs
   - LoRA rank/alpha

### Evolution Constraints

- Model must be <8B parameters
- Must run on AWS Trainium (trn1.2xlarge)
- No external search at inference time
- Pure language model inference only

## Directory Structure

```
global-chess-challenge-2025/
├── python/
│   ├── evaluate.py       # Local ACPL evaluation
│   ├── generate_data.py  # Training data generation
│   └── train.py          # Fine-tuning script
├── training/
│   ├── configs/          # Training configurations
│   └── datasets/         # Curated training data
├── data/
│   ├── games/            # Source game databases
│   └── puzzles/          # Tactical puzzles
├── mutations/            # Evolved variants
└── .evolve-sdk/          # Evolution state
```

## Local Development

```bash
# Clone starter kit
git clone https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit

# Install dependencies
pip install -r requirements.txt
pip install python-chess stockfish vllm

# Run local evaluation
python local_evaluation.py
```

## Submission

```bash
# Submit to AIcrowd
./aicrowd_submit.sh
```

## Evolution Workflow

```bash
# Run evolution on training strategy
/evolve-ml chess training strategy for <8B LLM --budget 50k

# The evolution will:
# 1. Generate training data variants
# 2. Fine-tune models with different configs
# 3. Evaluate ACPL against Stockfish
# 4. Select and crossover best strategies
```

## Key Resources

- Starter Kit: https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit
- AIcrowd Forum: https://www.aicrowd.com/challenges/global-chess-challenge-2025/discussions
- Stockfish: https://stockfishchess.org/
- python-chess: https://python-chess.readthedocs.io/

## Notes for NVIDIA Employee

This competition runs on AWS Trainium, not NVIDIA GPUs. However:
- Local development can use NVIDIA GPUs
- Evolution/hyperparameter search can run on NVIDIA hardware
- Only final inference submission requires Trainium
- Great showcase of cross-platform ML optimization

## Naming Convention Warning

**CRITICAL: Model versions ≠ Studio names**

- **Model versions**: V4, V5, V6 = iterations of our chess LLM training approach
- **Studio names**: `chess-llm-v4`, `chess-llm-v5` = Lightning.ai cloud machines

These are UNRELATED. Don't confuse them:
- V6 model training might run on studio `chess-llm-v5`
- Studio `chess-llm-v4` has nothing to do with model V4

When discussing, always clarify:
- "Model V6" or "V6 training" = our model version
- "Studio v4" or "chess-llm-v4" = the cloud machine
