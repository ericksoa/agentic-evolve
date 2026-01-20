# Global Chess Challenge 2025 - Evolve SDK Project

## CRITICAL: Read Memory EFFICIENTLY (Don't Context Dump)

The memory file (`.evolve-sdk/evolve_chess_training_strategy/memory.json`) is **40K+ tokens**. DO NOT read it all at once.

### How to Read Memory Based on Task

**For SUBMISSIONS:**
```bash
# 1. FIRST: Read DRIs (required before ANY submission)
grep -A 50 '"dont_repeat_incidents"' memory.json | head -100

# 2. List existing HuggingFace repos (USE THE API, not grep!)
python3 -c "from huggingface_hub import HfApi; [print(m.id) for m in HfApi().list_models(author='ericksoa')]"

# 3. Check recent submissions
grep -B 2 -A 20 '"submission_id"' memory.json | tail -60
```

**For TRAINING STATUS:**
```bash
grep -A 30 '"active_processes"' memory.json
grep -A 20 '"rejection_sampling_training"' memory.json
```

**For MODEL INVENTORY:**
```bash
grep -A 40 '"model_inventory"' memory.json
grep -A 10 '"hf_repos"' memory.json
```

**For EVALUATION RESULTS:**
```bash
grep -A 15 '"model_evaluations"' memory.json
```

### Priority Order When Starting a Task

1. **READ DRIs FIRST** - Always. No exceptions.
2. **Check what already exists** - Use APIs (HuggingFace, Lightning) not memory grep
3. **Read only relevant memory sections** - Use targeted grep, not full file reads
4. **Use proper tools** - `huggingface_hub` Python API, not curl workarounds

## CRITICAL: Golden Template for All Models

**NEVER write your own chat_template.jinja. ALWAYS use the golden template.**

- **Golden template**: `golden_chat_template.jinja` (copied from working ericksoa/chess-v6-aicrowd)
- **Why**: Simplified templates don't inject system prompt when AIcrowd sends no system message
- **Script**: `checkpoint_pipeline.py` uses this automatically via `CHESS_CHAT_TEMPLATE_FILE`

If you see a chat_template.jinja that's shorter than 2000 chars, it's probably broken.

## CRITICAL: DRI Instructions Override Everything

The `dont_repeat_incidents` array contains HARD-LEARNED LESSONS from wasted submissions.

**BEFORE any submission or major action:**
1. Read the DRIs section
2. Verify your planned action doesn't repeat a past incident
3. Follow the verification commands in each DRI
4. For new HF repos: verify chat_template.jinja matches golden template

## CRITICAL: Use Evolve SDK Memory System

**DO NOT create random .md files to store learnings, DRIs, roadmaps, or project state.**

Always use the evolve-sdk memory system:
- **Memory file**: `.evolve-sdk/evolve_chess_training_strategy/memory.json`
- **DRIs go in**: `memory.json` → `dont_repeat_incidents` array
- **Learnings go in**: `memory.json` → `learnings` object
- **Submissions go in**: `memory.json` → `submissions` array
- **Roadmaps go in**: `memory.json` → `comprehensive_roadmap` object
- **Credentials**: Always check `.env` file FIRST for API keys

**NEVER create these files** (use memory.json instead):
- IMPROVEMENT_PLAN.md
- WORK_IN_PROGRESS.md
- WINNING_STRATEGY.md
- Any other .md files for plans/state

## CRITICAL: Current Best Results

**Target: <100 ACPL for competitive submission**

| Model | ACPL (AIcrowd verified) | Status |
|-------|-------------------------|--------|
| **V6 checkpoint-4000** | **202** | BEST (submission 307752) |
| Baseline | 71.9 | Competition baseline |
| Leader | 46.4 | Target to beat |

- **CRITICAL**: 78.1 ACPL was HALLUCINATED. It was never real.
- Our actual best verified AIcrowd result is **202 ACPL**
- We are WORSE than baseline (71.9 ACPL)
- Gap to leader: 155.6 ACPL

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
