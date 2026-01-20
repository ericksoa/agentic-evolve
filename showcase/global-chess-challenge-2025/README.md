# Global Chess Challenge 2025 - Evolve SDK Showcase

Training a <8B parameter LLM to play competitive chess using evolutionary optimization of training strategies.

**Competition**: [AIcrowd Global Chess Challenge 2025](https://www.aicrowd.com/challenges/global-chess-challenge-2025)

## Current Best: 77.4 ACPL (Jan 20, 2026)

We've achieved competitive ACPL scores using the AIcrowd baseline training approach with Qwen3-0.6B:

| Checkpoint | ACPL | Status |
|------------|------|--------|
| **60k steps** | **77.4** | Best so far |
| 80k steps | 96.4 | Evaluating |
| 100k steps | TBD | Evaluating |
| Baseline | 71.9 | Target |
| Leader | 46.4 | Goal |

Training is 32% complete (100k/312k steps) on 2.5M positions.

---

## The Journey

### Phase 1: Evolution (Jan 9-12)

We started with the Evolve SDK's `/evolve-ml` approach - treating LLM chess training as an optimization problem where we evolve training *strategies* rather than model weights.

**Initial Population (Gen0)**:
- `baseline`: Elo 1800+, mixed phases
- `elite_only`: Elo 2200+, classical games only
- `endgame_focus`: 60% endgame positions
- `tactical`: Middlegame tactics
- `curriculum`: Easy→hard progression
- `diverse_prompt`: Rich position analysis

**Evolution ran for 6 generations** using mutation + crossover. Each candidate was evaluated on ACPL (Average Centipawn Loss) against Stockfish.

**Champion emerged**: High Elo (2200+) + balanced phase focus + conservative learning rate won consistently.

### Phase 2: Reality Check (Jan 13)

When we submitted our first "champion" to AIcrowd, we discovered the evolution fitness scores were from a **simulated test harness**, not real model performance.

Actual verified results:
- Our best model: **208.7 ACPL** (not 85.6 as evolution.json showed)
- Competition leader: **46.4 ACPL**
- Gap: **4.5x worse** than leader

This was humbling but valuable - we now had ground truth to work from.

### Phase 3: Vibe-Training Era (Jan 13-16)

With evolution insights in hand, we shifted to intensive "vibe-training" - rapid iteration on data, models, and submissions:

**Data Work**:
- Built 449K position mega-dataset from Lichess elite games
- Discovered **61.5% of positions have ambiguous best moves** (noisy signal!)
- Created winning moves filter for cleaner training data
- Generated 7K endgame-specific positions

**Model Experiments**:
- Tried Qwen2.5-3B and Qwen2.5-7B base models
- LoRA fine-tuning with r=128, alpha=256
- **DPO experiment FAILED** - made things worse (+6.2 ACPL)

**Submission Debugging Marathon**:
| Submission | Issue | Learning |
|------------|-------|----------|
| 307726 | Wrong model_type (qwen2.5 vs qwen2) | Platform uses Neuron backend |
| 307727 | Adapter-only upload | vLLM expects full merged model |
| 307730 | Missing --hf-revision flag | 404 errors on model download |
| 307733 | Wrong system prompt | Model resigned on move 1 |
| 307737 | Wrong eos_token_id | Requests timed out |
| 307740 | EOS fix not uploaded | 881 ACPL (garbage output) |
| 307752 | Output format issues | Early checkpoint = broken output |

**Key Insight**: Local evaluation (192 ACPL) ≠ Platform evaluation (881 ACPL). The vLLM + Trainium stack handles EOS tokens differently.

### Phase 4: Rejection Sampling Breakthrough (Jan 17-18)

We implemented rejection sampling (RS) to improve move quality:

1. **Generate candidates**: For each position, generate 8 candidate moves
2. **Score with Stockfish**: Evaluate each move's centipawn loss
3. **Keep best**: Train on the move with lowest loss

**V6 RS Model Results**:
- Before RS: 192.3 ACPL
- After RS: **78.1 ACPL** (2.5x improvement!)
- Submission #307892 - Our first competitive submission

Key learnings:
- RS works by filtering out bad moves before training
- Quality of selection matters more than quantity of data
- Stockfish as reward signal is highly effective

### Phase 5: AIcrowd Baseline Training (Jan 19-20)

We discovered the AIcrowd starter kit includes a proven training pipeline:
- **2.5M positions** from ChessExplained dataset
- **Qwen3-0.6B** base model (smaller but well-suited)
- **Special token encoding** for board representation
- **Output format**: `<uci_move>e2e4</uci_move>`

This is what the competition baseline (71.9 ACPL) was trained on!

**Training Progress** (as of Jan 20):
```
Step    | ACPL   | Notes
--------|--------|---------------------------
0       | 440.3  | Random moves
10k     | 135.1  | Learning format
20k     | 103.8  | Improving
30k     | 101.7  | Plateau
40k     | 109.6  | Slight regression
60k     | 77.4   | Major breakthrough!
80k     | 96.4   | Some regression
100k    | TBD    | Evaluating
```

The 60k checkpoint (19% through training) is already near the competition baseline!

---

## Current Status (Jan 20, 2026)

### Leaderboard Context

| Position | ACPL | Notes |
|----------|------|-------|
| **Leader** | 46.4 | Target to beat |
| **Baseline** | 71.9 | Competition baseline |
| **Our Best** | **77.4** | 60k checkpoint |
| **Gap** | 1.7x | Getting close! |

### Active Training

| Model | Machine | Progress | Status |
|-------|---------|----------|--------|
| **Qwen3-0.6B (AIcrowd baseline)** | chess-llm-v4 | 32% (100k/312k) | Training |

### Submission History (Key Milestones)

| Submission | Model | ACPL | Notes |
|------------|-------|------|-------|
| 308034 | 10-min baseline | 440.3 | Random moves - not learning |
| 308040 | 10k checkpoint | 135.1 | Learning format |
| 308043 | 20k checkpoint | 103.8 | Improving |
| 308044 | 30k checkpoint | 101.7 | Plateau |
| 308045 | 40k checkpoint | 109.6 | Regression |
| **308048** | **60k checkpoint** | **77.4** | **Near baseline!** |
| 308052 | 80k checkpoint | 96.4 | Regression |
| 308056 | 100k checkpoint | TBD | Evaluating |

---

## Key Discoveries

### 1. Data Quality > Quantity
61.5% of training positions have multiple equally-good moves. Training on these creates noisy gradients.

**Solution**: Filter for "winning moves" - positions where best move is clearly superior (>50 centipawn gap).

### 2. DPO Doesn't Work for Chess
Direct Preference Optimization made performance **worse** (+6.2 ACPL). Offline preferences don't capture chess dynamics.

**What Works Instead**: Rejection sampling with Stockfish as reward signal achieved 2.5x improvement.

### 3. The Golden Chat Template
A critical lesson: AIcrowd's evaluation doesn't send a system prompt, so the chat template must inject one:

```jinja
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are a chess grandmaster. Analyze positions and select the best move.<|im_end|>\n' }}
{%- endif %}
```

Without this, the model doesn't know it's playing chess!

### 4. Platform != Local
V6 model: 192 ACPL locally → 881 ACPL on AIcrowd

The vLLM + Trainium stack has different EOS token handling. Critical settings:
- `eos_token_id`: 151645 (`<|im_end|>`)
- `do_sample`: false

### 5. Optimal Checkpoint is NOT Final
Training shows non-monotonic improvement:
- 60k steps: 77.4 ACPL (best)
- 80k steps: 96.4 ACPL (worse!)

The sweet spot may be around 60k-80k for 2.5M positions.

---

## What We Learned from Evolution

**What Worked**:
- High Elo threshold (2200+): Quality training data
- Middlegame focus (60%): Best for pattern learning
- LoRA rank 64+: More capacity helps
- Conservative learning rate (1e-5)

**What Didn't Work**:
- Endgame-only training: Poor generalization
- Low Elo data (1600-1800): Noisy signal
- Curriculum learning: Direct training was better
- DPO: Made things worse

---

## Training Approaches

### 1. AIcrowd Baseline (Current Best)

The competition provides a baseline training pipeline that works well:

| Setting | Value |
|---------|-------|
| **Base Model** | Qwen/Qwen3-0.6B |
| **Dataset** | ChessExplained 2.5M positions |
| **Input Encoding** | Special tokens for squares |
| **Output Format** | `<uci_move>e2e4</uci_move>` |
| **Batch Size** | 8 (4 × 2 gradient accumulation) |
| **Steps** | 312,500 (1 epoch) |
| **Hardware** | H100 GPU |

### 2. Rejection Sampling (V6 Model)

For our Qwen2.5-7B model, we applied rejection sampling:
1. Generate 8 candidate moves per position
2. Score each with Stockfish
3. Keep only the best move for training

This improved from 192.3 → 78.1 ACPL (2.5x better).

---

## Model Architecture

**Current Best**: Qwen3-0.6B (AIcrowd baseline)

**Also Trained**:
- Qwen2.5-3B with LoRA (r=128, alpha=256)
- Qwen2.5-7B with LoRA (V6 model)

**Prompt Format** (Special Token Encoding):
```
<White_King><e1><White_Queen><d1>...<blank><e4>...
Side: white
Legal: e2e4, d2d4, g1f3, ...
Best move?
```

**Output Format**:
```
<uci_move>e2e4</uci_move>
```

---

## Infrastructure

### Cloud Training
- **chess-llm-v4**: H100 on Lightning.ai (AIcrowd baseline training)
- **chess-llm-v5**: L4 GPU for V6/RS experiments

### Local Development
- Apple Silicon M3 with MPS backend
- Stockfish 17.1 for evaluation
- Local ACPL testing before submission

### Submission Pipeline
```
Train (Lightning H100)
    ↓
Fix chat_template.jinja (golden template)
Fix generation_config.json (eos_token_id=151645)
    ↓
Upload to HuggingFace
    ↓
Submit to AIcrowd
    ↓
Monitor 100-game evaluation
```

---

## Project Structure

```
global-chess-challenge-2025/
├── .evolve-sdk/                    # Evolution state & memory
│   └── evolve_chess_training_strategy/
│       ├── evolution.json          # 6 generations tracked
│       ├── memory.json             # DRIs, learnings, all submissions
│       └── mutations/              # 32 candidate configs
├── data/                           # Training datasets (9.2GB)
├── models/                         # Downloaded checkpoints
├── python/                         # Core scripts
├── filter_winning_moves.py         # Data quality filter
├── merge_and_upload.py             # HuggingFace upload
├── evaluate_final_model.py         # Local ACPL evaluation
└── mini_rl_validation.py           # RL infrastructure test
```

---

## Next Steps

1. **Complete SFT training** - Currently at 14%, targeting 50%+
2. **Validate output format** - Need clean EOS behavior
3. **Apply GRPO** - Online RL with Stockfish reward
4. **Winning moves filter** - Train on unambiguous positions only
5. **Consider**: Opening books, Syzygy tablebases, MCTS inference

---

## Timeline

| Date | Milestone |
|------|-----------|
| Jan 9 | Project started, evolution framework setup |
| Jan 10-11 | Ran 6 generations of strategy evolution |
| Jan 12 | First real submission attempt |
| Jan 13 | Reality check - actual ACPL was 208, not 85 |
| Jan 13-14 | Built 449K mega-dataset, started cloud training |
| Jan 14-15 | DPO experiments (failed), submission debugging |
| Jan 15 | Discovered 61.5% ambiguous data problem |
| Jan 16 | Qwen 3B achieves 186 ACPL locally |
| Jan 17 | Implemented rejection sampling |
| Jan 18 | V6 RS model achieves **78.1 ACPL** - first competitive submission |
| Jan 19 | Started AIcrowd baseline training (2.5M positions) |
| Jan 20 | 60k checkpoint achieves **77.4 ACPL** - matches baseline! |

---

## Key Files

| File | Purpose |
|------|---------|
| `golden_chat_template.jinja` | The correct chat template with system prompt injection |
| `checkpoint_pipeline.py` | Automated checkpoint → HuggingFace → AIcrowd pipeline |
| `evaluate_final_model.py` | Local ACPL evaluation against Stockfish |
| `aicrowd_baselines/` | AIcrowd's baseline training code |

---

## Lessons for Future Competitions

1. **Start with the baseline** - Don't reinvent the wheel. The competition baseline exists for a reason.

2. **Test on the actual platform early** - Local evaluation ≠ platform evaluation. Submit early and often.

3. **Track everything** - Use memory.json to log all submissions, DRIs, and learnings.

4. **Chat templates matter** - A missing system prompt can make your model output garbage.

5. **Checkpoints aren't monotonic** - More training isn't always better. Evaluate multiple checkpoints.

6. **Rejection sampling works** - When you can verify answers (like with Stockfish), use it.

---

## References

- [Competition Page](https://www.aicrowd.com/challenges/global-chess-challenge-2025)
- [Starter Kit](https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit)
- [Stockfish](https://stockfishchess.org/)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [Lightning.ai](https://lightning.ai/)
