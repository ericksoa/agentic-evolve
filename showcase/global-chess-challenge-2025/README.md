# Global Chess Challenge 2025 - Evolve SDK Showcase

Training a <8B parameter LLM to play competitive chess using evolutionary optimization of training strategies.

**Competition**: [AIcrowd Global Chess Challenge 2025](https://www.aicrowd.com/challenges/global-chess-challenge-2025)

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
| 307762 | Qwen 3B 12K | *Pending evaluation* |

**Key Insight**: Local evaluation (192 ACPL) ≠ Platform evaluation (881 ACPL). The vLLM + Trainium stack handles EOS tokens differently.

---

## Current Status (Jan 16, 2026)

### Leaderboard Context

| Position | ACPL | Notes |
|----------|------|-------|
| **Leader** | 46.4 | Target to beat |
| **Baseline** | 71.9 | Competition baseline |
| **Our Best (local)** | 186.1 | Qwen 3B checkpoint-12000 |
| **Gap** | 4.0x | Work to do |

### Active Training

| Model | Machine | Progress | Eval Loss | Status |
|-------|---------|----------|-----------|--------|
| **Qwen2.5-3B** | chess-llm-v4 | 14.4% (12K/83K) | 0.420 | Training |
| **V6 r128 (7B)** | chess-llm-v5 | ~14% | - | Training |

### Latest Submission

**307762** - Qwen 3B checkpoint at 14% training
- Local ACPL: 186.1 (legal only)
- Legal move rate: 98%
- Status: Awaiting AIcrowd evaluation

---

## Key Discoveries

### 1. Data Quality > Quantity
61.5% of training positions have multiple equally-good moves. Training on these creates noisy gradients.

**Solution**: Filter for "winning moves" - positions where best move is clearly superior (>50 centipawn gap).

### 2. DPO Doesn't Work for Chess
Direct Preference Optimization made performance **worse** (+6.2 ACPL). Offline preferences don't capture chess dynamics.

**Pivot**: Plan GRPO (online RL) with real-time Stockfish reward.

### 3. Early Checkpoints Are Broken
At 7-14% training, models produce garbage after the move:
```
<uci_move>g8f6</uci_move>SmartyHeaderCode rematch FEN:...
```
Need to wait for 50%+ training before output format stabilizes.

### 4. Platform != Local
V6 model: 192 ACPL locally → 881 ACPL on AIcrowd

The vLLM + Trainium stack has different EOS token handling. What works locally may fail on the platform.

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

## Training Data

| Dataset | Size | Purpose |
|---------|------|---------|
| `training_mega_449k.jsonl` | 449K positions | Main SFT training |
| `training_sf3_500k.jsonl` | 500K positions | SF depth 3 annotations |
| `training_elite_30k.jsonl` | 30K positions | High-quality subset |
| `training_endgames.jsonl` | 7K positions | Endgame focus |

**Total**: 9.2 GB across 37 files

---

## Model Architecture

**Base**: Qwen2.5-3B (also experimenting with 7B)

**Fine-tuning**: LoRA
- Rank: 128, Alpha: 256
- Target: All attention + MLP modules
- Dropout: 0.05

**Prompt Format**:
```
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Moves: a7a6, b7b5, c7c5, d7d5, e7e5, g8f6, ...
Best?
```

**Output Format**:
```
<uci_move>e7e5</uci_move>
<rationale>Controls center</rationale>
```

---

## Infrastructure

### Cloud Training
- **chess-llm-v4**: Qwen 3B on Lightning.ai L4 GPU
- **chess-llm-v5**: V6 r128 (7B) on Lightning.ai L4 GPU

### Local Development
- Apple Silicon M3 with MPS backend
- Stockfish 17.1 for evaluation
- Local ACPL testing before submission

### Submission Pipeline
```
Train (Lightning) → Download checkpoint → Merge LoRA → Upload HuggingFace → Submit AIcrowd
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
| Jan 10-11 | Ran 6 generations of evolution |
| Jan 12 | First real submission attempt |
| Jan 13 | Reality check - actual ACPL was 208, not 85 |
| Jan 13-14 | Built 449K mega-dataset, started cloud training |
| Jan 14-15 | DPO experiments (failed), submission debugging |
| Jan 15 | Discovered 61.5% ambiguous data problem |
| Jan 16 | Qwen 3B checkpoint-12000 achieves 186 ACPL locally |

---

## References

- [Competition Page](https://www.aicrowd.com/challenges/global-chess-challenge-2025)
- [Starter Kit](https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit)
- [Stockfish](https://stockfishchess.org/)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [Lightning.ai](https://lightning.ai/)
