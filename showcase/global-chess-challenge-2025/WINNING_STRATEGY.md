# Winning Strategy for Global Chess Challenge 2025

## Current Status (Baseline)
- **ACPL (legal only)**: ~200-300 (amateur level, ~1400-1500 Elo)
- **Legal rate**: 96%
- **Training data**: 30K examples, unfiltered high-Elo games

## The Key Insight

**Train ONLY on moves that match Stockfish's recommendations.**

Current training data includes all human moves - including mistakes, blunders, and suboptimal choices. The model learns these bad habits.

By filtering to only include moves where the human played what Stockfish would recommend, we train the model on "correct" chess, not "human" chess.

## Estimated Impact

| Stockfish Filter | Acceptance Rate | Expected ACPL | Estimated Elo |
|------------------|-----------------|---------------|---------------|
| None (current)   | 100%            | 200-300       | 1400-1500     |
| SF top-10        | ~60%            | 100-150       | 1700-1900     |
| SF top-3         | ~30%            | 80-100        | 1900-2100     |
| SF top-1         | ~15%            | 50-80         | 2100-2300     |

## Winning Configuration (Hypothesis)

```python
DataConfig(
    min_elo=2400,           # Higher Elo = fewer blunders
    stockfish_top_n=3,      # Human move in SF top-3
    max_centipawn_loss=50,  # Skip moves with CPL > 50
    epochs=4,               # More training
    lora_rank=128,          # Larger adapter
)
```

## Execution Plan

### Phase 1: Data Extraction ($0, ~2 hours)
```bash
# Download fresh PGN data
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# Extract with SF filtering
python extract_filtered_data.py \
    --pgn lichess_db_standard_rated_2024-01.pgn \
    --min-elo 2400 \
    --sf-top 3 \
    --max-cpl 50 \
    --max-examples 50000 \
    --output training_sf3_50k.jsonl
```

### Phase 2: Training (~$15-20, ~6 hours)
```bash
# Train on Lightning.ai
python run_lightning_v3.py --data training_sf3_50k.jsonl --epochs 4 --lora-rank 128
```

### Phase 3: Evaluation ($0, ~30 min)
```bash
# Evaluate ACPL
python evaluate_final_model.py --positions 100 --depth 15
```

### Phase 4: Iterate (if needed)
Based on results, adjust:
- SF filter strictness (top-1, top-3, top-5)
- Elo threshold
- Training epochs
- LoRA rank

## Budget Breakdown

| Item | Cost | Time |
|------|------|------|
| Data extraction | $0 | 2 hours |
| Training run 1 | $15-20 | 6 hours |
| Training run 2 | $15-20 | 6 hours |
| Training run 3 | $15-20 | 6 hours |
| **Total** | **$45-60** | **~20 hours** |

With $200-300 budget, we can run 10-15 experiments.

## Alternative Strategies (if SF filtering isn't enough)

1. **Larger model**: Qwen2.5-7B instead of 3B
2. **More data**: 100K+ examples
3. **Curriculum learning**: Easy positions first
4. **Ensemble**: Average predictions from multiple models
5. **Self-play refinement**: Generate data from model playing itself

## Success Criteria

- **Minimum**: ACPL < 150 (competitive entry)
- **Target**: ACPL < 100 (top 10)
- **Stretch**: ACPL < 75 (top 3)
