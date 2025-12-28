# KV-Cache Eviction: 7% Improvement via Layer-Aware Scoring

This showcase demonstrates an evolved KV-cache eviction policy that achieves **7.07% improvement** over a hybrid baseline on attention reconstruction error.

## Results Summary

| Split | Baseline | Evolved | Improvement |
|-------|----------|---------|-------------|
| TRAIN | 0.4785 | 0.4447 | **7.07%** |
| VALID | 0.4796 | 0.4467 | **6.87%** |
| TEST  | 0.4792 | 0.4463 | **6.85%** |

**Consistent improvement across all splits indicates good generalization.**

---

## Why This Problem Matters

### The KV-Cache Memory Problem

Large Language Models (LLMs) during inference maintain a **key-value cache** storing attention keys and values for all previous tokens. For long-context models (100K+ tokens), this cache can consume **tens of gigabytes of GPU memory**.

**Real-world impact:**
- **Memory Limits**: A 70B parameter model with 128K context can require 100+ GB just for KV-cache
- **Throughput**: Memory bandwidth becomes the bottleneck for long sequences
- **Cost**: Larger GPUs or more GPUs needed per inference request

### The Eviction Challenge

When cache memory is constrained, we must **evict tokens** while minimizing information loss. The challenge: which tokens are safe to evict?

**Key observations from LLM attention research:**
1. **Attention Sinks**: First few tokens attract disproportionate attention (Xiao et al., 2023)
2. **Heavy Hitters**: Some tokens consistently receive high attention (H2O, Zhang et al., 2023)
3. **Layer Patterns**: Early layers have diffuse attention, late layers are focused (PyramidKV)
4. **Key Norm Outliers**: Tokens with large key norms may be noise (KnormPress, NVIDIA)

### Why Beating Baselines Matters

Existing approaches (StreamingLLM, H2O, SnapKV) use fixed heuristics. Our evolved scorer:
1. **Adapts to layer depth** - Different strategies for early vs late layers
2. **Combines multiple signals** - Recent attention, cumulative attention, key norms, position
3. **Discovered non-obvious relationships** - Layer-adaptive recency bonus

---

## The Evolution Journey

### Generation 1: Failed Strategies

| Mutation | Result | Status | Learning |
|----------|--------|--------|----------|
| `multiplicative` | +27-32% worse | REJECTED | Product collapses score range |
| `entropy-based` | +46% worse | REJECTED | Entropy alone is poor signal |
| `softmax-temperature` | +13% worse | REJECTED | Temperature scaling hurts |
| `complex-layer-adaptive` | +42% worse | REJECTED | Too many interacting terms |

**Key Learning**: Simple additive combinations work; multiplicative and complex formulas fail catastrophically.

### Generation 2: Breakthrough

| Mutation | Result | Status | Insight |
|----------|--------|--------|---------|
| **`layer_aware`** | **6.92%** | **CHAMPION** | PyramidKV-inspired layer weights |
| `weight_70_30` | 6.92% | accepted | 0.7/0.3 better than 0.6/0.4 |
| `key_norm_penalty` | ~6% | accepted | KnormPress-inspired penalty |
| `position_power_0.2` | ~5% | accepted | Weaker correction helps |

**Key Learning**: Layer-aware weighting is the key innovation. Early layers need different treatment than late layers.

### Generation 3: Crossovers

| Mutation | Result | Status | Insight |
|----------|--------|--------|---------|
| `layer_aware + key_norm` | 7.00% | improved | Signals are additive |
| `+ position_power` | 7.05% | improved | Layer-adaptive position helps |

### Generation 4: Final Champion

| Mutation | Result | Status | Insight |
|----------|--------|--------|---------|
| **`+ layer_aware_recency`** | **7.07%** | **CHAMPION** | Late layers need stronger recency |

**Final formula combines 4 layer-adaptive components.**

---

## The Winning Algorithm

```rust
fn score(&self, token: &TokenInfo) -> f64 {
    // Sink tokens always kept (attention sink phenomenon)
    if token.is_sink {
        return f64::MAX;
    }

    // Very recent tokens always kept
    if token.relative_pos < 4 {
        return 1e6 - token.relative_pos as f64;
    }

    // Layer-aware weighting (PyramidKV-inspired)
    // Early layers: diffuse attention -> favor recent
    // Late layers: focused attention -> balance recent/cumulative
    let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
    let recent_weight = 0.7 - 0.2 * layer_ratio;      // 0.7 → 0.5
    let cumulative_weight = 0.3 + 0.2 * layer_ratio;  // 0.3 → 0.5

    // Layer-adaptive position correction
    let position_power = 0.25 + 0.1 * layer_ratio;    // 0.25 → 0.35
    let position_factor = ((token.position + 1) / token.sequence_len).powf(position_power);

    // Layer-aware recency bonus
    let base_recency = 0.15 + 0.1 * layer_ratio;      // 0.15 → 0.25
    let recency_bonus = if token.relative_pos < 128 {
        base_recency * (1.0 - token.relative_pos / 128.0)
    } else {
        0.0
    };

    // Key norm penalty (KnormPress-inspired)
    let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

    // Combined score
    recent_weight * token.recent_attn
        + cumulative_weight * token.cumulative_attn * position_factor
        + recency_bonus
        - key_norm_penalty
}
```

### Key Innovations

1. **Layer-Aware Attention Weighting**
   - Early layers (layer 0): 70% recent, 30% cumulative
   - Late layers (layer 31): 50% recent, 50% cumulative
   - Rationale: Early layers have diverse attention patterns, late layers are more focused

2. **Layer-Adaptive Position Correction**
   - Power: 0.25 (early) → 0.35 (late)
   - Stronger correction for late layers where position bias is more pronounced

3. **Layer-Aware Recency Bonus**
   - Bonus: 0.15 (early) → 0.25 (late)
   - Late layers benefit more from recency since attention is focused

4. **Key Norm Penalty**
   - Penalize outlier tokens with large key norms
   - Capped at 2.0 to prevent excessive penalty

---

## Quick Start

### Prerequisites

- Rust toolchain (install via https://rustup.rs)

### Run the Benchmark

```bash
cd showcase/kv-cache-eviction/rust
cargo build --release
cargo run --release --bin micro_bench
```

### Expected Output

```
Summary:
============================================================
                               TRAIN           VALID            TEST
Hybrid baseline               0.4785          0.4796          0.4792
Evolved (layer-aware)         0.4447          0.4467          0.4463

Improvement (lower error = better):
  TRAIN: 7.07%
  VALID: 6.87%
  TEST:  6.85%
```

---

## Technical Details

### The Eviction Scoring Problem

Given a token in the KV-cache, assign an importance score. Higher score = keep, lower score = evict.

**Available information per token:**
- `position`: Absolute position in sequence (0-indexed)
- `relative_pos`: Distance from current generation position
- `recent_attn`: Sum of attention over last 32 queries
- `cumulative_attn`: Sum of attention over all past queries
- `key_norm`: L2 norm of the key vector
- `layer_idx`: Which layer (0 to num_layers-1)
- `is_sink`: Whether this is a sink token (position < 4)

### Metric: Attention Reconstruction Error

```
error = (attention_to_evicted_tokens) / (total_attention)
```

Measures the fraction of attention "lost" by evicting tokens. Lower is better.

Evaluated at three compression ratios: 25%, 50%, 75% cache retention.

### Baseline: Hybrid

The hybrid baseline combines:
- Recent attention (0.6 weight)
- Position-corrected cumulative attention (0.4 weight)
- Recency bonus for tokens within 128 positions
- Position factor: `(pos/seq_len)^0.3`

### Synthetic Attention Patterns

Patterns are generated to mimic real LLM attention:
- Attention sinks (first 4 tokens)
- Recency bias (recent tokens get more attention)
- Information-dense tokens (random 10% get 3x attention)
- Layer-dependent focus (later layers more peaked)

---

## Evolution Statistics

| Metric | Value |
|--------|-------|
| Generations | 4 |
| Candidates Tested | ~20 |
| Candidates Accepted | 6 |
| Stop Reason | User checkpoint for external feedback |

**Evolution can be resumed with `/evolve --resume`**

---

## File Structure

```
showcase/kv-cache-eviction/
├── README.md           # This file
└── rust/
    ├── Cargo.toml      # Build configuration
    └── src/
        ├── lib.rs          # Core types and evaluation
        ├── baselines.rs    # StreamingLLM, H2O, SnapKV, PyramidKV, etc.
        ├── evolved.rs      # Champion eviction scorer (7.07%)
        ├── benchmark.rs    # Full benchmark (slow)
        ├── micro_bench.rs  # Fast benchmark for iteration
        └── generate_data.rs # Synthetic attention pattern generator
```

---

## Comparison to Published Methods

| Method | Approach | Our Improvement |
|--------|----------|-----------------|
| StreamingLLM | Keep sinks + recent window | Evolved adds attention-based scoring |
| H2O | Cumulative attention (Heavy Hitters) | Evolved adds layer-awareness + position correction |
| SnapKV | Recent attention window | Evolved balances recent/cumulative by layer |
| PyramidKV | Layer-wise budget allocation | Evolved integrates layer-awareness into scoring |
| KnormPress | Key norm based eviction | Evolved uses key norm as penalty term |

---

## Future Work

This is an active evolution. Potential directions:
- Token-type awareness (special tokens, punctuation)
- Attention entropy signals
- Longer sequence benchmarks
- Real model validation (beyond synthetic patterns)

**To continue evolution:** Resume from `.evolve/kv-cache-eviction/evolution.json`

---

## References

- StreamingLLM: "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023)
- H2O: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (Zhang et al., 2023)
- SnapKV: "SnapKV: LLM Knows What You are Looking for Before Generation" (Li et al., 2024)
- PyramidKV: "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling" (Cai et al., 2024)
- KnormPress: NVIDIA's key-norm based compression approach

---

## Deterministic Reproduction

- [x] No external data files required (synthetic generation with fixed seeds)
- [x] No network requests
- [x] Fixed random seeds for reproducibility
- [x] Same results every run
