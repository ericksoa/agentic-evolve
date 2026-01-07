# MoE Kernel Optimization Research Notes

## Key Papers & Sources

### 1. PyTorch Labs - Column-Major GEMM Scheduling
**Source:** https://pytorch.org/blog/accelerating-moe-model/

**Key Insight:**
- Standard (row-major) GEMM: Iterate M fast, N slow
- Problem: For MoE, each expert processes few tokens (small M), large hidden dim (K)
- Weight matrices get evicted from cache before reuse

**Column-Major Solution:**
- Iterate N (output columns) fast, M (tokens) slow
- Weight columns stay in L2 cache across token batches
- Result: L1 hit rate +2,696%, L2 hit rate +254%
- Speedup: Up to 4.4x on H100

**Implementation:**
```python
# Standard: pid_m varies faster
pid_m = pid // num_n_blocks
pid_n = pid % num_n_blocks

# Column-major: pid_n varies faster
pid_n = pid % num_n_blocks
pid_m = pid // num_n_blocks
```

### 2. SplitK Work Decomposition
**Source:** PyTorch Labs MoE blog

**Problem:**
- Decode time: 1-8 tokens per expert
- Small M = few thread blocks = underutilized GPU

**SplitK Solution:**
- Split K dimension across multiple thread blocks
- Each block computes partial sum
- Final reduction to combine results
- Speedup: 18-20% on decode workloads

**Trade-off:**
- Extra memory for partial results
- Reduction overhead
- Only beneficial for small M (< 64 tokens)

### 3. X-MoE Padding-Free Token Buffers
**Source:** arXiv:2508.13337

**Problem:**
- Standard MoE pads all experts to max_tokens_per_expert
- With 256 experts, load imbalance = 30-50% wasted compute

**Padding-Free Solution:**
- Sort tokens by expert
- Variable-length expert batches
- No zero-padding overhead
- Speedup: 5.15x over DeepSpeed-TED

**Implementation:**
```python
# Sort tokens by expert
sort_indices = torch.argsort(expert_ids, stable=True)
sorted_tokens = tokens[sort_indices]

# Process each expert with exact token count
for expert in range(num_experts):
    start, end = expert_offsets[expert:expert+2]
    if start == end:
        continue
    expert_tokens = sorted_tokens[start:end]
    # No padding!
```

## DeepSeek-V3 Architecture

- **Experts:** 256
- **Top-K:** 8
- **Hidden dim:** 7168
- **Intermediate dim:** 2048
- **Shared experts:** 1

Key challenge: With 256 experts and top-8, each expert sees ~3% of tokens on average.
This creates highly variable batch sizes per expert, making padding very wasteful.

## Benchmark Configurations

### Decode (latency-critical)
- Batch: 1-32, Seq: 1 (single token generation)
- Challenge: Very small M, need SplitK

### Prefill (throughput-critical)
- Batch: 1-8, Seq: 128-2048
- Challenge: Variable expert load, need padding-free

## Memory Layout Considerations

Expert weights: `[num_experts, hidden_dim, intermediate_dim]`
- Gate: 256 * 7168 * 2048 = 3.75B params
- Up: 256 * 7168 * 2048 = 3.75B params
- Down: 256 * 2048 * 7168 = 3.75B params
- Total: ~11.25B params = 22.5GB in FP16

Token routing:
- Each token selects top-8 experts
- Routing weights: softmax over top-8

## Implementation Status

| Optimization | File | Status | Notes |
|-------------|------|--------|-------|
| Baseline (naive loop) | baseline_moe.py | Done | Reference implementation |
| Column-major GEMM | colmajor_moe.py | Done | Triton autotune |
| SplitK | splitk_moe.py | Done | For decode |
| Padding-free | padding_free_moe.py | Done | Token buffer class |
| Combined | optimized_moe.py | Done | All optimizations |

## Expected Speedups

Based on literature:
- Column-major: 1.5-4.4x (depends on batch size)
- SplitK: 1.18-1.20x (decode only)
- Padding-free: 1.3-5.15x (depends on load imbalance)
- **Combined: 2-4x target**

## Next Steps

1. Test correctness on T4
2. Profile memory bandwidth utilization
3. Benchmark on H200 with DeepSeek-V3 configs
4. Compare vs vLLM fused_moe kernel
