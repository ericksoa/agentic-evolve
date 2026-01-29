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

## State of the Art (Jan 2025)

### DeepGEMM (DeepSeek)
**Source:** https://github.com/deepseek-ai/DeepGEMM

- **Performance:** 1550 TFLOPS on H800 (~95% of peak FP8)
- **Architecture:** CUDA-based, JIT compilation
- **Features:** FP8 with fine-grained scaling (per-128-channel)
- **Hardware:** SM90 (Hopper) and SM100 (Blackwell) only
- **Key insight:** Groups only M-axis (N,K fixed for MoE experts)

This is essentially THE reference implementation. Years of work by DeepSeek team.

### PyTorch Labs Persistent Kernel
**Source:** https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/

- **Performance:** 2.62x over naive loop on H100
- **Dtype:** BF16 only (FP8 listed as "future work")
- **Key innovations:**
  1. Persistent CTAs (single wave, no launch overhead)
  2. Cache-aware tile scheduling (GROUP_SIZE_M for L2 reuse)
  3. Dynamic TMA descriptors for expert weight access

### TensorRT-LLM (NVIDIA)
**Source:** https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html

- **Performance:** 5.5x speedup on B200 (368 TPS)
- **Features:** Fused operations (LocalReduction + AllReduce + RMSNorm + Quantization)
- **Hardware tricks:** PDL, NVSwitch oneshot AllReduce

### vLLM fused_moe
- Integrating DeepGemm (PR #13932)
- Currently ~2x slower than DeepGemm at TP=1
- Solid Triton baseline with multiple quantization paths

## Why This Problem Resists Evolutionary/GA Approaches

This project revealed important lessons about the limits of genetic algorithm / evolutionary optimization:

| Characteristic | Good for GA | MoE Kernels |
|----------------|-------------|-------------|
| Large solution space | ✓ | ✗ (constrained by hardware) |
| Gradual fitness gradient | ✓ | ✗ (cliff edges) |
| Domain-agnostic mutations | ✓ | ✗ (needs TMA, tensor cores) |
| Baseline far from optimal | ✓ | ✗ (already at 95% peak) |

**Key barriers:**
1. Architecture-specific intrinsics (TMA descriptors, warp specialization)
2. Discrete performance cliffs (wrong block size = 10x slower)
3. Experts already at hardware limits
4. Deep systems knowledge required (L2 cache, memory coalescing)

**Conclusion:** Some domains require human insight into hardware architecture, not just compute. GA works well for code golf and algorithm discovery with smooth fitness landscapes, but not for GPU kernel optimization where the solution space is heavily constrained by hardware realities.

## Remaining Opportunities

1. **Triton FP8 persistent kernel** - PyTorch Labs says this is "future work"
2. **Decode-specific optimization** - SplitK + persistent for M=1-8 tokens
3. **Better tooling** - Auto-selection between DeepGemm/Triton based on shape
