# MoE Triton Kernel Optimization

**Goal:** Explore MoE kernel optimization techniques from recent research

**Target:** DeepSeek-V3 style architectures (256 experts, top-8 routing)

**Status:** Implementation complete, benchmarking in progress

## Quick Start

```bash
# Test correctness
python test_moe.py

# Run benchmark
python test_moe.py --bench

# Full benchmark suite
cd benchmarks && python benchmark_moe.py --mode test
```

## Background

MoE (Mixture of Experts) models like DeepSeek-V3 route tokens dynamically to different expert networks. This project explores optimization techniques from recent research papers.

### Potential Inefficiencies in Naive Implementations

| Issue | Description |
|-------|-------------|
| Padding overhead | Tokens padded to fixed expert capacity |
| Cache locality | Weight matrices may not stay in cache |
| Expert load imbalance | Some experts get more tokens than others |

## Optimization Techniques Explored

### 1. Padding-Free Token Buffers
- **Source:** X-MoE ([arXiv:2508.13337](https://arxiv.org/abs/2508.13337))
- **Approach:** Sort tokens by expert, process variable-length batches
- **Claimed benefit:** Up to 5.15x over DeepSpeed-TED (in their specific setup)
- **Note:** Actual speedup depends heavily on workload characteristics

### 2. Column-Major GEMM Scheduling
- **Source:** [PyTorch Labs MoE Blog](https://pytorch.org/blog/accelerating-moe-model/)
- **Approach:** Reorder computation for better L2 cache utilization
- **Claimed benefit:** Up to 4.4x on H100 for specific matrix shapes
- **Note:** Benefits primarily skinny matrices (small batch, large hidden)

### 3. SplitK Work Decomposition
- **Source:** PyTorch Labs
- **Approach:** Split K-dimension across thread blocks
- **Claimed benefit:** 18-20% for decode workloads
- **Note:** Adds reduction overhead, only helps small batches

## Important Caveats

1. **Paper claims vs reality:** Published speedups are often measured against specific baselines under ideal conditions. Real-world gains may differ.

2. **Workload dependence:** Optimizations have different impacts for:
   - Decode (1-32 tokens): Kernel launch overhead dominates
   - Prefill (128-2048 tokens): Memory bandwidth limited

3. **Hardware dependence:** Results on T4/L40S may not reflect H100/H200 behavior.

4. **Baseline matters:** Comparing against naive PyTorch loops vs optimized CUDA kernels gives very different speedup numbers.

## Files

```
kernels/
  baseline_moe.py      # Reference implementation (naive loop)
  colmajor_moe.py      # Column-major scheduling exploration
  splitk_moe.py        # SplitK exploration
  padding_free_moe.py  # Padding-free token buffers
  optimized_moe.py     # Combined approach

benchmarks/
  benchmark_moe.py     # Benchmark suite

research/
  NOTES.md             # Technical notes from papers
```

## Current Framework Landscape

| Framework | Notes |
|-----------|-------|
| vLLM | Triton fused MoE, well-optimized |
| SGLang | Similar to vLLM approach |
| PyTorch Labs | Research demos, not production |
| X-MoE | Academic paper, specific setup |

## References

- [Accelerating MoE Model Inference (PyTorch Labs)](https://pytorch.org/blog/accelerating-moe-model/)
- [X-MoE (arXiv:2508.13337)](https://arxiv.org/html/2508.13337v1)
- [vLLM Fused MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
