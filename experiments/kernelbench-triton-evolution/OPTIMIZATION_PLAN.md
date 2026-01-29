# Triton Softmax Optimization Plan

## Problem Analysis

Current 3-pass kernel achieves only **0.66x PyTorch speed** due to:
1. **3 global memory reads** (max, sum, output passes)
2. **31µs kernel launch overhead** (vs PyTorch's 7.5µs)
3. **188 GB/s bandwidth** (vs PyTorch's 354 GB/s effective)

## Research Insights

### From Flash Attention (Dao et al.)
- Online softmax with running statistics (m_i, l_i)
- Rescaling: `acc = acc * exp(m_old - m_new)` when max updates
- Single pass with register storage

### From Triton Attention Anatomy (arXiv:2511.11581)
- Tiled softmax keeps values in shared memory/registers
- Q-Block optimization for data reuse
- Up to 589% speedup with proper memory hierarchy

### From Liger Kernel
- Kernel fusion reduces memory 60%
- In-place operations where possible
- Chunking for large sequences

## Implementation Strategy

### Phase 1: Single-Pass Online Softmax (Target: 1.5-2x PyTorch)

```
Algorithm:
1. For each block of input:
   - Load values into registers
   - Update running_max = max(running_max, block_max)
   - Rescale: running_sum *= exp(old_max - new_max)
   - Compute exp_vals = exp(vals - running_max)
   - Store exp_vals in registers (not global memory!)
   - running_sum += sum(exp_vals)
2. Single output pass:
   - Divide stored exp_vals by running_sum
   - Write to output
```

Key insight: Store intermediate exp values in **registers** to eliminate 2nd read.

### Phase 2: Vectorized Loads (Target: 2-3x)
- Use `tl.load` with `vectorize=True` for coalesced access
- Process 4 floats per thread (float4 loads)
- Align block sizes to warp size (32)

### Phase 3: Shared Memory Tiling (Target: 3-4x)
- For sequences > register capacity
- Tile into shared memory chunks
- Process tiles with register blocking

### Phase 4: Autotune Configurations
- BLOCK_SIZE: [256, 512, 1024, 2048]
- NUM_WARPS: [4, 8, 16]
- NUM_STAGES: [2, 3, 4]

## Variants for Evolution Testing

| Variant | Description | Hypothesis |
|---------|-------------|------------|
| A | Single-pass, register storage | Baseline improvement |
| B | + Vectorized float4 loads | Better memory coalescing |
| C | + Shared memory for large N | Handle long sequences |
| D | + Fused exp2 (faster than exp) | Reduce ALU cycles |
| E | + Loop unrolling pragmas | Better instruction scheduling |

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Speedup vs PyTorch | 0.66x | 1.5x | 3x |
| Memory bandwidth util | 59% | 80% | 90% |
| Launch overhead | 31µs | 15µs | 10µs |

## References

- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Anatomy of Triton Attention Kernel](https://arxiv.org/html/2511.11581v1)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [TritonForge](https://arxiv.org/html/2512.09196v2)
