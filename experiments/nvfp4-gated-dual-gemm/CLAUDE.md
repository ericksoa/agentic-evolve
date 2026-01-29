# NVFP4 Gated Dual GEMM - Competition Project

## Competition: Blackwell NVFP4 Kernel Hackathon
- **Problem**: NVFP4 Gated Dual GEMM (Problem 3)
- **Deadline**: January 16, 2026
- **Leaderboard**: https://www.gpumode.com/v2/leaderboard/598

## Problem Specification

Compute: `C = silu(A @ B1) * (A @ B2)`

### Input Format
- `a`: [m, k, l] - FP4 (Float4E2M1FN) with scale factors
- `b1`, `b2`: [n, k, l] - FP4 with scale factors
- Scale factors: FP8 (Float8E4M3FN), one per 16 values
- Output `c`: [m, n, l] - Float16

### Performance Target
- Reference: PyTorch `_scaled_mm` (slow, batched)
- Target: Custom kernel approaching speed-of-light

## Submission Requirements

```bash
# Install popcorn-cli
pip install popcorn-cli

# Submit to leaderboard
popcorn-cli submit --gpu B200 --leaderboard nvfp4_dual_gemm python/submission.py
```

## Evolution Strategy

This project uses `/evolve-perf` to evolve NVFP4 kernels.

### Key Optimization Dimensions
1. **Tile sizes**: (128, 128, 256) baseline, explore variations
2. **Memory hierarchy**: TMEM for Blackwell, shared memory staging
3. **Warp specialization**: Separate MMA and epilogue warps
4. **Pipeline depth**: Overlap memory and compute
5. **Scale factor handling**: Efficient block-scaled operations

### Viable Algorithm Families
- CuTe DSL (Python/CUTLASS)
- Raw Triton
- CUDA inline (torch.cuda.compile)
- TensorRT-LLM patterns

## Cloud Execution

All benchmarks MUST run on Blackwell GPU (B200/GB200):
- Use provided cloud access
- Maximum 4 parallel mutation evaluations
- Each eval timeout: 5 minutes

## Directory Structure

```
nvfp4-gated-dual-gemm/
├── python/
│   ├── task.py           # Problem definition
│   ├── reference.py      # Baseline kernel
│   ├── submission.py     # Current best (evolve target)
│   └── evaluate.py       # Fitness function
├── mutations/            # All evolved variants
├── .evolve-sdk/          # Evolution state
└── CLAUDE.md             # This file
```

## Running Evolution

```bash
/evolve-perf NVFP4 Gated Dual GEMM kernel --budget 100k
```

The evolution will:
1. Generate kernel mutations
2. Evaluate on cloud Blackwell GPU
3. Select best performers
4. Crossover winning strategies
5. Submit to leaderboard periodically
