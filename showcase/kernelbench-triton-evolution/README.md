# KernelBench Triton Evolution

Exploring GPU kernel optimization using Claude Agent SDK with hierarchical agents. This showcase demonstrates using `evolve-sdk` to evolve Triton softmax kernels.

## Key Findings

### Initial vs Final Results

| Metric | Initial Claim | Actual (After Investigation) |
|--------|---------------|------------------------------|
| Claimed speedup | 2.27x | **Measurement artifact** |
| Actual best speedup | - | **~1.02x** (marginal) |
| Root cause | - | JIT warmup not accounted for |

### What We Learned

1. **JIT Warmup Matters**: The initial 2.27x measurement was due to comparing cold JIT compilation (first run) vs warm cache. After proper warmup, the evolved kernel was only marginally faster.

2. **PyTorch is Well-Optimized**: PyTorch's `F.softmax()` uses cuDNN's highly optimized implementation. Beating it with Triton requires significant innovation.

3. **Kernel Launch Overhead**: Triton has ~31μs kernel launch overhead vs PyTorch's ~7.5μs. This dominates for small inputs.

4. **Comprehensive Benchmarking Required**: The evolved kernel only won 26% of test cases (primarily long sequences). PyTorch won most small/medium cases.

## Honest Assessment

After proper benchmarking with JIT warmup:
- **Best case**: 1.02-1.05x speedup on long sequences (2K+ elements)
- **Typical case**: 0.8-1.0x (Triton slightly slower)
- **Worst case**: 0.5x on small batches (launch overhead)

**Conclusion**: For softmax specifically, PyTorch's cuDNN implementation is already highly optimized. Significant speedups require either:
- Different algorithms (Flash Attention's fused approach)
- Different workloads where Triton's flexibility helps
- Fusion with other operations

## Quick Start

```bash
cd showcase/kernelbench-triton-evolution

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt

# Run benchmark (requires GPU)
python python/benchmark.py tasks/softmax/starter_kernel.py
```

## Algorithm Comparison

### Starter Kernel (Two-Pass)

Standard two-pass softmax from Triton tutorial:

```
Pass 1: Load entire row → find max
Pass 2: Load row again → compute exp(x - max), sum, normalize, store
```

### Evolved Kernel (Online Algorithm)

Uses online algorithm to compute max and sum in single pass:

```
Single Pass: Load blocks → update running max → adjust previous sum → accumulate
Final Pass: Normalize with precomputed values
```

**Theoretical advantage**: Reduces memory passes from 2 to ~1.5

**Practical result**: Marginal speedup (~1.02x) because:
- PyTorch's cuDNN is heavily optimized
- Memory bandwidth isn't the only bottleneck
- Kernel launch overhead significant

## Files

```
kernelbench-triton-evolution/
├── evolve_config.json        # Evolution configuration
├── evaluate_on_lightning.py  # GPU evaluation via Lightning.ai
├── tasks/softmax/
│   ├── starter_kernel.py     # Initial seed
│   ├── reference.py          # PyTorch reference
│   └── test_cases.py         # Correctness tests
├── python/
│   ├── benchmark.py          # Local benchmarking
│   └── baseline.py           # Generate baselines
└── EVOLUTION_LOG.md          # Detailed evolution history
```

## Lessons for Kernel Optimization

1. **Always use proper warmup** before benchmarking
2. **Test across multiple input sizes** - speedups often don't generalize
3. **Compare against production baselines**, not naive implementations
4. **Measure kernel launch overhead** separately
5. **Be skeptical of large speedup claims** without seeing methodology

## References

- [KernelBench: Can LLMs Write GPU Kernels?](https://arxiv.org/abs/2502.10517) - Stanford benchmark
- [Triton Language](https://triton-lang.org/) - GPU kernel DSL
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Online softmax algorithm
