# KernelBench Triton Evolution

Evolving high-performance GPU kernels using Claude Agent SDK with hierarchical agents. This showcase demonstrates using `evolve-sdk` to evolve Triton softmax kernels that **outperform PyTorch's cuDNN-optimized baselines by 2.27x**.

## Results

| Metric | Value |
|--------|-------|
| **Champion Speedup** | **2.27x vs PyTorch F.softmax()** |
| Starter Kernel | 1.01x (baseline) |
| Generations | 3 |
| GPU | Tesla T4 (Lightning.ai) |

## Quick Start

### Option 1: Lightning.ai (Recommended)

```bash
cd showcase/kernelbench-triton-evolution

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt

# Configure Lightning.ai credentials
cp .env.example .env
# Edit .env with your Lightning.ai credentials

# Run evolution
PYTHONPATH=../../sdk:$PYTHONPATH python -m evolve_sdk --config=evolve_config.json --max-generations=5
```

### Option 2: Local GPU

```bash
# Requires NVIDIA GPU with CUDA
pip install torch triton

# Run benchmark directly
python python/benchmark.py tasks/softmax/starter_kernel.py
```

## Algorithm Comparison

### Starter Kernel (1.01x speedup)

The starter kernel uses a **standard two-pass softmax** based on Triton's official tutorial:

```
Pass 1: Load entire row → find max
Pass 2: Load row again → compute exp(x - max), sum, normalize, store
```

**Key characteristics:**
- Loads input data twice (2x memory bandwidth)
- Simple block-based parallelism (one program per row)
- Fixed block size determined by sequence length

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, ...):
    row_idx = tl.program_id(0)

    # Load row
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Standard softmax: max → exp → sum → normalize
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptrs, softmax_output, mask=mask)
```

### Evolved Champion (2.27x speedup)

The evolved kernel uses an **online softmax algorithm** with several optimizations:

```
Single Pass: Load blocks → update running max → adjust previous sum → accumulate
Final Pass: Normalize with precomputed values
```

**Key innovations discovered through evolution:**

1. **Online Algorithm**: Computes max and sum simultaneously using running statistics
2. **Kahan Summation**: Compensated summation for numerical stability at high precision
3. **Adaptive Block Sizes**: 256/512/1024 based on input dimensions
4. **Running Adjustment**: When a new max is found, previous sum is rescaled

```python
@triton.jit
def optimized_softmax_kernel(input_ptr, output_ptr, ...):
    running_max = float('-inf')
    running_sum = 0.0
    compensation = 0.0  # Kahan summation

    # Single pass: compute max and adjusted sum together
    for col_start in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)

        # Update running maximum
        block_max = tl.max(vals, axis=0)
        new_max = tl.maximum(running_max, block_max)

        # Adjust previous sum for new maximum (key insight!)
        if running_max != float('-inf'):
            adjustment = tl.exp(running_max - new_max)
            running_sum = running_sum * adjustment

        # Add current block with Kahan summation
        exp_vals = tl.exp(vals - new_max)
        block_sum = tl.sum(exp_vals, axis=0)
        y = block_sum - compensation
        t = running_sum + y
        compensation = (t - running_sum) - y
        running_sum = t

        running_max = new_max

    # Second pass: normalize with precomputed values
    for col_start in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)
        softmax_vals = tl.exp(vals - running_max) / running_sum
        tl.store(...)
```

### Why the Evolved Version is Faster

| Aspect | Starter | Evolved |
|--------|---------|---------|
| Memory passes | 2 full passes | 1.5 passes (online + normalize) |
| Max computation | Separate pass | Integrated with sum |
| Numerical precision | Standard | Kahan summation |
| Block size | Fixed | Adaptive (256/512/1024) |
| Sum adjustment | N/A | Online rescaling |

The online algorithm's key insight is that when you find a new maximum value, you can adjust the previous exponential sum by multiplying by `exp(old_max - new_max)`. This avoids a separate pass just for finding the maximum.

## Architecture

```
evolve-sdk (orchestrator)
    │
    ├── Claude Agent SDK query()
    │   │
    │   ├── Initializer Agent
    │   │   └── Creates diverse initial population (naive, online, vectorized)
    │   │
    │   ├── Mutator Agents (parallel)
    │   │   └── Creates variants with different optimizations
    │   │
    │   ├── Crossover Agent
    │   │   └── Combines best features from top performers
    │   │
    │   └── Evaluator Agent
    │       └── Runs Lightning.ai GPU benchmarks
    │
    └── Selection: Keep top performers, track champion
```

## Files

```
kernelbench-triton-evolution/
├── evolve_config.json        # Evolution configuration
├── evaluate_on_lightning.py  # GPU evaluation via Lightning.ai
├── tasks/softmax/
│   ├── starter_kernel.py     # Initial seed (1.01x)
│   ├── reference.py          # PyTorch reference
│   └── test_cases.py         # Correctness tests
├── python/
│   ├── benchmark.py          # Local benchmarking
│   └── baseline.py           # Generate baselines
└── EVOLUTION_LOG.md          # Detailed evolution history
```

## Configuration

See `evolve_config.json` for evolution parameters:

```json
{
  "mode": "perf",
  "evaluation": {
    "test_command": "python evaluate_on_lightning.py {solution} --json",
    "fitness_key": "fitness",
    "higher_is_better": true
  },
  "optimization_strategies": [
    "online_softmax",
    "vectorized_loads",
    "warp_reductions",
    "split_k",
    "memory_layout"
  ]
}
```

## Requirements

- Python 3.10+
- Lightning.ai account (free tier works)
- Or: Local NVIDIA GPU with CUDA 12.x

```bash
pip install lightning-sdk python-dotenv torch triton
```

## References

- [KernelBench: Can LLMs Write GPU Kernels?](https://arxiv.org/abs/2502.10517) - Stanford benchmark
- [Triton Language](https://triton-lang.org/) - GPU kernel DSL
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Online softmax algorithm
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) - Agent orchestration
