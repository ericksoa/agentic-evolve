# GPU-Accelerated Tree Packing Optimizer

GPU-accelerated Simulated Annealing for the Santa 2025 tree packing challenge.

## Quick Start (Lightning.ai)

### 1. Create Studio
1. Go to [lightning.ai](https://lightning.ai)
2. Create new Studio with **L40S GPU** (48GB VRAM)
3. Select Python 3.10 or 3.11

### 2. Setup Environment
```bash
# Clone repo
git clone https://github.com/your-repo/agentic-evolve.git
cd agentic-evolve/showcase/santa-2025-packing

# Install dependencies
pip install torch numpy

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Copy Best Solutions
Upload `submission_best.csv` to the studio root directory.

### 4. Run Optimization

**Global Search Mode** (recommended for finding new solutions):
```bash
# Small n values (fast, good for testing)
python python/gpu/lightning_run.py --n-range 1-20 --chains 500 --iterations 10000 --mode global

# Medium n values
python python/gpu/lightning_run.py --n-range 21-50 --chains 300 --iterations 15000 --mode global

# Large n values (more iterations needed)
python python/gpu/lightning_run.py --n-range 51-100 --chains 200 --iterations 20000 --mode global
```

**Refine Mode** (for fine-tuning existing solutions):
```bash
python python/gpu/lightning_run.py --n-range 1-50 --chains 500 --iterations 5000 --mode refine
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-range` | 1-20 | Range of n values (e.g., "1-20" or "5,10,15") |
| `--chains` | 200 | Number of parallel SA chains |
| `--iterations` | 5000 | SA iterations per chain |
| `--mode` | refine | `refine` (from best) or `global` (from scratch) |
| `--compare` | submission_best.csv | CSV to compare against |
| `--output` | gpu_sa_results.json | Output JSON file |

## Expected Performance

On L40S GPU with CUDA:
- Small n (1-20): ~500-1000 iter/s per chain
- Medium n (20-50): ~200-500 iter/s per chain
- Large n (50-100): ~100-200 iter/s per chain

With 500 chains running in parallel, this gives massive throughput compared to CPU.

## Output Format

Results saved to JSON:
```json
{
  "config": {...},
  "summary": {
    "total_time": 123.4,
    "improved_count": 5,
    "total_improvement": 0.1234
  },
  "results": {
    "5": {
      "gpu_side": 1.45,
      "current_side": 1.48,
      "improvement": 0.0123,
      "valid": true
    }
  }
}
```

## Costs

L40S on lightning.ai: ~$2-3/hour

Typical run times:
- Quick test (n=1-20, 500 chains, 5k iter): ~10 min = ~$0.50
- Full small n (n=1-50, 500 chains, 10k iter): ~1 hour = ~$2.50
- Comprehensive (n=1-100, 300 chains, 15k iter): ~3 hours = ~$7.50

## Files

- `gpu_sa.py` - Core GPU SA implementation
- `lightning_run.py` - Runner script for lightning.ai
- `benchmark.py` - GPU vs CPU benchmarking
