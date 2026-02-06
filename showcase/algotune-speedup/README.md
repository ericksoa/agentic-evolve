# AlgoTune Speedup: Evolutionary Code Optimization

**Target: >2.5x harmonic mean speedup** on AlgoTune benchmark (vs 1.72x o4-mini-high SOTA, 1.33x Opus 4 baseline).

## Architecture

Two-phase approach: AlgoTune's built-in agent produces baseline solutions, then evolve-sdk iteratively improves them through mutation, crossover, and selection.

![Architecture](architecture.svg)

## Results

*Results will be populated after running the benchmark.*

<!-- Results table will be inserted by aggregate_results.py -->

## Why This Problem Matters

[AlgoTune](https://github.com/oripress/AlgoTune) is a NeurIPS 2025 benchmark with 154 numerical computing tasks spanning linear algebra, optimization, signal processing, differential equations, and more. Models must write Python code that produces correct results **faster** than reference implementations.

The benchmark uses **harmonic mean speedup** — a single slow task drags down the overall score significantly, making consistent improvement across diverse problem types essential.

Current best single-shot result is 1.72x (o4-mini-high). By layering evolutionary optimization on top of a strong Opus 4.6 baseline, we aim to push past 2.5x.

## Methodology

### Phase 1: Single-Shot Baseline
- AlgoTune's built-in agent runs with Claude Opus 4.6 on ~25 selected tasks
- Tasks selected for maximum headroom (existing models scored <1.5x)
- Produces `solutions/{task}/solver.py` for each task

### Phase 2: Evolutionary Optimization
- Phase 1 solutions become starter solutions for evolve-sdk
- Evolution: 10 generations, population 6, plateau threshold 3
- Mutations: algorithm replacement, vectorization, JIT compilation, library backends, mathematical shortcuts
- Trust system with variance gates (3 evaluations, 15% CV threshold)
- Memory system tracks what worked/failed across mutations

### Evaluation
- All timing uses AlgoTune's own methodology (perf_counter_ns, median of N runs)
- Correctness validated via `task.is_solution()` on all test inputs
- Final aggregation picks best(Phase 1, Phase 2) per task
- Harmonic mean: `H = n / sum(1/s_i)`

### Transparency
- Results are on a curated ~25-task subset with the most optimization headroom
- Full-benchmark extrapolation uses published Opus 4.5 results for the remaining ~129 tasks
- All solutions, configs, and evolution logs are committed for reproducibility

## Quick Start

### Prerequisites
- Python 3.12
- `ANTHROPIC_API_KEY` set in environment
- evolve-sdk installed (`pip install -e ../../sdk`)

### Setup
```bash
cd showcase/algotune-speedup
bash setup.sh
source .venv/bin/activate
export PYTHONPATH=$(pwd)/algotune:$PYTHONPATH
```

### Run
```bash
# 1. Select task subset
python scripts/select_subset.py

# 2. Generate per-task configs
python scripts/generate_evolve_configs.py

# 3. Phase 1: baseline (smoke test first)
python scripts/run_phase1.py --tasks 3
python scripts/run_phase1.py              # Full run (~2 hours)

# 4. Phase 2: evolution (smoke test first)
python scripts/run_phase2.py --tasks 3
python scripts/run_phase2.py              # Full run (~12 hours)

# 5. Aggregate results
python scripts/aggregate_results.py
```

### Resume
Phase 1 and Phase 2 save results incrementally. Re-running skips completed tasks automatically. Use `--force` to re-run a task.

## File Structure

```
showcase/algotune-speedup/
├── README.md                       # This file
├── CLAUDE.md                       # Workflow instructions
├── setup.sh                        # Environment setup
├── architecture.svg                # Two-phase architecture diagram
├── evaluate_task.py                # Bridge: evolve-sdk ↔ AlgoTune timing
├── evolve_config_template.json     # Template for per-task configs
├── scripts/
│   ├── select_subset.py            # Pick ~25 high-headroom tasks
│   ├── generate_evolve_configs.py  # Create per-task configs
│   ├── run_phase1.py               # Phase 1: AlgoTune agent baseline
│   ├── run_phase2.py               # Phase 2: evolve-sdk evolution
│   └── aggregate_results.py        # Compute harmonic mean, tables
├── configs/                        # Generated per-task evolve configs
├── solutions/                      # Phase 1 baseline solutions
├── evolved/                        # Phase 2 evolved solutions
├── results/                        # JSON results + evolution logs
└── algotune/                       # Cloned AlgoTune repo (gitignored)
```

## Task Categories

Selected tasks span:
- **Linear Algebra**: Cholesky, eigenvalue, LU, QR, SVD, matrix exponential
- **Optimization**: LP, group lasso, basis pursuit, lasso, QP
- **Signal Processing**: FFT convolution, DCT, correlation, spectral estimation
- **Differential Equations**: Lorenz96, Brusselator, FitzHugh-Nagumo
- **Graph Algorithms**: Max clique, graph coloring, shortest path
- **Cryptography**: AES-GCM, ChaCha20
- **Computational Geometry**: Convex hull, Voronoi

## References

- [AlgoTune: Can Language Models Speed Up Numerical Programs?](https://arxiv.org/abs/2507.15887) (NeurIPS 2025)
- [AlgoTune GitHub](https://github.com/oripress/AlgoTune)
- [evolve-sdk](../../sdk/)
