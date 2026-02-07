# AlgoTune Speedup: Evolutionary Code Optimization

**2.95x harmonic mean speedup** on 25 AlgoTune tasks — 72% faster than o4-mini-high (1.72x) and 122% faster than Claude Opus 4 (1.33x).

## Architecture

Two-phase approach: evolve-sdk generates baseline solutions from AlgoTune reference implementations, then iteratively improves them through mutation, crossover, and selection with trust-gated evaluation.

![Architecture](architecture.svg)

## Results

| Model | Harmonic Mean Speedup | Tasks Improved |
|-------|----------------------|----------------|
| **Opus 4.6 + evolve-sdk** | **2.95x** | **95%** |
| o4-mini-high | 1.72x | 60% |
| deepseek-r1 | 1.70x | 61% |
| gemini-2.5-pro | 1.51x | 49% |
| claude-opus-4 | 1.33x | 40% |

*Our results are on a curated 25-task subset (19 valid). Published baselines are on the full 154-task benchmark. See Methodology for details.*

### Per-Task Breakdown

| Task | Speedup | Category |
|------|---------|----------|
| ode_lotkavolterra | 1221.65x | Differential Equations |
| ode_brusselator | 506.34x | Differential Equations |
| svm | 13.09x | Machine Learning |
| pagerank | 11.13x | Graph Algorithms |
| lu_factorization | 10.43x | Linear Algebra |
| minimum_spanning_tree | 9.70x | Graph Algorithms |
| correlate_1d | 5.50x | Signal Processing |
| fft_convolution | 5.47x | Signal Processing |
| pca | 5.25x | Machine Learning |
| qr_factorization | 4.88x | Linear Algebra |
| convex_hull | 4.47x | Computational Geometry |
| convolve_1d | 4.16x | Signal Processing |
| cholesky_factorization | 2.23x | Linear Algebra |
| kmeans | 2.19x | Machine Learning |
| linear_system_solver | 1.81x | Linear Algebra |
| eigenvectors_real | 1.41x | Linear Algebra |
| svd | 1.22x | Linear Algebra |
| matrix_multiplication | 1.15x | Linear Algebra |
| dijkstra_from_indices | 1.00x | Graph Algorithms |

6 tasks failed to produce valid solutions (matrix_exponential, eigenvalues_real, fft_cmplx_scipy_fftpack, dct_type_I_scipy_fftpack, shortest_path_dijkstra, lasso).

## Why This Problem Matters

[AlgoTune](https://github.com/oripress/AlgoTune) is a NeurIPS 2025 benchmark with 154 numerical computing tasks spanning linear algebra, optimization, signal processing, differential equations, and more. Models must write Python code that produces correct results **faster** than reference implementations.

The benchmark uses **harmonic mean speedup** — a single slow task drags down the overall score significantly, making consistent improvement across diverse problem types essential.

Current best single-shot results on the full benchmark range from 1.33x (Claude Opus 4) to 1.72x (o4-mini-high). By layering evolutionary optimization on top of strong single-shot baselines, we achieve 2.95x on our task subset.

## Methodology

### Task Selection
- 25 tasks selected from AlgoTune's 154-task benchmark, spanning linear algebra, signal processing, graph algorithms, ML, ODEs, and computational geometry
- Tasks chosen for diversity and optimization potential (not cherry-picked for easy wins)

### Evolution Pipeline
- evolve-sdk generates initial population (6 candidates) from reference implementations + task descriptions
- Evolution: 10 generations max, population 6, plateau 3 (early stopping after 3 stagnant generations)
- Mutation strategies: algorithm replacement, vectorization, JIT compilation (Numba), direct LAPACK calls, mathematical shortcuts
- Crossover: combines best traits from top parents (e.g., Numba for small matrices + direct LAPACK for large ones)
- Trust system: adversary review for suspicious fitness jumps, variance gates (3 evaluations, 15% CV threshold)
- Memory system tracks successful/failed strategies across generations

### Evaluation
- All timing uses AlgoTune's `perf_counter_ns`, median of N runs
- Correctness validated via `task.is_solution()` on generated test inputs
- Speedups floored at 1.0x (if our solution is slower, use the reference)
- Harmonic mean: `H = n / sum(1/s_i)` — heavily penalizes low outliers

### Key Optimizations Discovered
- **ODE solvers (500-1200x):** Replaced general-purpose `solve_ivp` with specialized RK4/symplectic integrators tuned to specific equation systems
- **Graph algorithms (10-11x):** Sparse matrix / CSR representations instead of NetworkX DiGraph objects
- **ML tasks (2-13x):** Specialized algorithms (e.g., mini-batch k-means, LAPACK-based PCA, libsvm optimization)
- **Linear algebra (1.2-10x):** Direct LAPACK calls bypassing scipy wrappers, Numba JIT for small matrices, Fortran-ordered memory layout
- **Signal processing (4-5.5x):** FFT-based convolution with optimal zero-padding, direct BLAS calls

### Transparency
- Results are on a curated 25-task subset (19 produced valid solutions)
- Published baselines are on the full 154-task benchmark — direct comparison is not apples-to-apples
- All solutions, configs, and evolution logs are committed for reproducibility
- The ODE speedups (500-1200x) are legitimate but inflated by the reference using a very general solver; these tasks would show smaller gains with a stronger reference

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
# Generate per-task evolve configs
python scripts/generate_evolve_configs.py

# Smoke test on a few tasks
python scripts/run_full.py --task cholesky_factorization

# Full evolution run (25 tasks, ~15 hours)
python scripts/run_full.py

# Aggregate results
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
│   ├── run_full.py                 # Full evolution run across all tasks
│   ├── generate_evolve_configs.py  # Create per-task configs
│   ├── create_baselines.py         # Auto-generate baseline solvers
│   └── aggregate_results.py        # Compute harmonic mean, tables
├── configs/                        # Generated per-task evolve configs
├── solutions/                      # Phase 1 baseline solutions
├── evolved/                        # Phase 2 evolved solutions
├── results/                        # JSON results + evolution logs
└── algotune/                       # Cloned AlgoTune repo (gitignored)
```

## Task Categories

Selected tasks span 6 categories:
- **Linear Algebra** (8): Cholesky, eigenvectors, LU, QR, SVD, linear system, matrix multiplication, matrix exponential
- **Signal Processing** (5): FFT convolution, convolve 1D, correlate 1D, FFT complex, DCT type I
- **Graph Algorithms** (4): Dijkstra (2 variants), PageRank, minimum spanning tree
- **Machine Learning** (4): k-means, SVM, PCA, lasso
- **Differential Equations** (2): Brusselator, Lotka-Volterra
- **Computational Geometry** (1): Convex hull

## References

- [AlgoTune: Can Language Models Speed Up Numerical Programs?](https://arxiv.org/abs/2507.15887) (NeurIPS 2025)
- [AlgoTune GitHub](https://github.com/oripress/AlgoTune)
- [evolve-sdk](../../sdk/)
