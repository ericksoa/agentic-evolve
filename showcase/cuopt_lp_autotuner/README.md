# cuOpt LP AutoTuner

Evolutionary optimization of NVIDIA cuOpt LP solver configurations. This package evolves robust parameter configurations for the cuOpt GPU-accelerated solver to maximize throughput and minimize p95 latency—without overfitting.

## Overview

The AutoTuner evolves cuOpt solver configurations including:
- **Method selection**: PDLP, Dual Simplex, Barrier, or Concurrent
- **PDLP modes**: Stable3, Methodical1, Fast1
- **Tolerances**: Primal, dual, and gap tolerances
- **Preprocessing**: Presolve, dualization, crossover
- **Barrier settings**: Folding, ordering, augmented system

The configurations are evolved using a genetic algorithm with:
- Train/validation/test splits for anti-overfitting
- Multi-objective fitness (throughput + latency)
- Hard constraint: no >2% p95 regression vs baseline on test set

## Installation

```bash
# From the agentic-evolve root directory
pip install scipy numpy

# Test the package
python3 -c "from showcase.cuopt_lp_autotuner import CuOptConfig, CuOptSolver; print('OK')"
```

## Quick Start

### Complete Demo

Run the full demonstration pipeline:

```bash
# Quick demo (5 generations, 50 problems)
python3 -m showcase.cuopt_lp_autotuner.demo --quick

# Full demo (15 generations, 100 problems)
python3 -m showcase.cuopt_lp_autotuner.demo
```

This will:
1. Create 15 reproducible LP benchmarks from real-world domains
2. Evolve optimal cuOpt configuration using GA
3. Benchmark baseline vs evolved on real problems
4. Show detailed comparison results

### Run Evolution

```bash
# With synthetic LP problems
python3 -m showcase.cuopt_lp_autotuner.evolution \
    --synthetic 100 \
    --population 20 \
    --generations 30 \
    --verbose

# With real MPS benchmark files
python3 -m showcase.cuopt_lp_autotuner.evolution \
    --lp-dir ./benchmarks \
    --generations 50 \
    --verbose
```

Example output:
```
Generated 60 train, 20 val, 20 test problems
Computing baseline metrics...
Baseline train: 58.6 LP/s, p95=282.34ms
Baseline val: 578.3 LP/s, p95=3.36ms
Baseline test: 536.1 LP/s, p95=2.89ms
Initializing population of 20...
Initial best fitness: 137.42

============================================================
GENERATION 1
============================================================
New best! Fitness: 139.59
Validation fitness: 1.41
...

============================================================
FINAL EVALUATION ON TEST SET
============================================================
Test fitness: 1.88
Test throughput: 593.1 LP/s (+1.06 vs baseline)
Test p95: 2.65ms (-8.2% vs baseline)

✓ Solution passes p95 constraint on test set
```

### Programmatic Usage

```python
from showcase.cuopt_lp_autotuner import (
    CuOptConfig,
    CuOptSolver,
    CuOptEvolution,
    LPCorpus,
    CuOptEvolutionConfig,
)

# Create LP corpus
corpus = LPCorpus(seed=42)
corpus.generate_synthetic(n_train=100, n_val=20, n_test=20)

# Configure evolution
config = CuOptEvolutionConfig(
    population_size=20,
    max_generations=50,
    max_p95_regression_pct=2.0,
)

# Run evolution
evolution = CuOptEvolution(corpus, config, seed=42)
best = evolution.run()

# Use the best configuration
solver = CuOptSolver()
result = solver.solve(lp_data, best.config)
print(f"Objective: {result.objective}, Time: {result.solve_time_ms}ms")
```

## Architecture

```
cuopt_lp_autotuner/
├── cuopt_config.py    # Real cuOpt parameter genome (26+ parameters)
├── cuopt_solver.py    # cuOpt API integration (scipy HiGHS fallback)
├── mps_loader.py      # Load LP problems from MPS/LP files
├── evolution.py       # GA with train/val/test anti-overfitting
├── lp_generator.py    # Synthetic LP corpus generators
│
├── # Legacy (policy-based approach):
├── policy.py          # JSON policy representation
├── cuopt_runner.py    # Old solver adapter
├── benchmark.py       # Old fitness evaluation
├── evolve.py          # Old evolution loop
└── adversarial.py     # Adversarial instance generator
```

## cuOpt Configuration Parameters

The genome includes all tunable cuOpt parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `method` | enum | 0-3 | CONCURRENT, PDLP, DUAL_SIMPLEX, BARRIER |
| `pdlp_solver_mode` | enum | - | Stable3, Methodical1, Fast1 |
| `presolve` | bool | - | Enable LP presolve |
| `dualize` | int | -1,0,1 | Auto/No/Yes dualization |
| `crossover` | bool | - | Crossover to basic solution |
| `absolute_primal_tolerance` | log_float | 1e-8 to 1e-2 | Primal feasibility |
| `relative_primal_tolerance` | log_float | 1e-8 to 1e-2 | Relative primal |
| `absolute_dual_tolerance` | log_float | 1e-8 to 1e-2 | Dual feasibility |
| `relative_dual_tolerance` | log_float | 1e-8 to 1e-2 | Relative dual |
| `absolute_gap_tolerance` | log_float | 1e-8 to 1e-2 | Duality gap |
| `relative_gap_tolerance` | log_float | 1e-8 to 1e-2 | Relative gap |
| `iteration_limit` | int | 1K to 10M | Max iterations |
| `time_limit` | float | 1 to 3600 | Time limit (sec) |
| `num_cpu_threads` | int | 1 to 32 | CPU threads |
| `num_gpus` | int | 1 to 2 | GPUs for concurrent |
| `barrier_folding` | int | -1,0,1 | Barrier folding |
| `barrier_ordering` | int | -1,0,1 | Barrier ordering |
| `barrier_augmented` | int | -1,0,1 | Augmented system |
| `infeasibility_detection` | bool | - | Detect infeasibility |
| `first_primal_feasible` | bool | - | Stop at first feasible |

## Fitness Function

The multi-objective fitness combines:

```
fitness = throughput_score + latency_score + success_bonus
```

Where:
- `throughput_score = (throughput / baseline - 1) × 10`
- `latency_score = (1 - p95 / baseline_p95) × 10`
- `success_bonus = (success_rate - baseline_success) × 5`

**Hard constraint**: Configuration is heavily penalized if p95 regresses >2% vs baseline.

## LP Problem Sources

### Synthetic LP Types
The generator creates realistic LP instances:

| Type | Description |
|------|-------------|
| `transportation` | Supply/demand flow problems |
| `capacity_planning` | Multi-period production |
| `resource_allocation` | Profit maximization with limits |
| `network_flow` | Minimum cost network flow |
| `blending` | Mixing/composition problems |
| `production_planning` | Multi-product scheduling |

### Real MPS Benchmarks
Load problems from standard MPS files:

```python
from showcase.cuopt_lp_autotuner import load_problem

lp = load_problem("benchmark.mps.gz")  # Supports .mps, .mps.gz, .mps.bz2
print(f"Loaded: {lp.n_vars} vars, {lp.n_cons} constraints")
```

## Results Format

After evolution, results are saved to JSON:

```json
{
  "best_config": {
    "method": 0,
    "pdlp_solver_mode": "Stable3",
    "absolute_primal_tolerance": 1.55e-08,
    ...
  },
  "train_fitness": {
    "fitness": 146.63,
    "throughput_score": 136.75,
    "p95_regression_pct": -98.81
  },
  "test_fitness": {
    "fitness": 1.88,
    "throughput_lps_per_sec": 593.1,
    "p95_regression_pct": -8.2,
    "violates_p95_constraint": false
  }
}
```

## cuOpt Integration

The solver uses scipy HiGHS as a fallback when cuOpt is unavailable. When cuOpt is installed:

```python
from showcase.cuopt_lp_autotuner import is_cuopt_available

if is_cuopt_available():
    print("Using NVIDIA cuOpt GPU solver")
else:
    print("Using scipy HiGHS fallback")
```

## CLI Reference

```bash
# NEW: Evolution with real cuOpt parameters
python3 -m showcase.cuopt_lp_autotuner.evolution [OPTIONS]
  --lp-dir PATH      Directory with MPS/LP benchmark files
  --synthetic N      Generate N synthetic problems (default: 100)
  --population N     Population size (default: 20)
  --generations N    Max generations (default: 30)
  --seed N           Random seed (default: 42)
  --output PATH      Results JSON path
  --verbose          Enable verbose output

# LEGACY: Policy-based benchmark
python3 -m showcase.cuopt_lp_autotuner.benchmark [OPTIONS]
  --policy PATH      Policy file or "baseline"
  --output PATH      Save results JSON
  --quiet            Suppress output

# LEGACY: Policy-based evolution
python3 -m showcase.cuopt_lp_autotuner.evolve [OPTIONS]
  --generations N    Max generations
  --population N     Population size
```

## Success Criteria

Evolution succeeds if on TEST set:
- ✅ Throughput improvement (any positive gain)
- ✅ p95 latency reduction (any reduction)
- ✅ No >2% p95 regression vs baseline
- ✅ Success rate ≥ baseline

## References

- [NVIDIA cuOpt LP/MILP Settings](https://docs.nvidia.com/cuopt/user-guide/latest/lp-milp-settings.html)
- [PDLP Algorithm Paper](https://arxiv.org/abs/2106.04756)

## License

Part of the agentic-evolve project.
