# Evolve SDK

Python SDK for evolutionary algorithm discovery using the Claude Agent SDK. Powers the `/evolve` family of skills with hierarchical agents, clean context per generation, and fine-grained control.

## Overview

The SDK orchestrates evolution through specialized subagents:

- **Initializer**: Creates diverse initial population
- **Mutators**: Generate variants of promising solutions (parallel)
- **Crossover**: Combines innovations from multiple parents
- **Evaluator**: Measures fitness against benchmarks

Each agent runs with **clean context**—they only see their specific task, not the full evolution history. This prevents context bloat and keeps agents focused.

## Installation

```bash
# From the sdk/ directory
pip install -e .

# Claude Agent SDK is required for actual evolution
pip install claude-agent-sdk
```

## Usage

### As CLI

```bash
# Performance optimization
python -m evolve_sdk "faster sorting algorithm" --mode=perf

# Size optimization (code golf)
python -m evolve_sdk "shortest Python sort" --mode=size

# ML optimization
python -m evolve_sdk "improve F1 for classification" --mode=ml

# With options
python -m evolve_sdk "faster string search" \
    --mode=perf \
    --max-generations=20 \
    --population-size=10

# Resume previous evolution
python -m evolve_sdk --resume
```

### As Library

```python
import asyncio
from evolve_sdk import EvolutionRunner

async def main():
    runner = EvolutionRunner(
        problem="shortest Python sort",
        mode="size",
        max_generations=50,
        parallel_mutations=True,
    )
    result = await runner.run()
    print(f"Champion: {result['champion']}")

asyncio.run(main())
```

### Via Skills

The `/evolve` skills are thin wrappers that call this SDK:

```
/evolve faster sorting algorithm      # Detects perf mode
/evolve shortest Python solution      # Detects size mode
/evolve improve accuracy              # Detects ml mode
```

## Architecture

```
EvolutionRunner (orchestrator)
├── Initializer Agent (Gen 0)
│   └── Creates diverse initial population
│
└── For each generation:
    ├── Mutator Agents (parallel)
    │   └── Each creates one mutation variant
    ├── Crossover Agent
    │   └── Combines top solutions
    └── Evaluator Agent
        └── Measures fitness of all new solutions
```

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | size | Optimization mode (size, perf, ml) |
| `--max-generations` | 50 | Maximum generations |
| `--population-size` | 10 | Population size |
| `--plateau` | 5 | Stop after N gens without improvement |
| `--no-parallel` | false | Run mutations sequentially |
| `--model` | sonnet | Model for subagents |
| `--config` | - | Path to evolve_config.json |

### Config File

```json
{
  "problem": "optimize sorting",
  "mode": "perf",
  "test_command": "python benchmark.py {solution}",
  "starter_solutions": ["baseline.py"],
  "max_generations": 20,
  "population_size": 10
}
```

### Programmatic

```python
from evolve_sdk import EvolutionRunner

runner = EvolutionRunner(
    problem="...",
    mode="size",
    max_generations=50,
    plateau_threshold=5,
    population_size=10,
    parallel_mutations=True,
    test_command="python eval.py {solution}",
)
```

## Directory Structure

Evolution state is stored in `.evolve-sdk/<problem>/`:

```
.evolve-sdk/<problem>/
├── evolution.json      # Full state (population, history)
├── champion.json       # Best solution manifest
├── benchmark.py        # Auto-generated evaluation harness
├── generations.jsonl   # Per-generation log
└── mutations/
    ├── gen0_a.py       # Initial population
    ├── gen0_b.py
    ├── gen1a.py        # Generation 1 mutations
    ├── gen1x.py        # Crossover
    └── ...
```

## Trust System

The SDK includes a comprehensive trust system to detect and prevent evaluator exploitation:

### Components

| Component | Description |
|-----------|-------------|
| **Adversary Agent** | Reviews suspicious improvements before promotion |
| **Variance Gates** | Re-evaluates N times, rejects inconsistent results |
| **Canary Tests** | Injects known-bad candidates at startup to verify system works |
| **Exploit Detection** | Checks timing anomalies, output integrity, determinism |
| **Trust Dossier** | Generates markdown reports of all trust decisions |
| **Validators** | Pluggable validation chain for escalation levels |

### Configuration

```json
{
  "trust": {
    "enabled": true,
    "accept_threshold": 0.7,
    "suspicious_jump_pct": 15.0,
    "require_adversary_for_champion": true,

    "n_evaluations": 3,
    "variance_threshold": 0.05,
    "require_variance_gate": false,

    "canary_test_enabled": false,
    "canary_test_strict": true,

    "check_timing_anomaly": true,
    "timing_anomaly_threshold_ms": 50.0,
    "check_output_integrity": true,
    "output_max_value": 1000.0,

    "generate_dossier": true,
    "validators": ["default", "extended"]
  }
}
```

### Trust Flow

```
Candidate → Exploit Detection → Variance Gate → Adversary Review → Escalation → Decision
               │                    │                │                │
               ├── NaN/Inf?         ├── CV > 5%?     ├── Trust < 0.4? ├── Level 1-3
               └── Too fast?        └── Inconsistent? └── Suspicious?   └── Extended tests
```

### Trust Dossier

After evolution, a `trust_dossier.md` is generated with:
- Summary statistics (accept/reject/challenge counts)
- Canary test results
- Champion decision history
- Per-evaluation trust scores and flags

## Mode-Specific Guidance

The SDK reads mode-specific guidance from skill files in `.claude/commands/`:

- `evolve-perf.md` - Evaluation contract, acceptance criteria for performance
- `evolve-size.md` - Golf tricks, byte counting for size
- `evolve-ml.md` - Overfitting detection, holdout validation for ML

This allows domain expertise to be maintained separately from the SDK code.

## Requirements

- Python 3.10+
- Claude Agent SDK (`pip install claude-agent-sdk`)
- Authenticated with Claude (`claude auth login`)
