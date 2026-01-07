# Evolve SDK

Agent SDK-powered evolutionary algorithm discovery with hierarchical agents, clean context per generation, and fine-grained control.

## Overview

This is an alternative implementation of `/evolve` that uses the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) for:

- **Hierarchical agents**: Dedicated subagents for mutation, crossover, and evaluation
- **Clean context**: Each subagent starts fresh, avoiding context bloat
- **Parallel mutations**: Run multiple mutation attempts concurrently
- **Validation hooks**: Block unsafe code before it executes
- **Structured logging**: Full audit trail of all tool usage

## Installation

```bash
# From the sdk/ directory
pip install -e .

# With Agent SDK (required for actual evolution)
pip install -e ".[sdk]"

# For development
pip install -e ".[dev]"
```

## Usage

### As CLI

```bash
# Basic usage
python -m evolve_sdk "shortest Python sort" --mode=size

# With options
python -m evolve_sdk "faster string search" \
    --mode=perf \
    --max-generations=100 \
    --population-size=20

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

### Via Skill (once integrated)

```bash
# Uses SDK under the hood
/evolve-sdk shortest Python sort
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

Each agent runs with **clean context** - they only see their specific task, not the full evolution history. This prevents context bloat and keeps agents focused.

## Configuration

```python
from evolve_sdk import EvolutionConfig

config = EvolutionConfig(
    problem="...",
    mode="size",                    # size, perf, or ml
    max_generations=50,             # max generations
    plateau_threshold=5,            # stop after N gens without improvement
    population_size=10,             # solutions to maintain
    elite_count=3,                  # top N to mutate each gen
    mutation_variants=4,            # mutations per generation
    parallel_mutations=True,        # run mutations concurrently
    enable_validation_hooks=True,   # block unsafe code
    blocked_imports=["os.system"],  # patterns to block
)
```

## Comparison with Standard `/evolve`

| Feature | `/evolve` (skill) | `/evolve-sdk` |
|---------|-------------------|---------------|
| Context management | Shared context | Clean per-agent |
| Parallel mutations | Sequential | Concurrent |
| Validation hooks | None | Pre-execution blocking |
| Structured logging | Basic | Full audit trail |
| Programmatic control | Limited | Full Python control |
| Distribution | Skill file | pip package |

## Directory Structure

```
.evolve-sdk/<problem>/
├── evolution.json      # Full state (population, history)
├── champion.json       # Best solution
├── tool_usage.jsonl    # Audit log of all tool calls
└── mutations/
    ├── gen0_a.py       # Initial population
    ├── gen0_b.py
    ├── gen1a.py        # Generation 1 mutations
    ├── gen1b.py
    ├── gen1x.py        # Crossover
    └── ...
```

## Requirements

- Python 3.10+
- Claude Code CLI installed (`brew install claude-code`)
- Claude Agent SDK (`pip install claude-agent-sdk`)
- Authenticated with Claude (`claude auth login`)

## Coexistence with `/evolve`

This SDK version is completely separate from the standard `/evolve` skill:

- Different directory: `.evolve-sdk/` vs `.evolve/`
- Different skill name: `/evolve-sdk` vs `/evolve`
- No shared code or state

You can use both in the same project without conflicts.
