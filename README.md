# Agentic Evolve

**Evolutionary algorithm discovery powered by Claude.** Evolves novel solutions through LLM-driven mutation, crossover, and selection—optimizing for speed, size, or ML accuracy.

## Features

- **Three optimization modes**: Performance (ops/sec), Size (bytes), ML (F1/accuracy)
- **Hierarchical agents**: Dedicated subagents for mutation, crossover, and evaluation
- **Clean context**: Each agent starts fresh, avoiding context bloat
- **Parallel mutations**: Run multiple mutation attempts concurrently
- **Validation hooks**: Block unsafe code patterns before execution
- **Resume support**: Checkpoint and continue evolution across sessions

## Quick Start

### 1. Install the SDK

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the SDK and dependencies
pip install -e sdk/
pip install claude-agent-sdk
```

### 2. Install the Skills (optional)

```bash
# Copy skills to your Claude commands directory
cp .claude/commands/evolve*.md ~/.claude/commands/
```

### 3. Use It

**Via CLI:**
```bash
# Activate venv first
source .venv/bin/activate

# Performance optimization
python -m evolve_sdk "faster sorting algorithm" --mode=perf

# Size optimization (code golf)
python -m evolve_sdk "shortest Python prime checker" --mode=size

# ML optimization
python -m evolve_sdk "improve F1 for classification" --mode=ml

# Resume previous evolution
python -m evolve_sdk --resume
```

**Via Claude Code skill:**
```
/evolve faster sorting algorithm
/evolve shortest Python solution for ARC task
/evolve improve accuracy on this classifier
/evolve --resume
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    /evolve <problem>                             │
│                                                                  │
│  1. Detect mode (perf/size/ml) from intent                      │
│  2. Run: python -m evolve_sdk "<problem>" --mode=<mode>         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EvolutionRunner (SDK)                         │
│                                                                  │
│  ┌─────────────┐                                                │
│  │ Initializer │ → Creates diverse initial population           │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Generation Loop                             │    │
│  │                                                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │    │
│  │  │ Mutator  │  │ Mutator  │  │ Crossover│  (parallel)   │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘               │    │
│  │       └─────────────┴─────────────┘                      │    │
│  │                     │                                    │    │
│  │                     ▼                                    │    │
│  │              ┌───────────┐                               │    │
│  │              │ Evaluator │ → Measure fitness             │    │
│  │              └───────────┘                               │    │
│  │                     │                                    │    │
│  │              Select top solutions, repeat                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Stop when: plateau OR max generations OR budget exhausted      │
└─────────────────────────────────────────────────────────────────┘
```

## Optimization Modes

| Mode | Metric | Use Case |
|------|--------|----------|
| **perf** | ops/sec, latency | Algorithm optimization, benchmarks |
| **size** | bytes, characters | Code golf, minimal implementations |
| **ml** | F1, accuracy, AUC | Feature engineering, model tuning |

## Project Structure

```
agentic-evolve/
├── .claude/commands/        # Skill files (thin SDK wrappers)
│   ├── evolve.md           # Master dispatcher
│   ├── evolve-perf.md      # Performance mode
│   ├── evolve-size.md      # Size mode
│   └── evolve-ml.md        # ML mode
├── sdk/                     # Python SDK
│   └── evolve_sdk/
│       ├── runner.py       # EvolutionRunner orchestrator
│       ├── agents/         # Subagent prompts
│       └── hooks/          # Validation hooks
├── showcase/                # Example evolution runs
│   ├── santa-2025-packing/ # Kaggle bin packing
│   ├── code-golf/          # ARC-AGI solutions
│   └── ...
└── .evolve-sdk/             # Evolution state (created per run)
    └── <problem>/
        ├── evolution.json   # Full state
        ├── champion.json    # Best solution
        └── mutations/       # All tested variants
```

## Example Results

| Problem | Mode | Result | Improvement |
|---------|------|--------|-------------|
| Fibonacci | perf | 834M ops/sec | 30x vs iterative |
| Prime checker | perf | 9.5M ops/sec | Sieve + 6k±1 |
| ARC task 0520fde7 | size | 57 bytes | -29% from baseline |
| TDE classification | ml | 0.76 F1 | +12% from baseline |

## Configuration

Use `evolve_config.json` for custom evaluation:

```json
{
  "problem": "optimize sorting",
  "mode": "perf",
  "test_command": "python benchmark.py {solution}",
  "max_generations": 20,
  "population_size": 10
}
```

Then run:
```bash
python -m evolve_sdk --config=evolve_config.json
```

## Requirements

- Python 3.10+
- Claude Code CLI (`brew install claude-code`)
- Claude Agent SDK (`pip install claude-agent-sdk`)
- Authenticated with Claude (`claude auth login`)

## License

MIT
