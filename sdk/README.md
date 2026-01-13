# Evolve SDK

Python SDK for evolutionary algorithm discovery using the Claude Agent SDK. Powers the `/evolve` family of skills with hierarchical agents, evolution memory, trust validation, and fine-grained control.

## Overview

The SDK orchestrates evolution through specialized subagents:

- **Initializer**: Creates diverse initial population
- **Mutators**: Generate variants of promising solutions (parallel)
- **Crossover**: Combines innovations from multiple parents
- **Evaluator**: Measures fitness against benchmarks
- **Adversary**: Reviews suspicious improvements for trust validation
- **Debugger**: Diagnoses failed mutations, identifies root causes, extracts lessons
- **Plateau Breaker**: Detects stalls, proposes radical interventions to escape local optima

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
├── For each generation:
│   ├── Mutator Agents (parallel)
│   │   └── Each creates one mutation variant
│   ├── Crossover Agent
│   │   └── Combines top solutions
│   ├── Evaluator Agent
│   │   └── Measures fitness of all new solutions
│   └── Adversary Agent (if suspicious)
│       └── Reviews large fitness jumps
│
└── Memory Store
    └── Records mutations, failures, checkpoints
```

## Evolution Memory System

The memory system provides persistent storage for evolution runs, enabling pattern learning, failure avoidance, and crash recovery.

### Frame Types

| Frame Type | Description | Key Fields |
|------------|-------------|------------|
| `mutation` | Successful mutation record | `parent_file`, `child_file`, `fitness_delta_pct`, `tags` |
| `failed_mutation` | Rejected mutation | `failure_reason`, `diff_content` |
| `checkpoint` | Recovery point | `generation`, `population_json`, `champion_fitness` |
| `generation` | Generation summary | `best_fitness`, `mutations_tried`, `mutations_valid` |
| `champion` | Winning solution | `file_path`, `fitness`, `code_content`, `trust_score` |
| `trust_decision` | Adversary review | `recommendation`, `flags`, `analysis` |
| `exploit` | Detected exploit attempt | `pattern_type`, `flags`, `code_preview` |

### Memory Configuration

```json
{
  "memory": {
    "enabled": true,
    "inject_mutation_context": true,
    "store_successful_mutations": true,
    "store_failed_mutations": true,
    "max_similar_mutations": 5,
    "max_failed_mutations": 5
  }
}
```

### Memory Queries

The SDK provides pre-built query patterns:

```python
from evolve_sdk.memory import EvolutionMemory
from evolve_sdk.memory.queries import (
    get_mutation_context,
    get_breakthrough_patterns,
    get_trust_calibration,
)

memory = EvolutionMemory(store_path=".evolve-sdk/problem/evolution.json")

# Get context for mutator agent
context = get_mutation_context(memory, parent_code)
# Returns: similar_successful, failed_to_avoid, high_impact_patterns

# Find patterns that broke plateaus
breakthroughs = get_breakthrough_patterns(memory)

# Calibrate trust thresholds
calibration = get_trust_calibration(memory)
```

### Memory Benefits

| Benefit | Description |
|---------|-------------|
| **Pattern Learning** | Mutators see what worked on similar code |
| **Failure Avoidance** | Don't repeat mutations that already failed |
| **Crash Recovery** | Resume from any checkpoint after system failure |
| **Cross-Problem Transfer** | Apply patterns from one problem to another |
| **Trust Calibration** | Tune thresholds based on historical decisions |

## Inter-Agent Messaging

The SDK includes a messaging system that enables agents to communicate with each other and keep the human operator informed in real-time.

### How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Mutator   │     │  Adversary  │     │   Runner    │
│     🧬      │     │     🛡️      │     │     🎯      │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   ▼
         ┌─────────────────┐
         │  Memory Store   │
         │  (MessageFrame) │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Reporter Agent  │  ◄── Polls every 2s
         │       📢        │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │    Terminal     │  ◄── Live output to operator
         │   (stdout)      │
         └─────────────────┘
```

1. **Agents post messages** to the memory store using `broadcast()`, `notify_human()`, etc.
2. **ReporterAgent** runs as a background task, polling the message queue
3. **Important messages** are printed to the terminal in real-time with colors and emojis

### Message Types

| Type | Emoji | Description |
|------|-------|-------------|
| `milestone` | 🏆 | Significant events (evolution start, new champion, completion) |
| `discovery` | 💡 | Useful findings to share across agents |
| `warning` | ⚠️ | Alert about issues (suspicious jumps, trust failures) |
| `status` | 📊 | Progress updates |
| `strategy` | 🎯 | Agent claiming a mutation strategy |
| `error` | ❌ | Something went wrong |

### Priority Levels

| Priority | Displayed | Description |
|----------|-----------|-------------|
| `critical` | Always | Evolution-stopping issues |
| `urgent` | Always | Needs immediate attention |
| `important` | Always | Should be noticed |
| `info` | Default | Normal updates |
| `debug` | Hidden | Verbose logging |

### Example Output

During evolution, operators see messages like:

```
  📢 [REPORTER] Reporter agent started - monitoring message queue

  🏆 [MILESTONE] Evolution started: N-Queens optimization
     Mode: perf, Max generations: 10

  🧬 [MUTATOR] Trying bitwise optimization on gen1b.py
     [Gen 2]

  🛡️ [ADVERSARY] Warning: Suspicious fitness jump detected
     +75% exceeds 20% threshold
     [Gen 2 | Fitness: 892.45]

  🏆 [MILESTONE] New champion: gen3a.py
     Fitness: 542.99

  📢 [REPORTER] Reporter agent stopped - displayed 5 messages, filtered 2
```

### Using the Messaging API

```python
from evolve_sdk.memory import EvolutionMemory

memory = EvolutionMemory(store_path=".evolve-sdk/problem/memory.json")

# Broadcast a message to all agents and human
memory.broadcast(
    from_agent="mutator_a",
    message_type="discovery",
    title="Inlining gave +26% improvement",
    content="Function call overhead dominates. Inline hot functions.",
    priority="important",
    generation=3,
)

# Announce a significant milestone
memory.announce_milestone(
    title="New champion crowned!",
    content="gen5b.py achieves 20,407 solutions/sec",
    generation=5,
    related_fitness=20407.0,
)

# Warn about issues
memory.warn(
    title="Timing anomaly detected",
    content="Solution runs in 0.1ms - suspiciously fast",
    generation=4,
)

# Get recent messages for display
messages = memory.get_human_messages(limit=20, min_priority="info")
```

### Reporter Agent Integration

The reporter agent starts automatically when memory is enabled:

```python
from evolve_sdk.agents import ReporterAgent

# Manually control the reporter
reporter = ReporterAgent(
    memory=memory,
    poll_interval=2.0,      # Check every 2 seconds
    min_priority="info",    # Filter out debug messages
)

await reporter.start()
# ... evolution runs, messages auto-display ...
await reporter.stop()

# Pause during human interaction (e.g., escalation prompts)
reporter.pause()
# ... get user input ...
reporter.resume()
```

## Diagnostic Agents

The SDK includes specialized agents that help evolution runs avoid wasted effort and escape local optima.

### Debugger Agent

When a mutation fails (crash, error, timeout), the Debugger Agent analyzes the failure to:
- Identify the root cause
- Categorize the failure type for pattern learning
- Extract lessons for future mutators

```python
from evolve_sdk.agents import get_debugger_prompt, get_debugger_summary_prompt

# Single failure analysis
prompt = get_debugger_prompt(
    failed_file="gen3b.py",
    error_message="IndexError: list index out of range",
    error_traceback="...",
    parent_file="gen2a.py",
    mutation_type="boundary_optimization",
    mode="perf",
    generation=3,
)

# Multiple failures in one generation (pattern detection)
summary_prompt = get_debugger_summary_prompt(
    failures=[
        {"file": "gen4a.py", "error": "bad character range z-a"},
        {"file": "gen4b.py", "error": "unterminated character set"},
    ],
    generation=4,
    mode="perf",
)
```

**Failure Categories:**
- `syntax_error` - Invalid code syntax
- `type_error` - Type mismatches
- `boundary_condition` - Off-by-one, index out of range
- `resource_exhaustion` - Memory, recursion limits
- `algorithmic_flaw` - Logic errors, incorrect results
- `runtime_crash` - Unhandled exceptions
- And 6 more...

### Plateau Breaker Agent

When evolution stalls (low improvement over multiple generations), the Plateau Breaker diagnoses the situation and proposes radical interventions.

```python
from evolve_sdk.agents import (
    detect_plateau,
    get_plateau_breaker_prompt,
    get_intervention_prompt,
)

# Check if evolution is stuck
fitness_history = [
    {"generation": 1, "best_fitness": 964.0, "improvement_pct": 0.0},
    {"generation": 2, "best_fitness": 964.0, "improvement_pct": 0.0},
    {"generation": 3, "best_fitness": 964.0, "improvement_pct": 0.0},
]

is_stalled, gens_stalled, avg_improvement = detect_plateau(
    fitness_history,
    threshold=0.02,  # <2% improvement
    window=3,        # Over 3 generations
)

if is_stalled:
    # Get diagnosis and interventions
    prompt = get_plateau_breaker_prompt(
        current_champion_code="...",
        fitness_history=fitness_history,
        mutation_history=["parameter_tweak", "loop_unroll", ...],
        mode="perf",
    )

    # After getting LLM response, apply intervention
    intervention_prompt = get_intervention_prompt(
        diagnosis="Local optimum - incremental changes exhausted",
        intervention_type="algorithm_swap",
        intervention_details="Try constraint propagation instead of backtracking",
        current_champion_code="...",
        mode="perf",
    )
```

**Intervention Types:**
- `algorithm_swap` - Replace current algorithm entirely
- `paradigm_shift` - Change fundamental approach
- `structural` - Major code reorganization
- `hyperparameter_reset` - Reset to explore different region
- `population_injection` - Add diverse orthogonal solutions

### Showcase: Regex Golf

The top-level `showcase/regex_golf/` directory demonstrates both diagnostic agents on a real problem.

**Problem:** Find the shortest regex matching Star Wars titles but not Star Trek titles.

**Results:**
- Debugger caught 33% of mutations before they wasted evaluation cycles
- Plateau Breaker detected stall at generation 4 and proposed breakthrough
- Fitness improved from 964 to 977 (+13 points, 36% shorter regex)

See `showcase/regex_golf/README.md` for detailed analysis.

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
| **Escalation Levels** | Extended validation for high-stakes promotions |

### Trust Configuration

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

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | size | Optimization mode (size, perf, ml) |
| `--max-generations` | 50 | Maximum generations |
| `--population-size` | 10 | Population size |
| `--plateau` | 5 | Stop after N gens without improvement |
| `--no-parallel` | false | Run mutations sequentially |
| `--model` | claude-opus-4-5-20251101 | Model for subagents (always use Opus 4.5 for best quality) |
| `--config` | - | Path to evolve_config.json |

### Config File

```json
{
  "description": "Evolve fast N-Queens solvers",
  "mode": "perf",
  "evaluation": {
    "test_command": "python evaluate.py {solution} --json"
  },
  "memory": {
    "enabled": true,
    "inject_mutation_context": true,
    "store_successful_mutations": true,
    "store_failed_mutations": true
  },
  "trust": {
    "enabled": true,
    "suspicious_jump_pct": 50.0,
    "require_adversary_for_champion": true
  },
  "starter_solutions": ["baseline.py"],
  "optimization_strategies": [
    {"name": "constraint_propagation", "description": "Add constraint propagation"},
    {"name": "bitwise_operations", "description": "Use bitwise ops for faster conflict detection"}
  ],
  "constraints": [
    "Must be pure Python (no numpy/external deps)",
    "Must handle any board size N >= 4"
  ]
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
├── evolution.json      # Full state + memory frames
├── champion.json       # Best solution manifest
├── trust_dossier.md    # Trust decision report
├── benchmark.py        # Auto-generated evaluation harness
└── mutations/
    ├── gen0_a.py       # Initial population
    ├── gen0_b.py
    ├── gen1a.py        # Generation 1 mutations
    ├── gen1x.py        # Crossover
    └── ...
```

## Module Structure

```
evolve_sdk/
├── __init__.py         # Package exports
├── __main__.py         # CLI entry point
├── runner.py           # EvolutionRunner orchestrator
├── config.py           # Configuration handling
├── agents/             # Subagent implementations
│   ├── mutator.py      # Mutation specialist
│   ├── evaluator.py    # Fitness measurement
│   ├── crossover.py    # Parent combination
│   ├── adversary.py    # Trust validation
│   ├── debugger.py     # Failed mutation diagnosis
│   └── plateau_breaker.py  # Stall detection and intervention
├── memory/             # Evolution memory system
│   ├── __init__.py     # Memory exports
│   ├── store.py        # Persistent storage engine
│   ├── schemas.py      # Frame type definitions
│   ├── queries.py      # Pre-built query patterns
│   └── embeddings.py   # Code similarity matching
└── hooks/              # Validation hooks
```

## Mode-Specific Guidance

The SDK reads mode-specific guidance from skill files in `.claude/commands/`:

- `evolve-perf.md` - Evaluation contract, acceptance criteria for performance
- `evolve-size.md` - Golf tricks, byte counting for size
- `evolve-ml.md` - Overfitting detection, holdout validation for ML

This allows domain expertise to be maintained separately from the SDK code.

## Example: N-Queens with Memory

```bash
# Create config
cat > evolve_config.json << 'EOF'
{
  "description": "Evolve fast N-Queens solvers",
  "mode": "perf",
  "evaluation": {"test_command": "python evaluate.py {solution} --json"},
  "memory": {"enabled": true, "inject_mutation_context": true},
  "trust": {"enabled": true, "require_adversary_for_champion": true},
  "starter_solutions": ["baseline.py"]
}
EOF

# Run evolution
python -m evolve_sdk --config evolve_config.json --population-size 4 --max-generations 10

# Check results
cat .evolve-sdk/*/champion.json
```

## Requirements

- Python 3.10+
- Claude Agent SDK (`pip install claude-agent-sdk`)
- Authenticated with Claude (`claude auth login`)

### Model Selection

**The SDK always defaults to Claude Opus 4.5 (`claude-opus-4-5-20251101`) for all subagents.**

This is intentional - evolution quality depends heavily on the reasoning capabilities of the agents performing mutations, evaluations, and adversarial review. Opus 4.5 provides:

- Superior code understanding and generation
- Better reasoning about optimization strategies
- More reliable adversarial analysis (trust system)
- Higher quality crossover decisions

While you can override with `--model`, we strongly recommend keeping the Opus 4.5 default for best results.
