# Evolve

Evolutionary algorithm discovery for Claude Code. Evolves novel solutions to hard programming problems through LLM-driven mutation with Rust benchmarks for precise performance measurement.

## How It Works

The `/evolve` skill uses **parallel Claude Code agents** for true genetic algorithm evolution:

### Agent-Based Architecture
- **8 parallel agents per generation**: Each explores a different mutation strategy
- **Semantic crossover**: Agents combine innovations from parent solutions
- **Adaptive stopping**: Runs until plateau (3 gens without >0.5% improvement)
- **Budget control**: `--budget 10k|50k|100k|unlimited`

### Generation 1: Divergent Exploration
- Spawn 8 mutation agents in parallel (tweak, unroll, specialize, vectorize, memoize, restructure, hybrid, alien)
- Extract **innovations** from each solution (what makes it fast?)
- Select top 4 with **diversity pressure** (max 2 from same algorithm family)

### Generation 2+: Crossover + Mutation
- **4 crossover agents**: Combine innovations from parent pairs
- **4 mutation agents**: Refine top performers
- Elitism: Never lose the champion
- Checkpoint state to `evolution.json` for resume

### Why Crossover Matters
```
Gen 1: Discovers radix sort (fast distribution) + quicksort (fast partition)
Gen 2: Crossover → radix+quick hybrid (uses both techniques)
Gen 3: Adds insertion sort base case from shellsort lineage
```

Each generation **combines innovations** rather than just refining one approach.

## Quick Start

### 1. Install the Skill

```bash
# Option A: Clone and copy skill
git clone https://github.com/ericksoa/openevolve-claude
cp openevolve-claude/.claude/commands/evolve.md ~/.claude/commands/

# Option B: Direct download
curl -o ~/.claude/commands/evolve.md \
  https://raw.githubusercontent.com/ericksoa/openevolve-claude/main/.claude/commands/evolve.md
```

### 2. Use It

```bash
claude
> /evolve sorting algorithm for integers
> /evolve fibonacci - beat naive recursion --budget 10k
> /evolve string search --budget 50k
> /evolve --resume   # Continue previous evolution
```

The skill will:
- Check for Rust toolchain (offers to install via rustup)
- Generate Rust benchmark infrastructure
- Run evolution with adaptive stopping (or until budget exhausted)
- Report the champion algorithm

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code /evolve                          │
│         "Optimize sorting algorithm for integers"                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION 1: Exploration                     │
│                                                                  │
│  8 Parallel Mutation Agents:                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                   │
│  │ radix  │ │ quick  │ │ heap   │ │ shell  │ ...               │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘                   │
│      │          │          │          │                          │
│      ▼          ▼          ▼          ▼                          │
│  [Evaluate] [Evaluate] [Evaluate] [Evaluate]                    │
│      │          │          │          │                          │
│      ▼          ▼          ▼          ▼                          │
│  [Extract Innovations: what makes each solution fast?]          │
│                                                                  │
│  Select Top 4 with Diversity (different algorithm families)     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               GENERATION 2+: Crossover + Mutation                │
│                                                                  │
│  Population: [radix, quicksort, heapsort, shellsort]            │
│                                                                  │
│  CROSSOVER (4 agents):              MUTATION (4 agents):        │
│  ┌──────────────────┐               ┌──────────────────┐        │
│  │ radix × quick    │               │ tweak(radix)     │        │
│  │ radix × heap     │               │ specialize(radix)│        │
│  │ quick × shell    │               │ vectorize(quick) │        │
│  │ heap × shell     │               │ unroll(quick)    │        │
│  └────────┬─────────┘               └────────┬─────────┘        │
│           │                                   │                  │
│           └───────────┬───────────────────────┘                  │
│                       ▼                                          │
│           [Evaluate 8 offspring]                                │
│           [Extract innovations]                                  │
│           [Select top 4 + elitism]                              │
│                                                                  │
│  Repeat until plateau or budget exhausted...                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Rust Evaluator                               │
│                                                                  │
│  1. Copy evolved code → rust/src/evolved.rs                     │
│  2. cargo build --release (LTO, opt-level=3)                    │
│  3. Run benchmarks against baselines                            │
│  4. Return JSON: { fitness, ops_per_second, correctness }       │
└─────────────────────────────────────────────────────────────────┘
```

## Demo: Bubble Sort → 71x Faster

Watch `/evolve` transform a naive O(n²) bubble sort into a fast O(n) radix sort:

```
> /evolve sorting algorithm for integers

Starting evolution...

Baseline:
  bubble:       1,289 ops/sec  ← The slow algorithm
  std:        114,592 ops/sec
  std_unstable: 168,417 ops/sec

Generation 1: Spawning 8 mutations...
  ✓ quicksort:   89,234 ops/sec
  ✓ radix:      156,892 ops/sec  ← Winner!
  ✓ heapsort:    78,456 ops/sec
  ...

Champion: Radix sort with sign-bit handling
  91,835 ops/sec → 71x faster than bubble sort!
```

The evolved algorithm:
- Uses 11-bit radix sort for large arrays
- Falls back to insertion sort for small arrays (≤64)
- Handles signed integers via sign-bit flipping

See [`showcase/sort-demo/`](showcase/sort-demo/) for the full benchmark.

## Results

| Problem | Champion | Improvement |
|---------|----------|-------------|
| **Fibonacci** | Unsafe lookup table | **834M ops/sec** (30x vs iterative) |
| **Sorting** | Radix sort | **71x** faster than bubble sort |
| Integer parsing | Custom parser | +51% vs std |
| String search | Rarebyte+memchr | +27% vs Boyer-Moore (scalar) |

## Fitness Function

```python
# Base: speed ratio to best baseline
speed_ratio = evolved_speed / best_baseline_speed

# Scale to 0-1, cap at 2x improvement
fitness = min(speed_ratio, 2.0) / 2.0

# Bonus for beating all baselines
if evolved_speed > best_baseline_speed:
    fitness = min(fitness + 0.1, 1.0)

# Correctness gate: 0 if tests fail
if not correctness:
    fitness = 0.0
```

## Why Rust Benchmarks?

- **No JIT warmup**: Consistent timing from first run
- **No GC pauses**: Predictable performance
- **Native speed**: Measure algorithmic improvements, not runtime overhead
- **LTO + codegen-units=1**: Maximum optimization for fair comparison

## Project Structure

```
.evolve/<problem>/           # Created per evolution
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Trait definition
│       ├── baselines.rs     # Known algorithms to beat
│       ├── evolved.rs       # Champion code
│       └── benchmark.rs     # Performance measurement
├── evaluator.py             # Fitness evaluation
├── evolution.json           # Checkpoint for resume
└── mutations/               # All tested mutations
```

## Credits

- Inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/) concepts
- Built with [Claude Code](https://claude.ai/code) by Anthropic

## License

MIT
