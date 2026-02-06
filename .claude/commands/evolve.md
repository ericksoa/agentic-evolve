---
description: Evolve novel algorithms through LLM-driven mutation, crossover, and selection
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite
argument-hint: <problem description>
---

# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection with **true genetic recombination**. Runs adaptively—continuing while improvement is possible, stopping when plateaued.

**Implementation**: Uses the evolve-sdk Python package for hierarchical agent orchestration.

---

## Available Modes

| Mode | Optimizes | Use When |
|------|-----------|----------|
| **perf** | Runtime speed (ops/sec, latency) | Faster algorithms, benchmarks |
| **size** | Length (bytes, chars) | Code golf, minimal configs |
| **ml** | Model accuracy (F1, loss) | ML optimization |

---

## Usage

```bash
/evolve <problem description>
/evolve <problem description> --mode=<perf|size|ml>
/evolve --resume
```

**Examples**:
```
/evolve faster sorting algorithm to beat std::sort
/evolve shortest Python solution for ARC task
/evolve improve F1 score for classification
/evolve --resume
```

---

## Execution Instructions

You are the master `/evolve` skill. Your job is to:
1. Detect the optimization mode
2. Run evolution via the SDK

### Step 1: Check for Explicit Override

If the request contains `--mode=perf`, `--mode=size`, or `--mode=ml`, use that mode directly.

### Step 2: Check for Resume

If the request is `--resume`:
1. Search for `.evolve-sdk/*/evolution.json`
2. Read it to get the mode from the `"mode"` field
3. Run SDK with `--resume`

### Step 3: Detect Mode from Intent

**Choose SIZE mode** when goal is minimizing length:
- "shortest", "smallest", "fewest bytes", "most concise"
- ARC-AGI tasks, code golf
- Reducing file/config/prompt size

**Choose PERF mode** when goal is maximizing speed:
- "faster", "optimize", "speed up", "high performance"
- Beating benchmarks, improving throughput
- Runtime optimization

**Choose ML mode** when goal is improving model metrics:
- "accuracy", "F1 score", "precision", "recall"
- Classification/regression tasks
- Model training, hyperparameter tuning

### Step 4: Handle Ambiguity

If genuinely ambiguous (e.g., just "optimize"), ask:
```
What are we optimizing for?
- Fastest runtime (speed) → perf
- Smallest code (bytes) → size
- Best accuracy (ML) → ml
```

### Step 5: Verify SDK Installation

```bash
python3 -c "import evolve_sdk" 2>/dev/null && echo "SDK ready" || echo "SDK missing"
```

If missing:
```
The evolve-sdk package is not installed. Install with:
  cd <repo-root>/sdk && pip install -e .
```

### Step 6: Check for evolve_config.json

```bash
ls evolve_config.json 2>/dev/null
```

If found, use `--config` flag.

### Step 7: Run Evolution

```bash
# With config file:
python3 -m evolve_sdk --config=evolve_config.json --mode=<detected_mode>

# Without config file:
python3 -m evolve_sdk "<problem description>" --mode=<detected_mode>

# Resume:
python3 -m evolve_sdk --resume
```

**Available flags**:
| Flag | Default | Description |
|------|---------|-------------|
| `--max-generations=N` | 50 | Maximum generations |
| `--population-size=N` | 10 | Population size |
| `--plateau=N` | 5 | Stop after N gens without improvement |
| `--no-parallel` | false | Run mutations sequentially |
| `--model=X` | claude-opus-4-6 | Model for subagents |

### Step 8: Report Results

After completion:
1. Read `.evolve-sdk/<problem>/evolution.json`
2. Summarize: generations run, final fitness, champion path
3. Show key innovations and improvement trajectory

---

## Mode Detection Examples

| Request | Mode | Reasoning |
|---------|------|-----------|
| "shortest Python solution for ARC task" | size | "shortest" = minimize length |
| "faster sorting algorithm to beat std::sort" | perf | "faster" = maximize speed |
| "improve accuracy on this classification task" | ml | "accuracy" = model metrics |
| "optimize this function" | **ask** | "optimize" is ambiguous |
| "--mode=size optimize this" | size | Explicit override |

---

## Core Features

1. **Population-based**: Maintains diverse solutions, not just the winner
2. **Semantic crossover**: Combines innovations from multiple parents
3. **Adaptive generations**: Continues while improving, stops on plateau
4. **Budget control**: User sets token/generation limits
5. **Checkpointing**: Resume evolution from where you left off
6. **Correctness first**: Invalid solutions get fitness 0
7. **Trust-aware validation**: Adversary agent challenges suspicious improvements

---

## Trust-Aware Evolution

The evolution system includes an **Adversary agent** that challenges candidate solutions before they're promoted. This prevents the evolution from exploiting evaluator blind spots or claiming false improvements.

### How It Works

```
Mutation → Evaluation → ADVERSARY CHALLENGE → Selection
                             │
                             ├── Trust score (0.0-1.0)
                             ├── Recommendation (accept/challenge/reject)
                             └── Escalation level (0-3 for deeper validation)
```

### Trust Triggers

The Adversary reviews candidates when:
- **Suspicious jump**: Fitness improves >20% in a single generation
- **New champion**: Any candidate that would become the new best solution

### Escalation Levels

| Level | Validation | Use Case |
|-------|------------|----------|
| 0 | Basic evaluator | Normal mutations |
| 1 | Extended tests | Edge cases, boundary conditions |
| 2 | Out-of-distribution | Unseen data patterns |
| 3 | Gold standard | Full reference validation |

### Configuration

Add `trust` section to `evolve_config.json`:

```json
{
  "description": "...",
  "mode": "perf",
  "trust": {
    "enabled": true,
    "accept_threshold": 0.7,
    "suspicious_jump_pct": 20.0,
    "max_escalation_level": 2,
    "extended_test_command": "python validate.py {solution} --thorough"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | true | Enable/disable adversary |
| `accept_threshold` | 0.7 | Trust >= this: accept without escalation |
| `challenge_threshold` | 0.4 | Trust >= this: escalate before decision |
| `reject_threshold` | 0.4 | Trust < this: reject outright |
| `suspicious_jump_pct` | 20.0 | Improvement threshold triggering review |
| `apply_trust_adjustment` | true | Multiply fitness by trust_score |
| `require_adversary_for_champion` | true | New champions must pass adversary |

### Red Flags (Lower Trust)

- Fitness jump >20% in a single mutation
- Hardcoded constants matching test cases
- Pattern matching instead of algorithms
- Suspiciously fast execution (cached results)
- No clear algorithmic improvement

### Green Flags (Higher Trust)

- Incremental improvements (<10% per generation)
- Clear algorithmic innovation in diff
- Performance scales correctly with input size
- Improvement consistent across test cases

---

## Baseline Selection: The Fitness Landscape

**Critical insight**: Starting with a near-optimal solution can be counterproductive.

### Why Suboptimal Baselines Often Work Better

Evolution navigates a "fitness landscape" - imagine terrain where height = fitness:

**Near-optimal start (local peak)**:
- Small mutations either break correctness or don't improve
- Remaining optimizations require simultaneous changes
- Evolution gets stuck at a "local maximum"

**Suboptimal start (valley with uphill paths)**:
- Many single mutations can improve fitness
- Diverse population emerges from different improvement paths
- Crossover combines independently-discovered optimizations
- Broader exploration of the search space

### Guidelines for Baseline Selection

| Scenario | Recommendation |
|----------|----------------|
| Known optimal exists | Start 20-50% worse to allow exploration |
| Competing algorithms | Start with the simpler/worse one |
| Hitting plateau quickly | Try a more naive baseline |
| Many invalid mutations | Baseline may be too optimized already |

**Example**: For sorting networks (n=16), starting from:
- Odd-even mergesort (63 comparators, near-optimal) → Plateau quickly
- Bubble sort (120 comparators, 2x optimal) → Rich optimization landscape

### When to Use Near-Optimal Baselines

- When you need to find **alternative** optimal solutions (not better ones)
- When exploring minor variations on a known-good approach
- When the fitness function rewards diversity, not just the metric

---

## Directory Structure

All modes use `.evolve-sdk/<problem>/`:

```
.evolve-sdk/<problem>/
├── evolution.json       # Full state (mode, population, history)
├── champion.json        # Best solution manifest
├── generations.jsonl    # Per-generation log (append-only)
├── mutations/           # All tested mutations
└── benchmark.py         # Auto-generated evaluation harness
```

---

## Subskill Reference

For mode-specific details, see:
- `/evolve-perf` - Performance optimization (ops/sec, latency)
- `/evolve-size` - Size optimization (bytes, characters)
- `/evolve-ml` - ML optimization (F1, accuracy)

Each contains domain-specific guidance for evaluation, mutation strategies, and acceptance criteria.

---

## Quick Reference

| Want to... | Command |
|------------|---------|
| Make code faster | `/evolve faster <algorithm>` |
| Make code shorter | `/evolve shortest <code>` |
| Improve ML model | `/evolve improve accuracy <task>` |
| Continue previous | `/evolve --resume` |
| Force specific mode | `/evolve --mode=<mode> <problem>` |
