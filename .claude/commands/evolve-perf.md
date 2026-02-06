---
description: Performance subskill for /evolve - optimizes runtime speed for algorithms
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite
argument-hint: <problem description>
---

# /evolve-perf - Performance Evolution Subskill

This is the **performance optimization subskill** for the `/evolve` command. It evolves algorithms for **runtime speed** (ops/sec, latency, throughput).

**Implementation**: Uses the evolve-sdk Python package for hierarchical agent orchestration.

---

## Usage

```bash
/evolve-perf <problem description>
/evolve-perf <problem description> --budget <tokens|generations>
/evolve-perf --resume
```

**Examples**:
```
/evolve-perf sorting algorithm for integers
/evolve-perf bin packing heuristic --budget 50k
/evolve-perf string search --budget 20gen
/evolve-perf --resume
```

---

## Execution Instructions

When this skill is invoked:

### Step 1: Verify SDK Installation

```bash
python3 -c "import evolve_sdk" 2>/dev/null && echo "SDK ready" || echo "SDK missing"
```

If missing, inform user:
```
The evolve-sdk package is not installed. Install with:
  cd <repo-root>/sdk && pip install -e .
```

### Step 2: Check for evolve_config.json

Look for a config file in the current directory:
```bash
ls evolve_config.json 2>/dev/null
```

If found, use `--config` flag. If not, pass problem description directly.

### Step 3: Run Evolution

```bash
# With config file:
python3 -m evolve_sdk --config=evolve_config.json --mode=perf

# Without config file:
python3 -m evolve_sdk "<problem description>" --mode=perf

# Resume previous:
python3 -m evolve_sdk --resume --mode=perf
```

**Available flags**:
| Flag | Default | Description |
|------|---------|-------------|
| `--max-generations=N` | 50 | Maximum generations |
| `--population-size=N` | 10 | Population size |
| `--plateau=N` | 5 | Stop after N gens without improvement |
| `--no-parallel` | false | Run mutations sequentially |
| `--model=X` | claude-opus-4-6 | Model for subagents |

### Step 4: Report Results

After completion:
1. Read `.evolve-sdk/<problem>/champion.json`
2. Summarize: generations, final fitness, champion path
3. Show key innovations and improvement trajectory

---

## Mode-Specific Guidance

The following sections define **performance mode requirements** that the SDK's subagents follow.

---

## Evaluation Contract (Hard Requirements)

These requirements are non-negotiable and must be enforced by the evolution loop:

1. **Three-way split**: Every candidate MUST be evaluated on TRAIN + VALID + HOLDOUT/TEST datasets.
   - TRAIN: Used for fitness scoring and selection
   - VALID: Used for promotion gate (no regression allowed)
   - HOLDOUT/TEST: Never used for selection; reported only for final analysis

2. **Selection vs Promotion**:
   - Selection is based on TRAIN performance only
   - Promotion to champion requires: (a) no regression on VALID mean, (b) meets acceptance criteria below

3. **Determinism Requirements**:
   - Fixed random seeds OR explicit seed lists for all stochastic operations
   - Fixed build mode (always `--release`, no debug builds during eval)
   - Log and record: Rust toolchain version, git commit hash, platform info
   - Command to reproduce any evaluation must be logged

4. **Data Integrity**:
   - TRAIN/VALID/HOLDOUT must be generated once at bootstrap and never modified
   - Store checksums of test data in evolution.json
   - If data changes, evolution must restart from scratch

5. **Generalization Testing** (required for promotion to champion):
   - Champions MUST be validated on multiple input distributions
   - Minimum 3 distributions required; 5 recommended
   - Champion must improve on majority (≥3/5) of distributions
   - Report per-distribution performance in final results

---

## Acceptance Criteria (To Keep a Candidate)

A candidate is accepted into the population only if ALL of the following hold:

1. **TRAIN improvement**: Candidate improves mean TRAIN objective by at least ε (epsilon).
   - Default ε = 0.001 (0.1% relative improvement)

2. **VALID non-regression**: Candidate must not regress on VALID mean.
   - Regression threshold: VALID_new >= VALID_old * 0.995 (allow 0.5% noise margin)

3. **Instance consistency** (at least ONE must hold):
   - Improves on at least K out of N instances (paired comparison), where K = ceil(N * 0.6)
   - OR improves median across all instances

4. **Noise handling**:
   - If TRAIN improvement is within 2× noise floor, rerun evaluation R times (default R=3)
   - Use median of R runs for final decision

5. **Correctness**: Must pass all correctness tests (implicit, always required)

6. **Statistical confidence** (for timing-sensitive benchmarks):
   - Run each evaluation N times (default N=5 for timing, N=1 for deterministic)
   - Report mean ± standard deviation
   - Require improvement > 2σ for acceptance (95% confidence)

---

## Complexity Budget

Evolved algorithms must remain simple and efficient:

### Runtime Complexity
- `priority()` / core function must be **O(1) per element** (no nested loops over input)
- No heap allocations in hot path
- No I/O, no system calls, no threading primitives
- No recursion deeper than O(log n)

### Expression Complexity
Cap maximum complexity to prevent overfitting:
- `max_ast_nodes`: 50
- `max_operations`: 30 (+, -, *, /, pow, sqrt, ln, exp, abs, min, max)
- `max_branches`: 5 (if/else, match arms)
- `max_constants`: 10

### Preference Ordering
Prefer simpler formulations (use as tiebreaker when fitness is equal):
1. Monotonic transforms: prefer `a * x + b` over `a * x^2 + b * x + c`
2. Smooth functions: prefer `ln(x)`, `sqrt(x)` over piecewise/discontinuous
3. Fewer magic constants: prefer derived constants over tuned literals

---

## Diversity Requirements

The evolution must maintain population diversity:

1. **Minimum unique families**: Population of N must have ≥ ceil(N/2) unique algorithm families
2. **Family cap**: No single family can exceed 50% of population
3. **Forced exploration**: When diversity < 0.5, next generation adds "alien" mutations

---

## Mutation Strategies (Performance Mode)

For performance optimization, prioritize these mutation strategies:

1. **Algorithm family changes**: Try fundamentally different approaches
2. **SIMD vectorization**: Add explicit SIMD instructions
3. **Cache optimization**: Improve memory access patterns
4. **Branch elimination**: Remove conditional branches
5. **Loop unrolling**: Unroll inner loops
6. **Unsafe optimizations**: Use unsafe Rust for bounds-check elimination
7. **Lookup tables**: Precompute frequently needed values
8. **Hybrid dispatch**: Use different algorithms based on input size

---

## Crossover Guidelines

When combining two parent solutions:

1. **Identify strengths**: Which inputs does each parent excel on?
2. **Extract innovations**: What unique techniques does each use?
3. **Dispatch strategy**: Can we route different inputs to different approaches?
4. **Constant blending**: Can we blend magic numbers from both?
5. **Feature combination**: Can innovations be combined in single implementation?

---

## Logging & Artifacts

All evolution artifacts are stored in `.evolve-sdk/<problem>/`:

```
.evolve-sdk/<problem>/
├── evolution.json       # Full state (mode, population, history)
├── champion.json        # Best solution manifest
├── generations.jsonl    # Per-generation log (append-only)
├── mutations/           # All tested mutations
└── tool_usage.jsonl     # Agent tool usage log
```

---

## Key Principles

1. **Correctness first**: Invalid solutions get fitness 0
2. **Statistical rigor**: Use multiple runs for timing benchmarks
3. **Generalization**: Test on multiple distributions
4. **Diversity**: Maintain algorithm family diversity
5. **Reproducibility**: Log all seeds, versions, commands
6. **Simplicity**: Prefer simpler solutions when fitness is equal
