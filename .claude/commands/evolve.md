---
description: Evolve novel algorithms through LLM-driven mutation, crossover, and selection
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, WebSearch, WebFetch, AskUserQuestion, Skill
argument-hint: <problem description>
---

# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection with **true genetic recombination**. Runs adaptively—continuing while improvement is possible, stopping when plateaued.

This is the **master skill** that auto-detects the appropriate mode and delegates to specialized subskills.

---

## Available Modes

| Mode | Subskill | Optimizes | Use When |
|------|----------|-----------|----------|
| **perf** | `/evolve-perf` | Runtime speed (ops/sec, latency) | Faster algorithms, benchmarks |
| **size** | `/evolve-size` | Length (bytes, chars) | Code golf, minimal configs |
| **ml** | `/evolve-ml` | Model accuracy (F1, loss) | ML optimization (coming soon) |

---

## Usage

```bash
/evolve <problem description>
/evolve <problem description> --mode=<perf|size|ml>
/evolve --resume
```

The mode is **auto-detected** by default based on keywords and context. Use `--mode=` to override.

### Examples

```bash
# Auto-detected as PERF mode
/evolve faster sorting algorithm
/evolve bin packing heuristic to beat FunSearch
/evolve optimize throughput for this parser

# Auto-detected as SIZE mode
/evolve shortest Python solution for ARC task 0520fde7
/evolve minimize bytes in this function
/evolve most concise git commit rules

# Explicit mode override
/evolve --mode=size minimize this Rust code
/evolve --mode=perf optimize this Python function

# Resume previous evolution
/evolve --resume
```

---

## Auto-Detection Algorithm

The master skill analyzes your request to determine the best mode:

### Keyword Scoring

| Mode | Keywords (weight) |
|------|-------------------|
| **size** | shortest (2), smallest (2), bytes (2), minimize code (2), ARC (2), code golf (3), byte count (2), code size (3), concise (2), minimal (2) |
| **perf** | fastest (2), speed (2), performance (2), throughput (2), latency (2), ops/sec (3), benchmark (2), ops per second (3), faster (2) |
| **ml** | accuracy (2), model (1), train (1), loss (2), predict (2), classify (2), neural (2), kaggle (2), F1 (2), epoch (2) |

### Context Signals

| Signal | Detected Mode |
|--------|---------------|
| Files in `code-golf/` directory | size |
| Files named `tasks/*.json` (ARC format) | size |
| Files with `.rs` + `benchmark` in path | perf |
| Files with `.h5`, `.pt`, `.pkl` extensions | ml |
| `evaluator.py` measuring bytes | size |
| `benchmark.rs` or perf harness | perf |

### Detection Flow

```python
def detect_mode(request: str, context: dict) -> str:
    """Auto-detect evolution mode from request."""

    # 1. Explicit override always wins
    if "--mode=" in request:
        return extract_mode(request)

    # 2. Score keywords
    scores = {"size": 0, "perf": 0, "ml": 0}

    size_signals = [
        ("shortest", 2), ("smallest", 2), ("bytes", 2),
        ("minimize code", 2), ("ARC", 2), ("code golf", 3),
        ("byte count", 2), ("code size", 3), ("concise", 2), ("minimal", 2)
    ]
    perf_signals = [
        ("fastest", 2), ("speed", 2), ("performance", 2),
        ("throughput", 2), ("latency", 2), ("ops/sec", 3),
        ("benchmark", 2), ("ops per second", 3), ("faster", 2)
    ]
    ml_signals = [
        ("accuracy", 2), ("model", 1), ("train", 1), ("loss", 2),
        ("predict", 2), ("classify", 2), ("neural", 2), ("kaggle", 2)
    ]

    request_lower = request.lower()
    for keyword, weight in size_signals:
        if keyword.lower() in request_lower:
            scores["size"] += weight
    for keyword, weight in perf_signals:
        if keyword.lower() in request_lower:
            scores["perf"] += weight
    for keyword, weight in ml_signals:
        if keyword.lower() in request_lower:
            scores["ml"] += weight

    # 3. Context signals (files, directories)
    if context.get("files"):
        files = context["files"]
        if any("tasks/" in f and f.endswith(".json") for f in files):
            scores["size"] += 3  # ARC task files
        if any("code-golf" in f for f in files):
            scores["size"] += 3
        if any("benchmark" in f and f.endswith(".rs") for f in files):
            scores["perf"] += 3
        if any(f.endswith((".h5", ".pkl", ".pt", ".onnx")) for f in files):
            scores["ml"] += 3

    # 4. Return highest scoring mode
    max_score = max(scores.values())
    if max_score == 0:
        return "ambiguous"

    # Check for close scores (ambiguous)
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    if sorted_scores[0][1] - sorted_scores[1][1] <= 2:
        return "ambiguous"

    return max(scores, key=scores.get)
```

---

## Handling Ambiguous Requests

When the mode is unclear, ask the user:

```python
def handle_ambiguous(request):
    return ask_user_question(
        question="What are we optimizing for?",
        header="Mode",
        options=[
            {"label": "Fastest runtime (speed)", "description": "Optimize for ops/sec, latency, throughput"},
            {"label": "Smallest code (bytes)", "description": "Minimize byte count, code golf"},
            {"label": "Best accuracy (ML)", "description": "Optimize model metrics (coming soon)"}
        ]
    )
```

---

## Delegation to Subskills

Once mode is determined, delegate to the appropriate subskill:

```python
def evolve(request: str):
    # 1. Parse request
    context = gather_context()

    # 2. Detect mode
    mode = detect_mode(request, context)

    if mode == "ambiguous":
        mode = handle_ambiguous(request)

    # 3. Inform user
    print(f"Evolution mode: {mode}")

    # 4. Delegate to subskill
    if mode == "perf":
        return invoke_skill("evolve-perf", request)
    elif mode == "size":
        return invoke_skill("evolve-size", request)
    elif mode == "ml":
        return invoke_skill("evolve-ml", request)
```

---

## Core Features (All Modes)

1. **Population-based**: Maintains diverse solutions, not just the winner
2. **Semantic crossover**: Combines innovations from multiple parents
3. **Adaptive generations**: Continues while improving, stops on plateau
4. **Budget control**: User sets token/generation limits
5. **Checkpointing**: Resume evolution from where you left off
6. **Correctness first**: Invalid solutions get fitness 0

---

## Budget Options

| Budget | Meaning | Approx. Generations |
|--------|---------|---------------------|
| `10k` | 10,000 tokens | ~2-3 generations |
| `50k` | 50,000 tokens | ~10-12 generations |
| `100k` | 100,000 tokens | ~20-25 generations |
| `5gen` | 5 generations | Fixed count |
| `unlimited` | No limit | Until plateau |
| (none) | Default 50k | ~10-12 generations |

---

## Resume Previous Evolution

Run `/evolve --resume` to continue a previous evolution:

1. Finds the most recent `evolution.json`
2. Loads population and champion state
3. Continues from last generation
4. Preserves all history and lineage

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  /evolve <request>                                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Mode Detection                                       │  │
│  │  • Parse keywords                                     │  │
│  │  • Check file context                                 │  │
│  │  • Score each mode                                    │  │
│  │  • Ask user if ambiguous                              │  │
│  └──────────────────┬────────────────────────────────────┘  │
│                     │                                       │
│         ┌───────────┼───────────┬───────────┐               │
│         ▼           ▼           ▼           ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│   │ size     │ │ perf     │ │ ml       │ │ resume   │      │
│   │ subskill │ │ subskill │ │ subskill │ │ (detect) │      │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  Each subskill runs the full evolution loop:                │
│  • Bootstrap → Baseline → Evolution → Finalize              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Subskill Responsibilities

### /evolve-perf (Performance)
- Optimizes runtime speed (ops/sec, latency, throughput)
- Supports any language (Rust, Python, Go, etc.)
- Uses statistical significance testing for timing benchmarks
- Focuses on algorithm families, SIMD, cache optimization
- See: `/evolve-perf` for full documentation

### /evolve-size (Size)
- Optimizes length (bytes, characters, tokens)
- Supports code (Python, Rust, Go) and text (markdown, prompts, configs)
- Uses trick library for systematic transformations
- Focuses on compression, golf tricks, minimal implementations
- See: `/evolve-size` for full documentation

### /evolve-ml (ML - Coming Soon)
- Will optimize model accuracy, loss, F1, etc.
- Will support hyperparameter tuning, architecture search
- See: `/evolve-ml` for planned features

---

## Directory Structure

All evolution modes use a consistent directory structure:

```
.evolve/<problem>/
├── evolution.json       # Full state (mode, population, history)
├── champion.json        # Best solution manifest
├── generations.jsonl    # Per-generation log (append-only)
├── mutations/           # All tested mutations
└── [mode-specific]/     # Mode-specific artifacts
    ├── rust/            # (perf) Rust benchmark code
    ├── solutions/       # (size) Working solutions by size
    └── models/          # (ml) Trained models
```

---

## Examples by Mode

### Performance Mode
```bash
/evolve faster sorting algorithm for integers
# Detected: perf (keyword: "faster")
# Delegates to: /evolve-perf
# Optimizes: ops/sec
# Output: .evolve/sorting/rust/src/evolved.rs
```

### Size Mode
```bash
/evolve shortest Python solution for ARC task 0520fde7
# Detected: size (keywords: "shortest", "ARC")
# Delegates to: /evolve-size
# Optimizes: byte count
# Output: solutions/0520fde7.py (57 bytes)
```

### Explicit Override
```bash
/evolve --mode=size this Rust function
# Mode: size (explicit)
# Delegates to: /evolve-size
# Optimizes: byte count (even though it's Rust)
```

---

## Backward Compatibility

All changes are **additive**:
- Default behavior unchanged (perf mode when ambiguous)
- Existing `/evolve` commands work identically
- Current evolution.json schemas still valid
- No breaking changes for existing users

---

## Quick Reference

| Want to... | Command |
|------------|---------|
| Make code faster | `/evolve faster <algorithm>` |
| Make code shorter | `/evolve shortest <code>` |
| Minimize config file | `/evolve minimal <file type>` |
| Continue previous | `/evolve --resume` |
| Force specific mode | `/evolve --mode=<mode> <problem>` |
