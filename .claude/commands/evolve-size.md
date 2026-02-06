---
description: Size subskill for /evolve - optimizes for minimal code/text length
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite
argument-hint: <problem description>
---

# /evolve-size - Size Optimization Subskill

This is the **size optimization subskill** for the `/evolve` command. It evolves solutions for **minimal length** (bytes, characters, tokens).

**Implementation**: Uses the evolve-sdk Python package for hierarchical agent orchestration.

---

## Usage

```bash
/evolve-size <problem description>
/evolve-size <problem description> --budget <tokens|generations>
/evolve-size --resume
```

**Examples**:
```
/evolve-size shortest Python solution for ARC task 0520fde7
/evolve-size minimize bytes for this function
/evolve-size shortest markdown rule for git commits
/evolve-size most concise regex for email validation
/evolve-size --resume
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
python3 -m evolve_sdk --config=evolve_config.json --mode=size

# Without config file:
python3 -m evolve_sdk "<problem description>" --mode=size

# Resume previous:
python3 -m evolve_sdk --resume --mode=size
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
2. Summarize: generations, final byte count, champion path
3. Show tricks applied and byte reduction trajectory

---

## Mode-Specific Guidance

The following sections define **size mode requirements** that the SDK's subagents follow.

---

## Supported Domains

| Domain | Measure | Correctness | Examples |
|--------|---------|-------------|----------|
| **Python** | bytes | Execute & test | ARC-AGI, code golf |
| **Rust** | bytes | Compile & test | Minimal implementations |
| **Go** | bytes | Compile & test | Minimal implementations |
| **Text/Markdown** | bytes/chars | LLM judges | Rule files, prompts, docs |
| **Regex** | chars | Test against examples | Pattern matching |
| **Config** | bytes | Functional test | .cursorrules, CLAUDE.md |

---

## Fitness Function

For all size optimization, smaller is better (while maintaining correctness):

```python
def size_fitness(candidate, domain, correctness_fn):
    """Universal size optimization fitness."""
    # Measure size (domain-specific)
    size = len(candidate.encode('utf-8'))  # bytes

    # Check correctness (domain-specific)
    is_correct, quality = correctness_fn(candidate)

    # Combine: must be correct, then minimize size
    if not is_correct:
        return 0.001  # Penalty for incorrect

    # Higher fitness for smaller correct solutions
    return quality * (10000 / (size + 1))
```

### Scoring Formula (Code Golf)

```python
score = 2500 - byte_count  # if correct
score = 0.001              # if incorrect
```

---

## Evaluation Contract

1. **Correctness First**: A solution that's wrong is worthless regardless of size
2. **Train/Valid Split**: Test on held-out examples to prevent overfitting
3. **Deterministic Evaluation**: Same solution = same byte count (no randomness)
4. **Byte Accuracy**: Use `len(code.encode('utf-8'))` for consistent measurement

---

## Python Golf Tricks

```python
PYTHON_GOLF_TRICKS = [
    # Whitespace removal
    {"name": "remove_space_after_colon", "from": ": ", "to": ":"},
    {"name": "remove_space_in_tuple", "from": ", ", "to": ","},

    # Lambda conversion (saves ~4 bytes typically)
    {"name": "def_to_lambda",
     "pattern": r"def (\w+)\(([^)]*)\):\s*return (.+)",
     "replacement": r"\1=lambda \2:\3"},

    # Operator shortcuts (for 0/1 values)
    {"name": "and_to_mult", "from": " and ", "to": "*"},
    {"name": "or_to_bitor", "from": " or ", "to": "|"},
    {"name": "eq_zero_to_lt1", "from": "==0", "to": "<1"},
    {"name": "ne_zero_to_gt0", "from": "!=0", "to": ">0"},

    # Range shortcuts
    {"name": "range_to_tuple", "from": "range(3)", "to": "(0,1,2)"},

    # List tricks
    {"name": "list_map_to_star", "from": "list(map(", "to": "[*map("},
]
```

---

## Mutation Strategies (Size Mode)

For size optimization, prioritize these mutation strategies:

1. **Whitespace removal**: Remove unnecessary spaces, newlines
2. **Lambda conversion**: Convert `def` to `lambda` where possible
3. **Variable shortening**: Use single-letter names
4. **Operator substitution**: Use shorter equivalents
5. **Comprehension conversion**: Lists/dicts to comprehensions
6. **Algorithm change**: Try fundamentally shorter approaches
7. **Import elimination**: Remove unused imports, inline small functions
8. **Built-in exploitation**: Use shorter built-in alternatives

---

## Crossover Guidelines

When combining two parent solutions:

1. **Take shorter structure**: Use the more compact algorithm
2. **Apply donor tricks**: Apply one parent's tricks to the other
3. **Combine best parts**: Cherry-pick shortest implementations of each part
4. **LLM hybrid**: Ask LLM to intelligently combine approaches

---

## Logging & Artifacts

All evolution artifacts are stored in `.evolve-sdk/<problem>/`:

```
.evolve-sdk/<problem>/
├── evolution.json       # Full state (mode, population, history)
├── champion.json        # Best solution manifest
├── generations.jsonl    # Per-generation log (append-only)
├── mutations/           # All tested mutations
└── solutions/           # Working solutions by size
```

---

## Key Principles

1. **Byte count is king**: For correct solutions, smaller always wins
2. **Correctness is binary**: Either it works or it doesn't
3. **Tricks compound**: Multiple small savings add up
4. **Algorithm matters**: Sometimes a different approach is fundamentally shorter
5. **Cross-task learning**: Tricks discovered on one task often apply to others
6. **Domain-aware**: Different domains need different optimization strategies
