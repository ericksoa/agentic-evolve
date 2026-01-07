---
description: Evolve algorithms using Agent SDK with hierarchical agents (experimental)
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
argument-hint: <problem description> [--mode=size|perf|ml]
---

# /evolve-sdk - Agent SDK Evolution (Experimental)

Evolve algorithms using the Claude Agent SDK for fine-grained control over context and parallelism.

This is an **experimental alternative** to `/evolve` that provides:
- Hierarchical subagents with clean context per generation
- Parallel mutation exploration
- Validation hooks to block unsafe code
- Full audit logging

---

## Prerequisites

Before using this skill, ensure the SDK is installed:

```bash
pip install -e sdk/  # From the agentic-evolve root
```

---

## Usage

```bash
/evolve-sdk <problem description>
/evolve-sdk <problem description> --mode=<size|perf|ml>
/evolve-sdk --resume
```

---

## Instructions

When invoked, run the evolution via the Python SDK CLI:

### Step 1: Check Installation

First verify the SDK is installed:
```bash
python3 -c "import evolve_sdk" 2>/dev/null && echo "SDK installed" || echo "SDK not installed"
```

If not installed, inform the user:
```
The evolve-sdk package is not installed. Install it with:
  cd <repo-root>/sdk && pip install -e .
```

### Step 2: Parse Arguments

Extract from the user's request:
- `problem`: The problem description (everything except flags)
- `mode`: From `--mode=X` flag, default to "size"
- `resume`: True if `--resume` is present
- Any additional flags to pass through

### Step 3: Run Evolution

Execute the SDK CLI:

```bash
# Normal run
python3 -m evolve_sdk "<problem>" --mode=<mode>

# Resume
python3 -m evolve_sdk --resume
```

Additional useful flags:
- `--max-generations=N`: Limit generations (default: 50)
- `--population-size=N`: Population size (default: 10)
- `--plateau=N`: Stop after N gens without improvement (default: 5)
- `--no-parallel`: Run mutations sequentially

### Step 4: Report Results

After evolution completes:
1. Read `.evolve-sdk/<problem>/champion.json` for the best solution
2. Summarize the evolution:
   - Generations run
   - Final fitness
   - Champion file path
3. Show a snippet of the winning solution

---

## Example Session

```
User: /evolve-sdk shortest Python function to check if a number is prime