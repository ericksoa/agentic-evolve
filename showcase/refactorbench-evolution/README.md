# RefactorBench Evolution

Evolving refactoring agent strategies to beat Microsoft's RefactorBench SOTA using agentic evolution.

## What is RefactorBench?

[RefactorBench](https://github.com/microsoft/RefactorBench) (ICLR 2025) is a benchmark of 100 multi-file Python refactoring tasks across 9 popular open-source projects (Django, Flask, Celery, etc.). Tasks include function renaming, module extraction, code movement, and structural changes — each validated by AST-based unit tests.

**Current SOTA:** 35% (Claude 3.5 Sonnet with descriptive prompts) | Humans: 87%

## What We Evolve

A **refactoring agent strategy** — the system prompt, step-by-step instructions, file discovery approach, state management, incremental validation, and error recovery heuristics that guide Opus 4.5 through multi-file refactoring tasks.

The paper identifies three key failure modes:
1. **File location** — missing files that need updating
2. **Intermediate broken state** — changes that break things midway
3. **Context flooding** — losing track in large codebases

Evolution discovers which combination of instructions and recovery strategies best mitigates these failures.

## Quick Start

```bash
# Prerequisites
pip install anthropic pytest

# Setup (already done if cloned)
cd showcase/refactorbench-evolution
git clone https://github.com/microsoft/RefactorBench .refactorbench

# List all tasks
python3 run_single_task.py --list

# Run a single task for debugging
python3 run_single_task.py --repo flask_refactor --task rename-send-from-directory -v

# Run baseline on 10 tasks
python3 evaluate.py baseline_strategy.json --subset 10

# Run baseline with JSON output (for evolve-sdk)
python3 evaluate.py baseline_strategy.json --json --subset 10

# Analyze failures
python3 evaluate.py baseline_strategy.json --json > results.json
python3 analyze_failures.py results.json
```

## Evolution

```bash
# Run evolution with evolve-sdk (from repo root)
cd /path/to/agentic-evolve
python3 -m evolve_sdk showcase/refactorbench-evolution/evolve_config.json
```

## Architecture

```
evaluate.py          — Fitness evaluator: runs strategy on task subset, returns pass rate
refactor_agent.py    — Refactoring agent: multi-turn Anthropic API with file tools
baseline_strategy.json — Vanilla starting strategy (no special instructions)
evolve_config.json   — Evolution configuration for evolve-sdk
run_single_task.py   — Debug tool for running individual tasks
analyze_failures.py  — Failure pattern analysis
```

## Strategy Format

Strategies are JSON files controlling agent behavior:

```json
{
  "name": "strategy_name",
  "system_prompt": "Expert system prompt...",
  "approach": {
    "steps": ["Step 1: ...", "Step 2: ..."]
  },
  "file_discovery": {"enabled": true, "strategy": "..."},
  "state_tracking": {"enabled": true, "instructions": "..."},
  "incremental_validation": {"enabled": true, "instructions": "..."},
  "error_recovery": {"enabled": true, "max_retries": 3, "instructions": "..."},
  "context_management": {"max_files_in_context": 15, "strategy": "..."}
}
```

## Results

| Strategy | Pass Rate | vs SOTA |
|----------|-----------|---------|
| SOTA (Claude 3.5 Sonnet descriptive) | 35% | baseline |
| Opus 4.5 baseline | TBD | TBD |
| Evolved strategy | TBD | TBD |

## Benchmark Details

**9 repositories, 100 tasks:**
- ansible (11) | celery (12) | django (18) | fastapi (6)
- flask (6) | requests (10) | salt (15) | scrapy (13) | tornado (9)

**Task types:** function rename, module move/extract, combine modules, add parameters, create new classes/utilities

**Validation:** AST-based unit tests checking for specific code patterns (import names, function definitions, class structures)
