# RefactorBench Evolution - Project Instructions

## CRITICAL: Output Rules
- **NEVER include cost estimates, dollar amounts, timing data, or token counts in summaries.** No `$0.88`, no `127 seconds`, no `16 turns`. Just report pass/fail, score, and iterations.
- **NEVER use `model: "sonnet"` or `model: "haiku"` for Task subagents.** Always use Opus or omit model parameter.

## Overview
Evolve refactoring agent strategies to beat Microsoft's RefactorBench SOTA (35%).

## Key Paths
- `.refactorbench/` - Cloned RefactorBench repo (gitignored)
- `strategies/` - Evolved strategy archive
- `workdirs/` - Temporary working directories for task execution (gitignored)

## Running
```bash
# Single task (debugging)
python3 run_single_task.py --repo flask_refactor --task rename-send-from-directory

# Evaluate a strategy on subset
python3 evaluate.py baseline_strategy.json --json --subset 10

# Evaluate on full benchmark
python3 evaluate.py baseline_strategy.json --json

# Analyze failures
python3 analyze_failures.py results/latest.json
```

## Architecture
- Strategy JSON controls the agent's system prompt, step-by-step approach, and recovery heuristics
- `refactor_agent.py` calls Anthropic API with multi-turn tool use (file read/write/test)
- `evaluate.py` orchestrates running tasks and collecting results
- Tests are AST-based checks from RefactorBench (not runtime tests)

## Resource Limits
- Max 4 parallel task evaluations locally
- Each task gets max 15 API turns
- Use `--subset N` during evolution to limit task count
