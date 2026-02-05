# RefactorBench: 100% Pass Rate with Iterative Self-Correction

**Preliminary results — verification pending.**

An agentic approach to [RefactorBench](https://github.com/microsoft/RefactorBench) (ICLR 2025), a benchmark of 100 multi-file Python refactoring tasks across 9 open-source repositories. We achieve a **100/100 pass rate** using RALPH (Reflective Agent Loop with Progressive Heuristics), an iterative self-correction framework built on top of Claude Code CLI.

| | Pass Rate | Model |
|--|-----------|-------|
| Microsoft SOTA (paper) | 35/100 (35%) | Claude 3.5 Sonnet |
| Our baseline | 53/100 (53%) | Claude 3.5 Sonnet |
| Our baseline + RALPH | **100/100 (100%)** | Claude Opus 4.5 |
| Human performance (paper) | 87/100 (87%) | — |

> **Caveat:** The baseline was run with Sonnet; RALPH used Opus. This confounds model capability with approach. An Opus-only baseline run is needed for fair attribution. See [Threats to Validity](#threats-to-validity).

---

## What is RefactorBench?

RefactorBench is a benchmark of 100 multi-file Python refactoring tasks spanning 9 popular open-source projects. Each task requires structural code changes — renaming functions, moving code between modules, extracting classes, combining files — validated by AST-based unit tests that check the resulting code structure (not runtime behavior).

The benchmark is designed to be hard for LLMs because:
1. **Multi-file scope** — changes must be coordinated across multiple files
2. **Reference tracking** — every import, call site, and re-export must be updated
3. **Context management** — large codebases overwhelm context windows

The published SOTA is 35% using Claude 3.5 Sonnet with descriptive prompts.

---

## Approach

### Phase 1: Single-Shot Baseline (53%)

A refactoring agent built on `claude -p` (Claude Code CLI in print mode) with:
- Vanilla system prompt ("You are an expert Python developer...")
- File tools: Read, Write, Edit, Glob, Grep, Bash
- 15-turn maximum per task
- AST-based test validation after changes

This simple single-shot approach solved 53/100 tasks with Sonnet — already 1.5x the published SOTA.

### Phase 2: RALPH Iterative Loop (53% -> 100%)

For the 47 tasks that failed single-shot, RALPH adds iterative self-correction:

```
for chain in range(5):          # 5 independent attempts
    fresh copy of repository
    for iteration in range(10): # up to 10 correction cycles
        build prompt with:
            - task description
            - previous test results (failing tests, error messages)
            - list of failed approaches ("DO NOT REPEAT")
        run agent (claude -p, 15 turns)
        run AST tests
        if all tests pass: done
        save progress to filesystem
```

Key design decisions:

1. **Filesystem persistence.** The working directory retains all file changes between iterations. Each new agent reads the current state of the code, not the original.

2. **Progressive feedback.** Each iteration's prompt includes the previous test output, the list of currently failing tests, and a record of all prior approaches that failed.

3. **Negative memory.** Failed approaches are explicitly listed with "DO NOT REPEAT" instructions, preventing the agent from cycling through the same mistakes.

4. **Multi-chain independence.** Each chain starts from a fresh copy of the repository, providing diversity. If chain 0 gets stuck in a dead end, chain 1 starts clean.

5. **Engineering notebook.** Every iteration records agent reasoning, tool calls, file diffs, and test results in both JSON and Markdown formats for post-hoc analysis.

### Phase 3: Strategy Evolution (not needed)

A planned evolution phase using `evolve-sdk` to mutate the agent's strategy (system prompt, instructions, recovery heuristics) was prepared but never triggered — RALPH solved all 100 tasks without it.

---

## Results

### Overall

| Metric | Value |
|--------|-------|
| Total tasks | 100 |
| Solved by baseline (single-shot) | 53 |
| Solved by RALPH (iterative) | 47 |
| Solved by evolution | 0 |
| **Final pass rate** | **100/100** |

### By Repository

| Repository | Tasks | Baseline | RALPH | Total |
|------------|-------|----------|-------|-------|
| ansible_refactor | 11 | 7 | 4 | 11/11 |
| celery_refactor | 12 | 7 | 5 | 12/12 |
| django_refactor | 18 | 8 | 10 | 18/18 |
| fastapi_refactor | 6 | 4 | 2 | 6/6 |
| flask_refactor | 6 | 4 | 2 | 6/6 |
| requests_refactor | 10 | 6 | 4 | 10/10 |
| salt_refactor | 15 | 12 | 3 | 15/15 |
| scrapy_refactor | 13 | 0 | 13 | 13/13 |
| tornado_refactor | 9 | 5 | 4 | 9/9 |

Scrapy was the hardest repository — zero baseline passes, all 13 tasks required RALPH. Every one was eventually solved.

### RALPH Convergence

All 47 RALPH tasks were solved on chain 0 (the first chain attempted), meaning the multi-chain redundancy was never needed. Most tasks converged within 1-3 iterations.

---

## Architecture

```
run_task.py              Coordinator: RALPH -> evolution fallback
  |
  +-- ralph_runner.py    RALPH loop: chains x iterations
  |     |
  |     +-- ralph_prompt_builder.py   Iteration-aware prompt construction
  |     +-- refactor_agent.py         Agent backend (claude -p)
  |     +-- notebook.py               Engineering notebook recording
  |
  +-- evolve_task.py     Evolution fallback (unused)

refactor_agent.py        Core agent: prompt -> claude CLI -> test validation
  |
  +-- claude -p           Claude Code CLI in print mode
  |     Tools: Read, Write, Edit, Glob, Grep, Bash
  |
  +-- pytest              AST-based test runner (Python 3.12)
```

### Key Files

| File | Purpose |
|------|---------|
| `run_task.py` | Orchestrates RALPH then evolution for a single task |
| `ralph_runner.py` | RALPH loop implementation (5 chains x 10 iterations) |
| `ralph_prompt_builder.py` | Builds iteration-aware prompts with test feedback |
| `refactor_agent.py` | Agent wrapper around `claude -p` with test runner |
| `notebook.py` | Engineering notebook: diffs, reasoning, tool traces |
| `baseline_strategy.json` | Vanilla strategy for single-shot baseline |
| `results/progress.json` | Source of truth for all task outcomes |

---

## Threats to Validity

This is a preliminary study with several important caveats:

### 1. Model Confound

The baseline used Claude 3.5 Sonnet; RALPH used Claude Opus 4.5. The improvement from 53% to 100% reflects both the iterative approach and the stronger model. To properly attribute gains:
- **Needed:** Re-run baseline with Opus to isolate approach contribution
- **Needed:** Re-run RALPH with Sonnet to isolate model contribution
- **Hypothesis:** Opus baseline alone likely scores 70-85%, meaning RALPH adds 15-30 points

### 2. No Independent Verification

All 100 results come from a single run. Test validation uses the same AST-based tests from RefactorBench, but:
- Results have not been independently verified
- No check for test-gaming (agent modifying test files)
- Need to re-run a random subset and confirm reproducibility

### 3. Single-Run Statistics

Each task was solved at most once. We have no data on:
- Pass@k rates (how reliably does RALPH solve each task?)
- Variance across runs
- Whether chain 0 always succeeds or if multi-chain is needed on re-runs

### 4. Benchmark Saturation

A 100% pass rate suggests the benchmark may not differentiate between strong approaches. This result says more about the ceiling of RefactorBench than about the general capability of the approach.

### 5. Comparison Fairness

The Microsoft SOTA (35%) used a different tooling setup. Our agent has access to `claude -p` which provides integrated file editing, search, and shell access — a richer tool environment than the paper's setup.

---

## Reproducing Results

### Prerequisites

- Python 3.12 (required for AST tests — 3.13+ has breaking AST changes)
- Claude Code CLI (`claude` command)
- RefactorBench repository

### Setup

```bash
cd showcase/refactorbench-evolution

# Clone RefactorBench
git clone https://github.com/microsoft/RefactorBench .refactorbench

# Create Python 3.12 venv for test runner
python3.12 -m venv .venv-3.12
.venv-3.12/bin/pip install pytest

# Verify setup
.venv-3.12/bin/python3 --version  # Must be 3.12.x
```

### Run a Single Task

```bash
# Single task with RALPH
python3 run_task.py --repo flask_refactor --task rename-send-from-directory --verbose

# Single task baseline only (no RALPH)
python3 refactor_agent.py baseline_strategy.json --repo flask_refactor --task rename-send-from-directory -v
```

### Check Progress

```bash
python3 -c "
import json
p = json.load(open('results/progress.json'))
print(f'{p[\"current_passed\"]}/{p[\"current_total\"]} ({p[\"current_passed\"]/p[\"current_total\"]*100:.0f}%)')
"
```

---

## Next Steps

1. **Opus baseline run** — Re-run the 53-task baseline with Opus to isolate model vs. approach gains
2. **Verification run** — Re-run all 100 tasks from scratch and compare results
3. **Pass@k analysis** — Run each RALPH task 3-5 times to measure reliability
4. **Sonnet RALPH run** — Run RALPH with Sonnet to test if the approach works with weaker models
5. **Ablation study** — Test RALPH components individually (negative memory, multi-chain, notebook)

---

## Citation

RefactorBench:
```
@inproceedings{ouyang2025refactorbench,
  title={RefactorBench: Evaluating Agents on Multi-File Repository-Level Code Refactoring},
  author={Ouyang, Zhe and Muennighoff, Niklas and Phung, Dung and Jain, Naman and Sun, Yuntong and Tran, Huy and Ding, Yangruibo and Wang, Xingyao and Peng, Baolin and Chen, Bei and Zhang, Lu},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
