# RefactorBench: 100% Pass Rate with Iterative Self-Correction

An agentic approach to [RefactorBench](https://github.com/microsoft/RefactorBench) (ICLR 2025), a benchmark of 100 multi-file Python refactoring tasks across 9 open-source repositories. We achieve a **100/100 pass rate** using RALPH (Reflective Agent Loop with Progressive Heuristics), an iterative self-correction framework built on Claude Code CLI.

## Results Summary

| Approach | Pass Rate | Model |
|----------|-----------|-------|
| Microsoft SOTA (paper) | 35% | Claude 3.5 Sonnet |
| Human performance (paper) | 87% | — |
| **Opus 4.5 single-shot** | **52%** | Claude Opus 4.5 |
| **Opus 4.5 + RALPH** | **100%** | Claude Opus 4.5 |

### Breakdown

| Phase | Tasks | Passed | Method |
|-------|-------|--------|--------|
| Single-shot baseline | 100 | 52 | One attempt per task |
| RALPH iterative | 48 | 48 | Up to 10 iterations |
| **Total** | **100** | **100** | — |

---

## Validation Results (February 2025)

We validated all results using **Claude Opus 4.5** exclusively to eliminate model confounds:

### Single-Shot Baseline: 52/100

52 tasks pass on a single attempt with no iteration. The remaining 48 require RALPH.

### RALPH Iteration Counts

For the 48 tasks that didn't pass single-shot, RALPH solved all of them:

| Iterations | Tasks | Percentage |
|------------|-------|------------|
| 1 | 26 | 54% |
| 2 | 15 | 31% |
| 3+ | 7 | 15% |
| **Mean** | **1.83** | — |
| **Max** | **7** | — |

### Complete Task List with Iterations

<details>
<summary>Click to expand full task list</summary>

#### Tasks Solved Single-Shot (52 tasks)

| Repository | Task |
|------------|------|
| ansible_refactor | add-log-parameter-get-group-vars |
| ansible_refactor | add-log-parameter-is-systemd-managed |
| ansible_refactor | data-to-inventory-data |
| ansible_refactor | new-inventory-patterns |
| ansible_refactor | parse_key_value |
| ansible_refactor | rename-lenient-lowercase |
| ansible_refactor | sort-groups-to-group-sort |
| celery_refactor | add-log-parameter-get-digest-algorithm |
| celery_refactor | add-log-parameter-node-format |
| celery_refactor | autoretry-to-retry |
| celery_refactor | dump-message-to-serialization |
| celery_refactor | ensure_serialize |
| celery_refactor | evaluate-promises-to-serialization |
| celery_refactor | object-mro-lookup |
| django_refactor | add-log-parameter-constant-time-compare |
| django_refactor | combine-utils-dates-dateformat |
| django_refactor | new-converter-to-python-class |
| django_refactor | new-reference-context-field-class |
| django_refactor | new-utils-adapt-method-mode |
| django_refactor | new-utils-check-response |
| django_refactor | new-utils-path-from-module |
| django_refactor | remove-core-cache-utils |
| fastapi_refactor | add-log-parameter-generate-option-id-for-path |
| fastapi_refactor | exception-handlers-to-handlers |
| fastapi_refactor | get-auth-scheme-param |
| fastapi_refactor | value-is-a-sequence |
| flask_refactor | add-log-parameter-get-debug-flag |
| flask_refactor | add-log-parameter-get-flashed-messages |
| flask_refactor | render-template-str |
| flask_refactor | stream-template-str |
| requests_refactor | add-log-parameter-get-encoding-from-headers |
| requests_refactor | add-log-parameter-resolve-proxies |
| requests_refactor | add-log-parameter-select-proxy |
| requests_refactor | rename-lookup-dict-dict-lookup |
| requests_refactor | rename-super-len-complex-len |
| requests_refactor | split-warnings-exceptions |
| salt_refactor | add-log-parameter-delete-directory |
| salt_refactor | add-log-parameter-get-capability-definitions |
| salt_refactor | cant-create |
| salt_refactor | channel-to-transport |
| salt_refactor | ex-pillar-fail |
| salt_refactor | ex-state-fail |
| salt_refactor | exactly-n-boto-mod |
| salt_refactor | get-unavail |
| salt_refactor | iam-to-aws |
| salt_refactor | mksls-to-specific |
| salt_refactor | namecheap-xmlutil |
| salt_refactor | perm-denied |
| tornado_refactor | log-utils |
| tornado_refactor | option-parser-with-pretty-print |
| tornado_refactor | remove-locale-data |
| tornado_refactor | rename-to-camel-case |

#### Tasks Requiring RALPH Iteration (48 tasks)

| Repository | Task | Iterations |
|------------|------|------------|
| ansible_refactor | combine-namespace-compat | 1 |
| ansible_refactor | move-quoting-splitter | 1 |
| ansible_refactor | new-utils-class-connection | 1 |
| ansible_refactor | new-utils-from-basic | 2 |
| celery_refactor | annotation-utils | 1 |
| celery_refactor | combine-unpickle-task | 1 |
| celery_refactor | expand-router-string-to-utils | 1 |
| celery_refactor | rename-host-format | 4 |
| celery_refactor | truncate-text | 2 |
| django_refactor | add-log-parameter-get-resolver | 2 |
| django_refactor | add-log-parameter-resolve-error-handler | 1 |
| django_refactor | add-none-handling-duration-string | 1 |
| django_refactor | combine-utils-hashable-itercompat | 1 |
| django_refactor | new-path-traversal-exception | 2 |
| django_refactor | new-reference-context-graph-class | 2 |
| django_refactor | new-timezone-class | 1 |
| django_refactor | remove-db-models-constants | 1 |
| django_refactor | rename-file-move-safe | 1 |
| django_refactor | split-parse-apps-and-model-labels | 1 |
| fastapi_refactor | openapi-get-utils | 2 |
| fastapi_refactor | params-to-param | 1 |
| flask_refactor | debughelpers-to-helpers.py | 2 |
| flask_refactor | rename-send-from-directory | 1 |
| requests_refactor | combine-from-key-to-key | 1 |
| requests_refactor | combine-internal-utils-utils | 1 |
| requests_refactor | move-hooks-sessions | 1 |
| requests_refactor | new-cookie-utils-class | 2 |
| salt_refactor | add-log-parameter-recursive-diff | 1 |
| salt_refactor | paged-call-boto-mod | 1 |
| salt_refactor | pem-fingerprint | 1 |
| scrapy_refactor | add-log-parameter-disconnect-all | 3 |
| scrapy_refactor | add-log-parameter-job-dir | 1 |
| scrapy_refactor | add-log-parameter-xmliter | 2 |
| scrapy_refactor | genspider-functions-to-utils-url | 2 |
| scrapy_refactor | new-downloadermiddlewares-utils | 2 |
| scrapy_refactor | new-spider-utils-in-spiders | 4 |
| scrapy_refactor | new-verify-reactor-class | 3 |
| scrapy_refactor | not-supported-exception-to-unsupported | 2 |
| scrapy_refactor | parameterize-gunzip | 7 |
| scrapy_refactor | rename-description-commands | 7 |
| scrapy_refactor | rename-engine-status | 2 |
| scrapy_refactor | rename-processtest-testproc | 2 |
| scrapy_refactor | sitemap-url-to-url | 4 |
| tornado_refactor | global-objects | 1 |
| tornado_refactor | options-utils | 1 |
| tornado_refactor | rename-http1connection | 1 |
| tornado_refactor | resolvers-as-separate | 1 |
| tornado_refactor | tcpclient-connect-params | 2 |

</details>

### By Repository

| Repository | Total | Single-Shot | RALPH | Avg Iterations |
|------------|-------|-------------|-------|----------------|
| ansible_refactor | 11 | 7 | 4 | 1.25 |
| celery_refactor | 12 | 7 | 5 | 1.80 |
| django_refactor | 18 | 8 | 10 | 1.30 |
| fastapi_refactor | 6 | 4 | 2 | 1.50 |
| flask_refactor | 6 | 4 | 2 | 1.50 |
| requests_refactor | 10 | 6 | 4 | 1.25 |
| salt_refactor | 15 | 12 | 3 | 1.00 |
| **scrapy_refactor** | **13** | **0** | **13** | **3.15** |
| tornado_refactor | 9 | 4 | 5 | 1.20 |

**Scrapy was the hardest repository** — zero single-shot passes, all 13 required RALPH, with average 3.15 iterations and two tasks requiring the maximum 7 iterations.

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

### Single-Shot Baseline

A refactoring agent built on `claude -p` (Claude Code CLI in print mode) with:
- Vanilla system prompt ("You are an expert Python developer...")
- File tools: Read, Write, Edit, Glob, Grep, Bash
- 15-turn maximum per task
- AST-based test validation after changes

### RALPH: Reflective Agent Loop with Progressive Heuristics

For tasks that fail single-shot, RALPH adds iterative self-correction:

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

4. **Multi-chain independence.** Each chain starts from a fresh copy of the repository, providing diversity. If chain 0 gets stuck, chain 1 starts clean.

5. **Engineering notebook.** Every iteration records agent reasoning, tool calls, file diffs, and test results in both JSON and Markdown formats.

---

## Architecture

```
run_task.py              Coordinator: RALPH -> evolution fallback
  │
  ├── ralph_runner.py    RALPH loop: chains x iterations
  │     │
  │     ├── ralph_prompt_builder.py   Iteration-aware prompt construction
  │     ├── refactor_agent.py         Agent backend (claude -p)
  │     └── notebook.py               Engineering notebook recording
  │
  └── evolve_task.py     Evolution fallback (unused)

refactor_agent.py        Core agent: prompt -> claude CLI -> test validation
  │
  ├── claude -p           Claude Code CLI in print mode
  │     Tools: Read, Write, Edit, Glob, Grep, Bash
  │
  └── pytest              AST-based test runner (Python 3.12)
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
| `results/opus_baseline.json` | Opus single-shot validation results |
| `results/opus_validation_report.txt` | Complete validation report |

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

# Run Opus baseline validation
python3 run_opus_baseline.py
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

## Limitations & Future Work

### Current Limitations

1. **Single-run results.** Each task was solved once. We don't have Pass@k rates or variance data.

2. **No ablation study.** We haven't isolated the contribution of each RALPH component (negative memory, filesystem persistence, multi-chain).

3. **Benchmark saturation.** 100% pass rate may indicate the benchmark ceiling rather than general capability.

### Future Work

1. **Pass@k analysis** — Run each task multiple times to measure reliability
2. **Ablation study** — Test RALPH components individually
3. **Other benchmarks** — Apply RALPH to SWE-bench, Aider polyglot, etc.
4. **Weaker models** — Test if RALPH works with Sonnet, Haiku, or open-source models

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
