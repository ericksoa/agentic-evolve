# Agentic Evolve - Project Instructions

## CRITICAL: Resource Management

**NEVER start more than 4-6 parallel processes on this computer.**

This is especially important for evolution runs which can spawn many agents/processes.

### For Evolution Workloads
1. **Local execution**: Max 4 parallel mutations at a time
2. **Cloud execution**: Use lightning.ai or similar for heavy parallelism
3. **Best-of-N runs**: Run sequentially, not all at once

### Cloud Providers for Heavy Compute
- **lightning.ai**: Preferred for GPU/CPU serverless compute
- Configure evolve-sdk to submit jobs to cloud instead of running locally

### Safe Evolution Commands
```bash
# SAFE: Sequential with limited parallelism
python3 -m evolve_sdk "problem" --no-parallel
python3 -m evolve_sdk "problem" --max-workers=4

# DANGEROUS: Don't do this locally
# python3 -m evolve_sdk "problem" --population-size=100
```

## Directory Structure

**CRITICAL: Before creating ANY new file or directory, check where similar things already exist.**

```
agentic-evolve/
├── sdk/                    # The evolve-sdk Python package ONLY
│   ├── evolve_sdk/         # Source code
│   ├── tests/              # SDK unit tests
│   └── README.md           # SDK documentation
│
├── showcase/               # Verified showcase projects (polished results + docs)
│   ├── regex_golf/
│   ├── nqueens-evolution/
│   └── ...
│
├── experiments/            # WIP/exploratory projects (not yet verified)
│   ├── phase1_demo/
│   ├── sorting-network-evolution/
│   └── ...
│
├── plugin-package/         # Claude Code plugin packaging
└── CLAUDE.md               # This file
```

### Rules
1. **Always check existing patterns first** - run `ls` to see where similar files/directories live before creating new ones
2. **Showcases**: `showcase/` for verified projects with polished results and documentation
3. **Experiments**: `experiments/` for WIP, exploratory, or unverified projects
4. **SDK code**: `sdk/evolve_sdk/` only - nothing else in sdk/ except tests and docs
4. **Tests**: SDK tests in `sdk/tests/`, showcase-specific tests stay with their showcase
5. **When in doubt**: Look at 2-3 existing examples before deciding where to put something new

### Diagram Standards
- **Always use SVG** for architecture diagrams, flowcharts, and visual documentation
- **Never use ASCII art** for diagrams - SVG is more professional, readable, and version-controllable
- Place diagram SVGs near the content they document (e.g., `showcase/nqueens-evolution/evolution-factory.svg`)
- Reference SVGs in markdown with `![Alt text](path/to/diagram.svg)`

## Showcase Projects (verified results)
- `regex_golf/` - Phase 1 agents demonstration (Debugger + Plateau Breaker)
- `linkage-evolution/` - 25% improvement, 3D-printable output
- `cuopt_lp_autotuner/` - 1.07x speedup, 73% problems improved
- `santa-2025-packing/` - Kaggle Christmas tree packing challenge
- `code-golf/` - 72 ARC tasks solved, 163K points
- `nqueens-evolution/` - 14,000x speedup, memory system demo
- `global-chess-challenge-2025/` - 77.4 ACPL, AIcrowd competition
- `airfoil-evolution/` - 44% L/D improvement, 3D-printable
- `molecular-admet-prediction/` - 0.890 ROC-AUC, TDC benchmark
- `openml-automl-benchmark/` - OpenML-CC18 protocol, 2.38% avg improvement
- `refactorbench-evolution/` - 100/100 RefactorBench (vs 35% SOTA), preliminary

## Experiment Projects (WIP/exploratory)
Located in `experiments/` — projects that are still being tuned, have limited docs, or unverified results.
- `algotune-speedup/` - AlgoTune speedup attempt; custom validation claimed 2.62x but official eval shows ~1.01x (methodology was flawed)
