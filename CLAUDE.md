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

**CRITICAL: Check existing patterns before creating new directories.**

```
agentic-evolve/
├── sdk/                    # The evolve-sdk Python package
│   ├── evolve_sdk/         # Source code
│   ├── tests/              # SDK unit tests
│   └── README.md           # SDK documentation
│
├── showcase/               # ALL showcase projects go here (NOT in sdk/)
│   ├── regex_golf/         # Example showcase
│   ├── nqueens-evolution/
│   └── ...
│
├── plugin-package/         # Claude Code plugin packaging
└── CLAUDE.md               # This file
```

### Rules
1. **Showcases**: Always in top-level `showcase/`, never in `sdk/showcase/`
2. **SDK code**: Only in `sdk/evolve_sdk/`
3. **Tests**: SDK tests in `sdk/tests/`, showcase tests stay with their showcase

## Showcase Projects
- `regex_golf/` - Phase 1 agents demonstration (Debugger + Plateau Breaker)
- `santa-2025-packing/` - Kaggle Christmas tree packing challenge
- `mallorn-astro-classification/` - Astronomy classification
- `kernelbench-triton-evolution/` - GPU kernel optimization
- `code-golf/` - Code size optimization
