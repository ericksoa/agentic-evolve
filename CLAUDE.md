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
├── showcase/               # ALL showcase/demo projects
│   ├── regex_golf/
│   ├── nqueens-evolution/
│   └── ...
│
├── plugin-package/         # Claude Code plugin packaging
└── CLAUDE.md               # This file
```

### Rules
1. **Always check existing patterns first** - run `ls` to see where similar files/directories live before creating new ones
2. **Showcases**: Top-level `showcase/` directory
3. **SDK code**: `sdk/evolve_sdk/` only - nothing else in sdk/ except tests and docs
4. **Tests**: SDK tests in `sdk/tests/`, showcase-specific tests stay with their showcase
5. **When in doubt**: Look at 2-3 existing examples before deciding where to put something new

### Diagram Standards
- **Always use SVG** for architecture diagrams, flowcharts, and visual documentation
- **Never use ASCII art** for diagrams - SVG is more professional, readable, and version-controllable
- Place diagram SVGs near the content they document (e.g., `showcase/nqueens-evolution/evolution-factory.svg`)
- Reference SVGs in markdown with `![Alt text](path/to/diagram.svg)`

## Showcase Projects
- `regex_golf/` - Phase 1 agents demonstration (Debugger + Plateau Breaker)
- `santa-2025-packing/` - Kaggle Christmas tree packing challenge
- `mallorn-astro-classification/` - Astronomy classification
- `kernelbench-triton-evolution/` - GPU kernel optimization
- `code-golf/` - Code size optimization
