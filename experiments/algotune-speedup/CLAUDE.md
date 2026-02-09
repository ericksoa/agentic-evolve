# AlgoTune Speedup Experiment (WIP)

**Status: Custom validation was flawed. Official eval shows ~1.01x. Needs re-work.**

## Workflow

### First-Time Setup
```bash
cd experiments/algotune-speedup
bash setup.sh
source .venv/bin/activate
export PYTHONPATH=$(pwd)/algotune:$PYTHONPATH
```

### Running the Benchmark

**Phase 1: Baseline (AlgoTune agent with Opus 4.6)**
```bash
# Smoke test on 2-3 tasks first
python scripts/run_phase1.py --tasks 3 --dry-run
python scripts/run_phase1.py --tasks 3

# Full run (~25 tasks, ~2 hours)
python scripts/run_phase1.py
```

**Phase 2: Evolution (evolve-sdk on top of Phase 1)**
```bash
# Smoke test
python scripts/run_phase2.py --tasks 3

# Full run (~25 tasks, ~12 hours, run overnight)
python scripts/run_phase2.py
```

**Aggregate Results**
```bash
python scripts/aggregate_results.py
```

### Resuming
- Phase 1 saves results per-task to `results/phase1_baseline.json` incrementally
- Phase 2 uses evolve-sdk checkpointing (`.evolve-sdk/` dirs) — auto-resumes
- To re-run a specific task: `python scripts/run_phase2.py --task svm`

## Resource Limits
- **Max 4-6 parallel processes** on this machine
- All evolution runs use `--no-parallel` (sequential mutations)
- Population size capped at 6
- One task at a time

## Architecture
- `evaluate_task.py` bridges evolve-sdk to AlgoTune's timing_core
- Phase 1 uses AlgoTune's built-in agent (AlgoTuner/main.py)
- Phase 2 layers evolve-sdk evolution on Phase 1 solutions
- All timing uses AlgoTune's own methodology for comparability

## Key Files
| File | Purpose |
|------|---------|
| `evaluate_task.py` | Bridge: evolve-sdk fitness ↔ AlgoTune timing |
| `scripts/select_subset.py` | Pick tasks with most optimization headroom |
| `scripts/run_phase1.py` | Run AlgoTune agent baseline per task |
| `scripts/run_phase2.py` | Run evolve-sdk evolution per task |
| `scripts/aggregate_results.py` | Compute harmonic mean, generate tables |
