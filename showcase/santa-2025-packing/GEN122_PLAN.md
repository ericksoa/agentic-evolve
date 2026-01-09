# Gen122 Plan: Evolve SDK for Santa Packing

## Objective
Use the `evolve_sdk` Python package to evolve novel packing algorithms that beat the current champion (85.10).

## Current State
- **Score**: 85.10 (Gen120)
- **Target**: ~69.02 (top leaderboard)
- **Gap**: 23.3% (16 points)

## Why Evolve SDK (vs Gen121)?

Gen121 ran 5 generations with basic settings. Gen122 will use:
- **More generations** (20-30 vs 5)
- **Larger population** (10 vs 6)
- **Better crossover** (mandatory 50% per generation)
- **Plateau detection** (stop after 5 gens without improvement)
- **Proper mode** (perf mode for optimization)

## Configuration

Using existing `evolve_config.json`:
```json
{
  "mode": "perf",
  "description": "Evolve better Christmas tree packing algorithms",
  "evaluation": {
    "test_command": "python evaluate_santa.py {solution} --json"
  },
  "optimization_strategies": [
    "hierarchical_packing",
    "constraint_programming",
    "tree_shape_exploitation",
    "nfp_optimization",
    ...
  ]
}
```

## Execution Command

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Run evolution with SDK
source ml_env/bin/activate
python -m evolve_sdk \
  --config evolve_config.json \
  --mode perf \
  --max-generations 20 \
  --population-size 10 \
  --plateau 5 \
  --no-parallel
```

### Parameters Explained
- `--config evolve_config.json` - Load problem definition and evaluation command
- `--mode perf` - Optimize for performance (lower score = better)
- `--max-generations 20` - Run up to 20 generations
- `--population-size 10` - Maintain 10 diverse solutions
- `--plateau 5` - Stop if no improvement for 5 generations
- `--no-parallel` - Run sequentially (respects 4-thread limit)

## Evaluation Script

`evaluate_santa.py` tests solutions on n=5..20:
1. Imports the solution module
2. Calls `pack(n)` for each test n
3. Calculates bounding box and score
4. Checks for overlaps
5. Returns `{"valid": true, "fitness": <value>}` as JSON

Fitness formula: `1.0 / (avg_score + 0.1)` (higher is better)

## SDK Evolution Flow

```
Gen 0: Initialize population
  └── Creates 10 diverse algorithms using different strategies

Gen 1+: Evolution loop
  ├── Select top 3 (elite)
  ├── Spawn 4 mutators (one per top solution)
  ├── Spawn 1 crossover (combines top 2)
  ├── Evaluate all 5 new solutions
  ├── Update population (keep top 10)
  └── Check plateau (stop if no improvement)

Final: Save champion
  └── .evolve-sdk/santa-packing/champion.json
```

## Expected Output

```
═══════════════════════════════════════════════════════════════════════
  Evolve SDK - santa-2025-packing (perf mode)
  Working directory: .evolve-sdk/evolve_better_christmas_tree_p
═══════════════════════════════════════════════════════════════════════

[Gen 0] Initializing population...
[Gen 0] Created 10 initial solutions
[Gen 0] Best initial fitness: 2.34

───────────────────────────────────────────────────────────────────────
  Generation 1 | Champion: gen0_d.py (fitness: 2.34) | Pop: 10 | Plateau: 0/5
───────────────────────────────────────────────────────────────────────

Spawning agents:
  a: mutation of gen0_d.py
  b: mutation of gen0_c.py
  c: mutation of gen0_a.py
  x: crossover of gen0_d.py+gen0_c.py

Results:
  [a] KEEP  fitness: 2.41 (+0.07)
  [b] DROP  fitness: 2.10 (regression)
  [c] KEEP  fitness: 2.38 (+0.04)
  [x] KEEP  fitness: 2.52 (+0.18) ★ NEW CHAMPION

[+] Improvement: 2.34 -> 2.52 (+0.18)

... (more generations)

[=] No improvement (plateau: 5/5)

═══════════════════════════════════════════════════════════════════════
  Evolution Complete
  Generations: 12
  Final champion: gen4_x.py (fitness: 2.89)
═══════════════════════════════════════════════════════════════════════
```

## Recovery Notes

If session restarts:
```bash
# Check state
cat .evolve-sdk/evolve_better_christmas_tree_p/evolution.json

# Resume from last checkpoint
python -m evolve_sdk --resume
```

## Critical Constraints

**NEVER start more than 4 parallel processes locally.**
- Using `--no-parallel` to ensure sequential execution
- Each evaluation runs with max 4 threads internally

## Progress Log
- [ ] Run evolution with SDK
- [ ] Monitor progress
- [ ] Document results in GEN122_RESULTS.md
- [ ] Commit if improvement found

## Post-Evolution

If champion improves on 85.10:
1. Extract winning algorithm
2. Test on full n=1..200
3. Run Python refinement (CMA-ES, SA)
4. Validate with `python/validate_submission.py`
5. Update submission_best.csv
6. Submit to Kaggle
