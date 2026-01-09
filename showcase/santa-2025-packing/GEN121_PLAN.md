# Gen121 Plan

## CRITICAL CONSTRAINT
**NEVER start more than 4-6 parallel processes LOCALLY.** Running massive parallel workloads melts the local computer and impedes progress.

**ALLOWED**: Using external serverless cloud providers (lightning.ai, etc.) for heavy compute.

## Objectives
1. **Sequential Best-of-1000**: Run 1000 best-of-N trials per n value
   - Locally: Sequential or max 4 threads
   - Cloud option: lightning.ai for massive parallelism
2. **Evolve SDK Exploration**: Use `/evolve-sdk` to discover novel algorithmic approaches
   - Can leverage lightning.ai for GPU/CPU compute
   - Agents can spawn cloud jobs instead of local processes

## Current State
- Score: **85.10** (Gen120)
- Target: ~69.02 (top leaderboard)
- Gap: 23.3% (16 points)

## Approach 1: Best-of-1000 with Cloud Compute

### Option A: Local Sequential (Safe, Slow)
- Run Rust solver 1000 times per n value sequentially
- Max 4 concurrent processes
- Takes hours but won't crash machine

### Option B: Lightning.ai Cloud (Fast, Parallel)
- Deploy Rust solver to lightning.ai
- Run 1000 parallel jobs per n value
- Results aggregated back to local
- Cost: ~$X per run (estimate)

### Implementation Steps
1. Create lightweight Python wrapper for Rust solver
2. Configure lightning.ai job submission
3. Run batch jobs for each n
4. Collect and merge best results

## Approach 2: Evolve SDK with Cloud Backend

### Goal
Use Evolve SDK to generate algorithmic mutations with cloud-powered evaluation:
- Define fitness function (packing score)
- Mutations modify algorithm parameters or structure
- Evaluation runs on lightning.ai (GPU/CPU as needed)

### Architecture
```
Local (Claude/Agent)
  └── Evolve SDK orchestration
        └── lightning.ai jobs
              ├── Evaluate mutation A
              ├── Evaluate mutation B
              └── ...
        └── Collect results, select best
```

### Benefits
- No local resource constraints
- Massive parallelism available
- GPU acceleration if beneficial

## Execution Order
1. Set up lightning.ai integration (if user has account)
2. Start Best-of-1000 approach (cloud or local)
3. Run Evolve SDK exploration with cloud backend
4. Merge best results from both approaches

## Files Created
- `evolve_config.json` - Evolution configuration
- `evaluate_santa.py` - Local/cloud evaluation script
- `python/gen121_starter.py` - Starter solution for evolution
- `.env` - Lightning.ai credentials (copied from kernelbench)

## Execution Commands

### Best-of-1000 (Sequential, Safe)
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Run Rust solver 100 times per n (with 4 threads max)
./rust/target/release/ultimate_submission 100 4 submission_gen121.csv

# Compare with current best
python3 python/analyze_submission.py submission_gen121.csv
python3 python/analyze_submission.py submission_best.csv
```

### Evolve SDK
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Run with limited parallelism
/Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing/ml_env/bin/python -m evolve_sdk \
  --config evolve_config.json \
  --max-generations 20 \
  --population-size 6 \
  --no-parallel

# Or with 4-worker limit
/Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing/ml_env/bin/python -m evolve_sdk \
  --config evolve_config.json \
  --max-workers 4
```

## Recovery Notes
If session restarts:
1. Read this file for context
2. Check `gen121_results.log` for progress
3. Check `.evolve-sdk/` directory for evolution state
4. Continue from last checkpoint

## Progress Log
- [x] Setup lightning.ai credentials
- [x] Created evolve_config.json
- [x] Created evaluate_santa.py (updated to handle PlacedTree objects)
- [x] Created gen121_starter.py
- [x] Tested evaluation locally (works)
- [x] Installed Claude Agent SDK
- [x] Ran Evolve SDK (5 generations, champion=gen0_d.py)
- [x] Ran Best-of-100 (score=85.39, 5.01% improvement from variance)
- [x] Documented results in GEN121_RESULTS.md

## Evolve SDK Results (COMPLETED)
- **Champion**: gen0_d.py (Bottom-Left-Fill Heuristic)
- **Fitness**: 27.71 (higher is better)
- **Generations**: 5 (stopped at plateau threshold)
- **Initial population** (Gen 0):
  | Approach | Fitness |
  |----------|---------|
  | Bottom-Left-Fill (champion) | 27.71 |
  | Greedy Spiral | 17.67 |
  | Grid-Based DP | 16.36 |
  | Physics-Based Simulation | 14.62 |
  | Genetic Algorithm | 14.59 |
  | Multi-Scale SA | 13.37 |
- **Key Finding**: Initial population contained best solution; later mutations couldn't improve

## Best-of-100 Results (COMPLETED)
- **Score**: 85.39 (vs current best 85.10 from Gen120)
- **Improvement from variance**: +5.01% (first run was 89.89)
- **N values improved**: 199/200
- **Compute time**: 93 minutes with 4 threads
- **Key Finding**: Rust-only Best-of-100 (85.39) is worse than Rust+Python (85.10)
- **Conclusion**: Python refinements (CMA-ES, full-config SA) add real value

## Final Status: NO IMPROVEMENT
Both approaches (Evolve SDK novel algorithms + Best-of-100 variance) did not beat Gen120's 85.10.
See GEN121_RESULTS.md for detailed analysis.

## Lightning.ai Cloud Test (COMPLETED)
- Tested cloud compute as alternative to local parallelism
- Result: **4.1x slower** than local M3 Mac due to:
  - 27% CPU steal time (shared VMs)
  - Slower per-core performance (Intel Xeon vs M3)
- Score: 85.53 (worse than local's 85.39 due to variance)
- **Conclusion**: Lightning.ai CPU unsuitable for this workload
- For cloud benefit, would need many parallel studios (horizontal scaling)
