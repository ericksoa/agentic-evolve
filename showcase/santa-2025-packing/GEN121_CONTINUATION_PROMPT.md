# Gen121 Continuation Prompt

## Current State
- Score: **85.10** (Gen120, validated and submitted to Kaggle)
- Target: ~69.02 (top leaderboard)
- Gap: 23.3% (16 points)

## Quick Commands
```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Validate
python3 python/validate_submission.py submission_best.csv

# Analyze score breakdown
python3 python/analyze_submission.py submission_best.csv

# Check Kaggle leaderboard
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions leaderboard santa-2025 --show

# Submit to Kaggle
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions submit -c santa-2025 -f submission_best.csv -m "message"
```

## What's Been Tried (Gen117-120)
| Generation | Approach | Result |
|------------|----------|--------|
| Gen117 | Pattern CMA-ES, boundary SA | No improvement |
| Gen118 | Angle-only refinement | 0.04 points |
| Gen119 | Position + angle refinement | 0.24 points |
| Gen120 | Full-config SA, genetic algorithm, strip packing, exact search | 0.07 points |

## The Problem
Local refinement approaches keep hitting the same wall. Top solutions achieve ~70% packing efficiency; we're at ~57%. This is a paradigm problem, not a parameter tuning problem.

## Untried Ideas for Gen121
1. **Massive brute force** - 1000+ best-of-N restarts instead of 20
2. **Novel initialization strategies** - Tetris-style, compression waves, MCTS
3. **Constraint programming** - OR-Tools for small n exact solutions
4. **CMA-ES on relative positions** - Optimize distances/angles between trees
5. **Divide and conquer** - Solve n/2, mirror/merge

## Key Files
- `submission_best.csv` - Current best (85.10)
- `rust/src/evolved.rs` - Sophisticated Rust solver
- `python/gen120_*.py` - Gen120 optimization attempts
- `GEN120_RESULTS.md` - Detailed analysis of what was tried

## Request
Pick up where we left off on the Santa 2025 packing challenge. Ready to start Gen121 - need to choose a direction.
