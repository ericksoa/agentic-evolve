# Gen121 Continuation Prompt

## Current State
- Score: 85.10 (Gen120)
- Target: ~69 (top leaderboard)
- Gap: 23.3% (16 points)

## What's Been Tried (Gen117-120)

| Generation | Approach | Result |
|------------|----------|--------|
| Gen117 | Pattern CMA-ES, boundary SA | No improvement |
| Gen118 | Angle-only refinement | 0.04 points |
| Gen119 | Position + angle refinement | 0.24 points |
| Gen120 | Full-config SA, exact search, strip packing | 0.07 points |

All approaches hit the same wall: local refinement can't bridge a 23% gap.

## The Fundamental Problem

Top solutions achieve ~70% packing efficiency. We're at ~57%.
This isn't a parameter tuning problem - it's an algorithmic paradigm problem.

## Ideas Not Yet Tried

### A. Genetic Algorithm with Crossover
The key insight SA lacks: **combining good sub-solutions**.

```
Population: 100 configurations per n
Fitness: bounding box side length
Crossover: Exchange subsets of trees between parents
Mutation: Position/angle perturbation
Selection: Tournament selection
```

Why this might work:
- Can combine good arrangements from different configurations
- Maintains diversity to avoid local minima
- Works well for combinatorial optimization

### B. Reinitialize from Scratch with Different Strategy
The Rust solver uses 6 strategies. What if we tried 20 more?

Ideas for new initialization strategies:
1. **Tetris-style**: Add trees to "lowest" valid position
2. **Compression waves**: Start spread out, compress in waves
3. **Template-based**: For specific n, use known good templates
4. **Monte Carlo Tree Search**: For small n, search placements

### C. CMA-ES on Relative Positions
Instead of optimizing absolute positions, optimize:
- Relative distances between trees
- Angular offsets from neighbors
- This reduces dimensionality and maintains structure

### D. Divide and Conquer
For large n:
1. Solve for n/2 optimally
2. Mirror/rotate that solution
3. Merge and refine

### E. Learn from Submission History
Use best solutions from each n to learn patterns:
- What angles are common?
- What spacings work?
- Can we extract templates?

## Quick Commands

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Current best
python3 python/validate_submission.py submission_best.csv

# Score breakdown
python3 python/analyze_submission.py submission_best.csv
```

## Priority Recommendation

**Try Genetic Algorithm first.** It's the most different approach from what we've tried and has strong theoretical motivation for escaping local optima.

## Key Files
- `submission_best.csv` - Current best (85.10)
- `rust/src/evolved.rs` - The sophisticated Rust solver
- `python/gen120_full_sa.py` - Reference for SA implementation
