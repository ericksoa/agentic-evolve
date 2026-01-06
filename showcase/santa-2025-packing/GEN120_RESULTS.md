# Gen120 Results

## Summary
- Starting score: 85.17
- Final score: 85.10
- Improvement: 0.07 points

## What Was Tried

### 1. Full-Configuration Simulated Annealing
Created `python/gen120_full_sa.py` with:
- SA on all trees together (not just boundary trees like Gen117)
- Metropolis criterion with temperature annealing
- Multiple restarts with different temperature schedules
- Aggressive exploration at high temperatures

**Result**: Found improvements on small n (2-10):
- n=2: 0.951 -> 0.950 (Δscore=0.0014)
- n=5: 1.501 -> 1.478 (Δscore=0.0139)
- n=7: 1.795 -> 1.754 (Δscore=0.0206)
- n=8: 1.968 -> 1.908 (Δscore=0.0292)
- n=9: 2.081 -> 2.066 (Δscore=0.0068)
- Total: ~0.07 points improvement

### 2. Exact Optimization for Small n (2-5)
Created `python/gen120_fast_search.py` with:
- Differential evolution on full parameter space
- Nelder-Mead refinement
- Specific angle pair search for n=2
- Multiple random restarts

**Result**: Very small improvements (~0.002 points on n=2-3). The current solutions are already near-optimal.

### 3. Strip Packing with Alternating Orientations
Created `python/gen120_strip_pack.py` with:
- Horizontal strip arrangement
- Alternating 0°/180° orientations
- Hexagonal close-packing patterns
- Radial with alternating angles

**Result**: No improvements. The greedy + SA approach in Rust produces better configurations.

## Analysis: Why the 23% Gap?

Current score: 85.10
Top leaderboard: 69.02
Gap: 16.08 points (23.3%)

### Theoretical Limits
- Tree area: 0.2456 square units
- Theoretical minimum (100% packing efficiency): 49.13
- Top solutions: 69.02 (40% above theoretical)
- Our solution: 85.10 (73% above theoretical)

### Efficiency Analysis
| Metric | Our Solution | Top Solutions | Gap |
|--------|--------------|---------------|-----|
| Total score | 85.10 | 69.02 | 16.08 |
| Avg per n | 0.4255 | 0.3451 | 0.0804 |
| Packing efficiency | ~57% | ~70% | ~13% |

### What Top Solutions Are Doing Differently
The gap suggests top solutions have found fundamentally better packing configurations. Possibilities:

1. **Different search algorithm**: Genetic algorithms, constraint optimization, or hybrid approaches
2. **More compute**: Running many more iterations (1000+ best-of-N instead of 20)
3. **Geometric insight**: Specific optimal arrangements that we haven't discovered
4. **Problem-specific patterns**: The tree shape may have optimal packing patterns we haven't found

## Key Learnings

1. **Local refinement has limits**: SA on full configurations helps slightly but doesn't bridge the gap
2. **Pattern-based approaches don't help**: Strip, hexagonal, and radial patterns all produce worse results than greedy + SA
3. **The Rust solver is well-optimized**: Multiple strategies and extensive SA already find near-optimal solutions for our approach
4. **Gap is fundamental, not incremental**: Need a paradigm shift, not parameter tuning

## Recommendations for Future Generations

1. **Population-based search**: Genetic algorithms with crossover on configurations
2. **Learned initialization**: Train a model to predict good initial placements
3. **Constraint programming**: Use OR-Tools or similar for exact optimization on small n
4. **Study top solutions**: When competition ends, analyze winning approaches

### 4. Genetic Algorithm
Created `python/gen120_genetic.py` with:
- Population-based search with crossover
- Tournament selection
- Mutation and repair operators
- Seeded with current best solutions

**Result**: No improvements found. The current solutions are already at optima that even population-based search can't escape.

## Files Created
- `python/gen120_full_sa.py` - Full-configuration Simulated Annealing
- `python/gen120_exact_small.py` - Exact optimization (DE-based)
- `python/gen120_fast_search.py` - Fast Nelder-Mead search
- `python/gen120_strip_pack.py` - Strip/hexagonal packing patterns
- `python/gen120_genetic.py` - Genetic algorithm with crossover
