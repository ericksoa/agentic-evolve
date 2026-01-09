# Gen122 Results: Evolve SDK Evolution

## Summary
- **Starting score**: 85.10 (Gen120)
- **Final score**: 85.10 (no improvement)
- **Goal**: ~69.02 (top leaderboard)
- **Gap**: Still 23.3%

## Evolution Run

Used `evolve_sdk` Python package to evolve novel packing algorithms:

```bash
python -m evolve_sdk \
  --config evolve_config.json \
  --mode perf \
  --max-generations 20 \
  --population-size 10 \
  --plateau 5 \
  --no-parallel
```

### Results

| Metric | Value |
|--------|-------|
| Generations | 10 (5 from Gen121 + 5 new) |
| Population size | 10 |
| Plateau threshold | 5/5 (hit) |
| Champion | gen0_d.py (Bottom-Left-Fill) |
| Champion fitness | 27.7128 |

### Population (Final)

| Rank | Algorithm | Fitness | Approach |
|------|-----------|---------|----------|
| 1 | gen0_d.py | 27.71 | Bottom-Left-Fill Heuristic |
| 2 | gen0_a.py | 17.67 | Greedy Spiral Packing |
| 3 | gen0_b.py | 16.36 | Grid-Based DP |
| 4 | gen0_c.py | 14.62 | Physics-Based Simulation |
| 5 | gen0_e.py | 14.59 | Genetic Algorithm |
| 6 | gen0_f.py | 13.37 | Simulated Annealing |
| 7-10 | Mutations | 0.15-0.80 | Various improvements |

### Fitness Progression

```
Gen 1:  27.7128 (no change)
Gen 2:  27.7128 (no change)
Gen 3:  27.7128 (no change)
Gen 4:  27.7128 (no change)
Gen 5:  27.7128 (no change)
Gen 6:  27.7128 (no change)
Gen 7:  27.7128 (no change)
Gen 8:  27.7128 (no change)
Gen 9:  27.7128 (no change)
Gen 10: 27.7128 (plateau reached)
```

## Champion Algorithm

The winning algorithm (`gen0_d.py`) is a **Bottom-Left-Fill Heuristic**:

```python
# Key features:
# 1. Scan positions from bottom-left to top-right
# 2. Place tree at first valid position
# 3. Try different rotations (0, 90, 180, 270 degrees)
# 4. Pick rotation that minimizes bounding box
# 5. Use bounding box for fast collision detection
```

This simple greedy approach outperformed:
- Spiral packing
- Grid-based dynamic programming
- Physics simulation
- Genetic algorithms
- Simulated annealing
- All crossover hybrids

## Key Learnings

1. **Simple heuristics are hard to beat**: The Bottom-Left-Fill heuristic from the initial population was never improved upon in 10 generations.

2. **LLM-generated mutations regress**: All mutations and crossovers produced lower fitness (0.15x - 0.80x vs champion's 27.71x). The LLM struggles to maintain algorithmic quality while innovating.

3. **Initial population contains the winner**: Like Gen121, the best solution came from the initial diverse population, not from evolution.

4. **Fitness function matters**: The evaluation only tests n=5..20, so algorithms optimized for this range may not generalize to n=1..200.

5. **Our Rust solution is still superior**: The champion Python BLF algorithm (fitness 27.71) is testing on a subset of n values. Our Rust solution with SA+CMA-ES refinement (score 85.10) operates on all 200 n values with much more sophisticated optimization.

## Comparison with Rust Solution

| Aspect | Python BLF (champion) | Rust Evolved |
|--------|----------------------|--------------|
| Algorithm | Bottom-Left-Fill | Greedy + SA + Wave |
| Optimization | None (greedy only) | SA + CMA-ES + Python refinement |
| Test range | n=5..20 | n=1..200 |
| Collision | Bounding box only | Segment intersection |
| Angles | 8 discrete (45° steps) | Continuous |
| Score | Unknown (subset) | 85.10 (full) |

## Why No Improvement?

1. **Evaluation gap**: Evolve SDK tests on n=5..20, but the real challenge is n=100..200 where packing density matters most.

2. **Algorithm complexity**: Novel LLM-generated algorithms are often buggy or suboptimal. Simple proven heuristics are more reliable.

3. **No iterative refinement**: The Python algorithms lack the SA/CMA-ES post-processing that makes our Rust solution effective.

4. **Search space mismatch**: Evolving Python code operates at the wrong abstraction level. Better to evolve parameters/weights within a fixed algorithm structure.

## Recommendations for Future

1. **Evolve within existing framework**: Instead of evolving whole algorithms, evolve parameters/weights of the Rust solver.

2. **Better fitness function**: Evaluate on full n=1..200 range with the actual scoring metric.

3. **Hybrid approach**: Use LLM to suggest structural improvements, but validate carefully before accepting.

4. **Focus on bottlenecks**: The gap to 69.02 requires algorithmic breakthroughs (NFP, constraint programming), not incremental evolution.

## Files

- `.evolve-sdk/evolve_better_christmas_tree_p/` - Evolution state and mutations
- `.evolve-sdk/evolve_better_christmas_tree_p/champion.json` - Champion solution
- `evolve_config.json` - Evolution configuration
- `evaluate_santa.py` - Fitness evaluation script
