# Sales Territory Optimization

Evolve algorithms that assign sales accounts to sales reps, optimizing for coverage, workload balance, and geographic efficiency.

## Problem

Given 25 sales accounts spread across a metro area, assign them to 4 sales reps such that:

1. **Coverage** - All accounts are assigned (maximize total revenue coverage)
2. **Balance** - Each rep has roughly equal revenue potential
3. **Compactness** - Territories are geographically tight (minimize travel)

This is a classic multi-objective combinatorial optimization problem that arises in sales operations, delivery routing, and service territory planning.

## Data

Synthetic accounts generated with:
- **Location**: Lat/lon coordinates clustered around a metro area
- **Revenue**: $10k - $500k annual contract value (log-normal distribution)
- **Industry**: Technology, Healthcare, Manufacturing, Retail, Financial Services

## Fitness Function

```
fitness = (coverage * 0.4) + (balance * 0.3) + (compactness * 0.3)
```

| Component | Description | Perfect Score |
|-----------|-------------|---------------|
| Coverage | % of total revenue assigned | 1.0 = all assigned |
| Balance | 1 - CV of rep revenues | 1.0 = equal revenue per rep |
| Compactness | Geographic tightness | 1.0 = tight clusters |

## Solution Interface

Solutions must implement:

```python
def assign_territories(accounts: list[dict], num_reps: int) -> dict[int, list[int]]:
    """
    Args:
        accounts: List of {id, lat, lon, revenue, industry, name}
        num_reps: Number of sales reps (e.g., 4)

    Returns:
        Dict mapping rep_id (0 to num_reps-1) -> list of account_ids
    """
```

## Quick Start

```bash
# View the synthetic data
python data.py

# Run baseline solution
python baseline.py

# Evaluate baseline fitness
python evaluate.py baseline.py --json

# Visualize territories (requires matplotlib, scipy)
python visualize.py baseline.py --output baseline_territories.png
```

## Running Evolution

```bash
# From the showcase directory
python -m evolve_sdk "Optimize territory assignments" --config evolve_config.json
```

## Evolution Results

Evolution ran for 7 generations with population size 4, producing 28 mutations.

### Performance Comparison

| Metric | Baseline | Best (gen2x) | Improvement |
|--------|----------|--------------|-------------|
| Fitness | 0.8818 | **0.9434** | +7.0% |
| Coverage | 1.0 | 1.0 | - |
| Balance | 0.7137 | **0.9889** | +38.5% |
| Compactness | 0.8923 | 0.8224 | -7.8% |

### Revenue Distribution

| Rep | Baseline | Evolved (gen2x) |
|-----|----------|-----------------|
| Alice | $926,112 | $664,095 |
| Bob | $421,824 | $659,420 |
| Carol | $728,241 | $665,274 |
| Dave | $559,446 | $646,834 |
| **Range** | **$504k** | **$19k** |

The evolved solution achieves nearly perfect revenue balance (range reduced from $504k to just $19k) while maintaining good geographic compactness.

### Key Innovations

The best solution (`evolved_best.py`) combines:
1. **Precomputed distance matrix** - O(1) distance lookups
2. **Recursive pivot-based partitioning** - Find farthest pair, split by proximity
3. **CV-based revenue balancing** - Move/swap operations to minimize coefficient of variation
4. **Compactness bonuses** - Prefer moves that keep territories geographically tight

## Baseline

The baseline uses simple k-means clustering - it assigns accounts to the nearest centroid without considering revenue balance.

## Evolution Strategies Explored

The evolve-sdk explored:

1. **Recursive pivot partitioning** - Split accounts using farthest-pair pivots
2. **K-means++ initialization** - Better seed selection for clustering
3. **CV-based balancing** - Move and swap operations targeting coefficient of variation
4. **Compactness bonuses** - Penalize moves that increase territory sprawl
5. **Genetic crossover** - Combine features from multiple parent solutions

## Dependencies

- Python 3.10+
- matplotlib (visualization)
- scipy (convex hull for territory boundaries)
- numpy
