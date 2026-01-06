# Gen116 Plan: Four-Pronged Optimization Attack

## Current State
- **Score**: 85.45 (Gen115)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

## Priority 1: Large Search Budget for Medium n (11-30)

### Rationale
- Medium n values (11-30) contribute ~20% of total score
- Currently using Rust Best-of-20 output with no Python optimization
- CMA-ES quick tests (3 restarts, 5k evals) found no improvements
- **Hypothesis**: Need 10x+ search budget to find improvements

### Approach
```python
# For n in 11-30:
restarts = 10
max_evals = 20000  # 10x current
sigma = 0.25       # Larger search radius
penalty = 5000     # High overlap penalty
```

### Expected Impact
- If 1% improvement average: ~0.2 score points
- Focus on n=11-15 first (easier to optimize)

## Priority 2: Pattern-Based Optimization

### Rationale
- Top solutions likely use structured patterns
- Known patterns: radial, hexagonal, diagonal strips
- Current solutions may be stuck in chaotic local optima

### Patterns to Try
1. **Radial with angle offset**: Trees arranged in rings, each ring rotated
2. **Diagonal strips**: Trees aligned along 45° diagonals
3. **Hexagonal close-pack**: Modified for tree shape
4. **180° pairs**: Two trees facing opposite directions (works well for small n)

### Implementation
```python
def radial_pattern(n):
    """Initialize trees in concentric rings."""
    trees = []
    ring = 0
    while len(trees) < n:
        radius = 0.3 + ring * 0.4
        count = max(1, int(6 * (ring + 1)))
        for i in range(count):
            if len(trees) >= n:
                break
            angle = 2 * pi * i / count + ring * pi/6  # Offset per ring
            x = radius * cos(angle)
            y = radius * sin(angle)
            tree_angle = degrees(angle) + 90  # Point outward
            trees.append(Tree(x, y, tree_angle))
        ring += 1
    return trees
```

### Expected Impact
- Could unlock 2-5% improvements for medium/large n
- Especially helpful for n=20-50 range

## Priority 3: Large n (50+) Optimization

### Rationale
- n=50-200 contributes ~60% of total score
- These are hardest to optimize (high dimensionality)
- Even 0.5% improvement = significant score delta

### Approach
1. **Extract current solutions** from submission_best.csv
2. **Run lighter optimization**: SA with 50k iterations
3. **Focus on boundary trees**: Only optimize trees on bbox boundary
4. **Validate strictly** before accepting

### Implementation
```python
def optimize_large_n(n, initial_trees):
    """Optimize large n with boundary focus."""
    # Identify boundary trees
    bbox = compute_bbox(initial_trees)
    boundary_indices = [i for i, t in enumerate(initial_trees)
                        if is_on_boundary(t, bbox)]

    # SA only on boundary trees
    for iteration in range(50000):
        idx = random.choice(boundary_indices)
        # ... SA move ...
```

### Expected Impact
- 0.1-0.3 score points if 0.5% average improvement

## Priority 4: CMA-ES + SA Hybrid

### Rationale
- CMA-ES good at global search, SA good at local refinement
- Combine: CMA-ES to find promising regions, SA to refine

### Approach
```python
def hybrid_optimize(n, initial_trees):
    # Phase 1: CMA-ES exploration (fewer evals, larger sigma)
    cmaes_result = optimize_with_cmaes(
        initial_trees,
        max_evals=2000,
        sigma=0.3,  # Large search
        penalty_weight=1000
    )

    # Phase 2: SA refinement (many iterations, small moves)
    sa_result = simulated_annealing(
        cmaes_result,
        iterations=50000,
        initial_temp=0.5,
        move_scale=0.02
    )

    return sa_result
```

### Expected Impact
- May find solutions neither method finds alone
- Especially useful for medium n (10-30)

## Implementation Order

1. **Priority 1 first** (easiest, most likely to work)
2. **Priority 4 next** (builds on existing code)
3. **Priority 2** (requires new pattern code)
4. **Priority 3 last** (hardest, lowest expected yield)

## Success Criteria

| Target | Score | Improvement |
|--------|-------|-------------|
| Stretch | 85.0 | -0.45 |
| Goal | 85.2 | -0.25 |
| Minimum | 85.35 | -0.10 |

## Files to Create

- `python/gen116_medium_n.py` - Medium n optimizer
- `python/gen116_patterns.py` - Pattern-based initialization
- `python/gen116_large_n.py` - Large n boundary optimizer
- `python/gen116_hybrid.py` - CMA-ES + SA hybrid

## Validation Checklist

Before any submission:
1. Run `python3 python/validate_submission.py submission_best.csv`
2. Check for overlaps using strict segment intersection
3. Verify score is actually better than 85.45
