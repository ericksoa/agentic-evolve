# Gen107 Plan: Hybrid GPU/CPU Optimization (Based on 70.1 Analysis)

## Key Insight from 70.1 Solution

The 70.1 solution uses a **hybrid approach**:
1. **GPU for fast filtering**: Bbox overlap, transforms, scoring
2. **CPU for accurate collision**: Polygon intersection only for bbox-overlapping pairs
3. **Multi-stage pipeline**: Rotation → Compaction → SA → Polish

Our Gen106 failure was using bbox as the ONLY collision check. The fix is to use it as a FILTER.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gen107 Hybrid Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. GPU: BATCH TRANSFORM                                         │
│     - Rotate/translate all trees in all configs                  │
│     - Output: (batch, n_trees, 15, 2) vertices                   │
│                                                                  │
│  2. GPU: BBOX FILTER (Fast)                                      │
│     - Compute all pairwise bbox overlaps                         │
│     - Output: List of (config_idx, tree_i, tree_j) candidates    │
│     - Filters out ~90% of pairs                                  │
│                                                                  │
│  3. CPU: POLYGON COLLISION (Accurate)                            │
│     - Only check bbox-overlapping pairs                          │
│     - Use SAT or edge intersection                               │
│     - Output: Actual collision count per config                  │
│                                                                  │
│  4. GPU: SCORING                                                 │
│     - Compute side_length for all configs                        │
│     - Combine with collision penalty                             │
│                                                                  │
│  5. OPTIMIZATION LOOP                                            │
│     - SA with GPU-accelerated moves                              │
│     - Multi-stage: rotation opt → compaction → SA → polish       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Phase 1: Hybrid Collision Detection

```python
def hybrid_check_overlaps(configs, tree_tensor):
    """
    Hybrid GPU/CPU overlap detection.

    1. GPU: Fast bbox filter
    2. CPU: Accurate polygon check for candidates
    """
    # GPU: Transform all trees
    vertices = gpu_transform_trees(tree_tensor.base_vertices, configs)
    bbox = gpu_compute_bbox(vertices)

    # GPU: Get bbox overlap candidates
    bbox_overlaps = gpu_check_bbox_overlaps(bbox)  # (batch, n, n) bool

    # For each config, check actual polygon overlap
    actual_overlaps = torch.zeros(configs.shape[0], device='cpu')

    for b in range(configs.shape[0]):
        # Get pairs that have bbox overlap
        pairs = bbox_overlaps[b].nonzero()  # (num_pairs, 2)

        if pairs.shape[0] == 0:
            continue

        # CPU: Check polygon overlap for each candidate pair
        verts_b = vertices[b].cpu().numpy()  # (n_trees, 15, 2)

        for i, j in pairs.cpu().numpy():
            if i >= j:  # Skip duplicates and diagonal
                continue
            if polygons_overlap(verts_b[i], verts_b[j]):
                actual_overlaps[b] += 1

    return actual_overlaps.to(configs.device)
```

### Phase 2: Optimized Polygon Collision (Numba)

Use Numba JIT to accelerate CPU polygon checks:

```python
@numba.jit(nopython=True)
def polygons_overlap_fast(poly1, poly2):
    """SAT-based polygon overlap using Numba."""
    # Check edge normals of poly1
    for i in range(len(poly1)):
        j = (i + 1) % len(poly1)
        edge = (poly1[j][0] - poly1[i][0], poly1[j][1] - poly1[i][1])
        normal = (-edge[1], edge[0])

        # Project both polygons onto normal
        min1, max1 = project_polygon(poly1, normal)
        min2, max2 = project_polygon(poly2, normal)

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found

    # Repeat for poly2 edges...
    # (similar code)

    return True  # No separating axis = overlap
```

### Phase 3: Multi-Stage Optimization

Based on 70.1 approach:

```python
def optimize_group(n, initial_config=None):
    """Multi-stage optimization pipeline."""

    # Stage 1: Initial placement (from Rust greedy)
    if initial_config is None:
        config = rust_greedy_pack(n)
    else:
        config = initial_config

    # Stage 2: Global rotation optimization
    config = optimize_global_rotation(config)

    # Stage 3: Compaction passes
    for _ in range(15):
        config = compaction_pass(config)

    # Stage 4: Squeeze toward center
    config = squeeze_toward_center(config, factor=0.99)

    # Stage 5: Simulated annealing
    config = gpu_simulated_annealing(
        config,
        iterations=40000,
        restarts=6,
        move_types={
            'position': 0.4,
            'rotation': 0.3,
            'combined': 0.2,
            'squeeze': 0.1
        }
    )

    # Stage 6: Final polish
    config = final_polish(config)

    return config
```

### Phase 4: SA Move Types (from 70.1)

```python
def sa_move(config, move_type):
    """Generate SA move based on type."""

    if move_type == 'position':
        # Move random tree by small delta
        tree_idx = random.randint(0, n-1)
        config[tree_idx, :2] += np.random.randn(2) * position_scale

    elif move_type == 'rotation':
        # Rotate random tree
        tree_idx = random.randint(0, n-1)
        config[tree_idx, 2] += np.random.randn() * angle_scale

    elif move_type == 'combined':
        # Move and rotate together
        tree_idx = random.randint(0, n-1)
        config[tree_idx, :2] += np.random.randn(2) * position_scale
        config[tree_idx, 2] += np.random.randn() * angle_scale

    elif move_type == 'squeeze':
        # Squeeze all trees toward center
        center = config[:, :2].mean(axis=0)
        config[:, :2] = config[:, :2] * 0.995 + center * 0.005

    return config
```

## Key Parameters (from 70.1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| SA iterations | 40,000 | Per group |
| Restarts | 6 | Different random seeds |
| Compaction passes | 15 | After rotation opt |
| Workers | 8 | Parallel group processing |
| Position move prob | 40% | |
| Rotation move prob | 30% | |
| Combined move prob | 20% | |
| Squeeze move prob | 10% | |
| Temperature schedule | Exponential | T = T0 * exp(log_ratio * progress) |

## File Structure

```
santa-2025-packing/
├── python/
│   ├── gpu_primitives.py      # GPU operations (DONE)
│   ├── hybrid_collision.py    # NEW: Hybrid GPU/CPU collision
│   ├── polygon_collision.py   # NEW: Numba-accelerated polygon SAT
│   ├── multi_stage_opt.py     # NEW: Multi-stage optimizer
│   ├── gpu_sa.py              # NEW: GPU-accelerated SA
│   └── gen107_runner.py       # NEW: Main runner
```

## Expected Performance

| Component | Time (n=200) |
|-----------|--------------|
| GPU transform | 0.25 ms |
| GPU bbox filter | 0.63 ms |
| CPU polygon check (filtered) | ~5-10 ms |
| Full evaluation | ~10 ms |
| **SA iteration** | **~10 ms** |
| **40k iterations** | **~7 min per group** |
| **Full submission (200 groups)** | **~3 hours with 8 workers** |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CPU polygon check too slow | Use Numba, parallelize with ThreadPoolExecutor |
| MPS doesn't support all ops | Fall back to CPU for unsupported |
| Still can't close 24% gap | At least faster iteration, learn from results |

## Success Criteria

1. **Minimum**: Valid submissions with GPU acceleration working
2. **Target**: Match current 86.17 with 2x faster runtime
3. **Stretch**: Break 85.0 (1.4% improvement toward 70)

## Implementation Order

1. `polygon_collision.py` - Numba SAT implementation
2. `hybrid_collision.py` - Integrate GPU bbox + CPU polygon
3. `gpu_sa.py` - SA with hybrid collision
4. `multi_stage_opt.py` - Full pipeline
5. `gen107_runner.py` - Benchmark and submission

## References

- [70.1 Solution](https://github.com/berkaycamur/Santa-Competition)
- [NVIDIA GPU Gems: Collision Detection](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
- [SAT Algorithm](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem)
