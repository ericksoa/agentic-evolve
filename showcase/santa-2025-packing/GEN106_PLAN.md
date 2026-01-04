# Gen106 Plan: GPU-Accelerated Optimization

## What We Learned (Gen104-105)

### Gen104: ML Selection
- Trained pairwise ranking model (80.87% validation accuracy)
- **Failed**: Post-hoc selection cannot beat min(side_length) - the metric IS the objective
- **Key insight**: ML can only help DURING search, not after

### Gen105: Global Rotation + Squeeze
- Implemented based on 70.1 GitHub solution analysis
- **Failed**: All variants performed worse than baseline
- **Root causes**:
  1. Global rotation after compaction creates overlaps (trees too tightly packed)
  2. O(n²) overlap validation is expensive, rejects most rotations
  3. Squeeze after wave_compaction redundant (already maximally compressed)
  4. Our greedy approach incompatible with their GPU-parallel pipeline

### 70.1 Solution Analysis
From `berkaycamur/Santa-Competition`:
- Uses CUDA/CuPy for GPU acceleration
- Custom CUDA kernels for tree transformation
- **Key insight**: GPU handles bbox checks, CPU handles polygon validation only for candidates
- Pipeline: rotation → compaction (15 passes) → squeeze → SA (80 worst groups) → polish
- Continuous angles during SA (not just 45° discrete)
- 40,000 iterations, 6 restarts per group

## The Gap Analysis

| Metric | Our Best | Top Leaderboard | Gap |
|--------|----------|-----------------|-----|
| Score | 86.17 | ~69 | 24% |
| Approach | Greedy + SA | Unknown (likely GPU + global opt) | Fundamental |

The gap is NOT closable with local improvements. Need paradigm shift.

## Gen106 Design: GPU-Parallel Population Optimization

### Core Idea
Use GPU to evaluate MANY candidate configurations in parallel, then evolve the population.

### Hardware
- Apple M2 Pro (19 GPU cores)
- PyTorch 2.9.1 with MPS (Metal Performance Shaders)
- ~8GB unified memory available for GPU tensors

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gen106 Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INITIALIZATION (CPU)                                    │
│     - Generate P initial configurations (P=32-64)           │
│     - Use Gen91b greedy as seed for diversity               │
│                                                             │
│  2. GPU-PARALLEL EVALUATION                                 │
│     - Batch all P configurations as tensors                 │
│     - Compute all pairwise bbox overlaps in parallel        │
│     - Compute side_length for all P configs                 │
│     - Mark invalid configs (overlaps) with penalty          │
│                                                             │
│  3. EVOLUTIONARY OPERATORS                                  │
│     - Selection: Keep top K configs (K=P/2)                 │
│     - Mutation: Small perturbations (position, rotation)    │
│     - Crossover: Swap tree positions between configs        │
│     - Immigration: Inject fresh random configs              │
│                                                             │
│  4. GPU-PARALLEL SA REFINEMENT                              │
│     - Run SA on top K configs in parallel                   │
│     - Each chain: 1000 iterations                           │
│     - Share best solution across chains                     │
│                                                             │
│  5. ITERATE until convergence                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key GPU Optimizations

1. **Batched Overlap Detection**
   ```python
   # Instead of O(n²) sequential checks:
   # Compute all pairwise bbox overlaps as matrix operation

   # trees: (batch, n_trees, 4) - x, y, angle, tree_id
   # bbox: (batch, n_trees, 4) - min_x, min_y, max_x, max_y

   # Pairwise overlap matrix: (batch, n_trees, n_trees)
   overlap = (bbox[:,:,None,2] > bbox[:,None,:,0]) & \  # max_x1 > min_x2
             (bbox[:,:,None,0] < bbox[:,None,:,2]) & \  # min_x1 < max_x2
             (bbox[:,:,None,3] > bbox[:,None,:,1]) & \  # max_y1 > min_y2
             (bbox[:,:,None,1] < bbox[:,None,:,3])      # min_y1 < max_y2
   ```

2. **Batched Side Length Computation**
   ```python
   # Compute side_length for all configs in parallel
   min_coords = bbox[:,:,:2].min(dim=1)  # (batch, 2)
   max_coords = bbox[:,:,2:].max(dim=1)  # (batch, 2)
   side_length = (max_coords - min_coords).max(dim=1)  # (batch,)
   ```

3. **Parallel SA Chains**
   ```python
   # Run K independent SA chains on GPU
   # Each chain mutates different trees
   # Share best solution via reduction every N iterations
   ```

### Implementation Plan

#### Phase 1: GPU Primitives (Python + PyTorch)
1. `gpu_transform_trees()` - Rotate and translate trees on GPU
2. `gpu_compute_bbox()` - Compute bounding boxes for all trees
3. `gpu_check_overlaps()` - Batched pairwise overlap detection
4. `gpu_score_configs()` - Score multiple configurations

#### Phase 2: Population Optimization
1. Initialize population from Gen91b seeds
2. Implement evolutionary operators
3. GPU-parallel evaluation loop
4. Convergence detection

#### Phase 3: Parallel SA Refinement
1. Run SA chains on GPU
2. Implement move operators (translate, rotate)
3. Cross-chain communication for best solution sharing

#### Phase 4: Integration
1. Export best solution to CSV
2. Validate with shapely
3. Benchmark against baseline

### Expected Improvements

| Operation | Current (CPU) | Gen106 (GPU) | Speedup |
|-----------|---------------|--------------|---------|
| Overlap check (n=200) | ~1ms | ~0.01ms | 100x |
| Score config | ~0.5ms | ~0.01ms | 50x |
| Full SA (28k iter) | ~3 min | ~30s | 6x |
| Best-of-20 | ~60 min | ~10 min | 6x |

With 6x speedup, we could run best-of-100+ in the same time as current best-of-20.

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| MPS not supporting all ops | Fall back to CPU for unsupported ops |
| Memory limits | Batch processing, limit population size |
| Polygon overlap (not just bbox) | Keep CPU validation for candidates |
| May not close 24% gap | At least faster iteration for experiments |

### Success Criteria

1. **Minimum**: 2x faster evaluation → can run more best-of-N
2. **Target**: Match or beat 86.17 with 10x faster runtime
3. **Stretch**: Break 85.0 (1.4% improvement)

## Files to Create

```
santa-2025-packing/
├── python/
│   ├── gpu_primitives.py      # Core GPU operations
│   ├── population_opt.py      # Evolutionary optimization
│   ├── parallel_sa.py         # GPU-parallel SA
│   └── gen106_benchmark.py    # Benchmark runner
└── rust/
    └── src/bin/gen106_export.rs  # Export best to submission
```

## Continuation Prompt

```
Continue working on the Santa 2025 packing competition. Read GEN106_PLAN.md for full context.

Gen106: GPU-Accelerated Population Optimization
- Use PyTorch MPS on M2 Pro (19 GPU cores)
- Implement batched overlap detection and scoring
- Run multiple SA chains in parallel
- Evolutionary operators: selection, mutation, crossover

Current status:
- Plan documented in GEN106_PLAN.md
- Need to implement Phase 1: GPU primitives

Start by creating python/gpu_primitives.py with:
1. Tree representation as PyTorch tensors
2. gpu_transform_trees() - batch rotate/translate
3. gpu_compute_bbox() - batch bounding boxes
4. gpu_check_overlaps() - batched pairwise overlap
5. gpu_score_configs() - batch side_length computation

Test on MPS device with small n (n=10-20) before scaling up.
```
