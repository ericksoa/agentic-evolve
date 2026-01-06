# Gen109 Plan: Multi-Strategy Optimization with GPU Acceleration

## Current Status
- **Score**: 86.17 (Rust Gen91b + best-of-20)
- **Gap to #1** (~69): 24%
- **Gen108 Finding**: Python SA adds only 0-0.04% to Rust; value is in variance exploitation

## Three Main Approaches

### Approach A: Rust SA Improvements
Enhance Rust's simulated annealing with moves from 70.1 solution.

**A1. Squeeze Move**
```rust
// Shrink all trees toward center by factor
fn sa_squeeze_move(trees: &mut Vec<PlacedTree>, factor: f64) -> bool {
    let (cx, cy) = compute_center(trees);
    let old_trees = trees.clone();

    for tree in trees.iter_mut() {
        tree.x = tree.x * factor + cx * (1.0 - factor);
        tree.y = tree.y * factor + cy * (1.0 - factor);
        tree.vertices = compute_vertices(tree.x, tree.y, tree.angle_deg);
    }

    // Validate no overlaps created
    if has_any_overlap(trees) {
        *trees = old_trees;
        return false;
    }
    true
}
```

**A2. Combined Position+Rotation Move**
```rust
// 20% of SA moves: adjust both position and rotation together
fn sa_combined_move(tree: &mut PlacedTree, temp: f64, rng: &mut impl Rng) -> bool {
    let scale = (temp / INITIAL_TEMP).max(0.1);
    let old_tree = tree.clone();

    tree.x += rng.gen_range(-0.08..0.08) * scale;
    tree.y += rng.gen_range(-0.08..0.08) * scale;
    tree.angle_deg = (tree.angle_deg + rng.gen_range(-30.0..30.0) * scale) % 360.0;
    tree.vertices = compute_vertices(tree.x, tree.y, tree.angle_deg);

    // Return success (caller checks overlap)
    true
}
```

**A3. Extended SA Parameters**
```rust
// Current: 28k iterations, 2 passes
// Proposed: 40k iterations, 3 passes, with squeeze phases
const SA_ITERATIONS: usize = 40_000;
const SA_PASSES: usize = 3;
const SQUEEZE_INTERVAL: usize = 5000;  // Try squeeze every 5k iters
```

**Expected Impact**: +0.5-1% improvement

---

### Approach B: ILP for Small n (n ≤ 10)

Small n contributes disproportionately to score: s²/1 = s², s²/2, s²/3...
Optimal solutions for n=1..10 could give significant score improvement.

**B1. Discretized ILP Formulation**
```python
from ortools.sat.python import cp_model

def ilp_pack(n: int, grid_resolution: int = 200, angle_steps: int = 8):
    """
    ILP formulation for tree packing.

    Variables:
    - x[i], y[i]: discretized position (0 to grid_resolution)
    - r[i]: rotation index (0 to angle_steps-1)
    - s: side length (minimize)

    Constraints:
    - Trees within [0, s] x [0, s]
    - No polygon overlaps (via precomputed NFP lookup)
    """
    model = cp_model.CpModel()

    # Variables
    x = [model.NewIntVar(0, grid_resolution, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, grid_resolution, f'y_{i}') for i in range(n)]
    r = [model.NewIntVar(0, angle_steps - 1, f'r_{i}') for i in range(n)]
    s = model.NewIntVar(1, grid_resolution, 'side')

    # Containment constraints
    for i in range(n):
        # Tree bounds must be within [0, s]
        # (requires precomputing bounds for each rotation)
        ...

    # Non-overlap constraints via NFP
    for i in range(n):
        for j in range(i + 1, n):
            # Add NFP constraints for each rotation pair
            ...

    model.Minimize(s)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)
```

**B2. GPU-Accelerated NFP Precomputation**
```python
def precompute_nfp_gpu(angle_steps: int = 8) -> torch.Tensor:
    """
    Precompute No-Fit Polygon for all rotation pairs on GPU.

    NFP[r1, r2] defines forbidden relative positions for tree2
    when tree1 is at origin with rotation r1 and tree2 has rotation r2.
    """
    # Use GPU Minkowski sum approximation
    # Returns (angle_steps, angle_steps, resolution, resolution) tensor
    ...
```

**Expected Impact**: +2-5% for n≤10 (high variance - depends on finding optimal)

---

### Approach C: Scaled Best-of-N with GPU Batch Evaluation

**C1. GPU Batch Collision Checking**
```python
def gpu_batch_validate(configs_batch: torch.Tensor) -> torch.Tensor:
    """
    Validate multiple configurations in parallel on GPU.

    Args:
        configs_batch: (batch_size, n_trees, 3) tensor

    Returns:
        valid_mask: (batch_size,) boolean tensor
    """
    # Transform all trees for all configs
    transformed = gpu_transform_trees(base_vertices, configs_batch)  # (B, N, 15, 2)

    # Compute pairwise bbox overlaps
    bbox = gpu_compute_bbox(transformed)  # (B, N, 4)
    bbox_overlaps = gpu_check_bbox_overlaps(bbox)  # (B, N, N)

    # For configs with bbox overlaps, do precise polygon check
    # (This is the bottleneck - can parallelize across configs)
    ...
```

**C2. Parallel Rust Generation + GPU Selection**
```
Pipeline:
1. Rust: Generate K packings in parallel (K = num_cores * 2)
2. Python/GPU: Batch evaluate all K packings
3. Select best valid packing
4. Repeat M times, keep overall best
```

**C3. Adaptive Best-of-N**
```python
def adaptive_best_of_n(n: int, target_quality: float = None):
    """
    Keep generating until quality target met or time budget exhausted.

    For small n: run more attempts (higher variance, optimal matters)
    For large n: fewer attempts (lower variance, diminishing returns)
    """
    base_attempts = {
        1: 50, 2: 40, 3: 35, 4: 30, 5: 25,
        10: 20, 20: 15, 50: 10, 100: 8, 200: 5
    }
    ...
```

**Expected Impact**: +1-3% improvement (more attempts = better luck)

---

## Combined Strategies

### Strategy 1: Rust Improvements + Best-of-N (Low Risk, Medium Reward)
```
1. Implement A1, A2, A3 in Rust evolved.rs
2. Run best-of-20 with improved algorithm
3. Expected: 1-2% total improvement
```

### Strategy 2: ILP Small + Rust Large (Medium Risk, High Reward)
```
1. ILP solver for n=1..10 (find optimal/near-optimal)
2. Rust evolved for n=11..200
3. Merge best solutions
4. Expected: 2-4% total improvement (if ILP finds better small n)
```

### Strategy 3: GPU-Accelerated Pipeline (Medium Risk, Medium Reward)
```
1. GPU batch collision for faster validation
2. Generate more candidates per time budget
3. Better best-of-N selection
4. Expected: 1-2% improvement (faster = more attempts)
```

### Strategy 4: Full Combination (High Effort, Highest Reward)
```
Phase 1: Rust SA Improvements
- Implement squeeze, combined moves
- Test on n=20, 50, 100

Phase 2: ILP for Small n
- Implement OR-Tools solver
- Precompute NFP (GPU accelerated)
- Solve n=1..10 with 60s timeout each

Phase 3: GPU Batch Pipeline
- Implement batch validation
- Parallel Rust generation
- Adaptive attempts per n

Phase 4: Integration
- ILP results for n=1..10
- Improved Rust + best-of-N for n=11..200
- Final submission
```

---

## Implementation Priority

| Task | Effort | Expected Gain | Priority |
|------|--------|---------------|----------|
| A1: Squeeze move | Low | +0.3% | 1 |
| A2: Combined move | Low | +0.2% | 2 |
| A3: Extended SA | Low | +0.3% | 3 |
| C2: Parallel Rust + select | Medium | +1% | 4 |
| B1: ILP solver | High | +2-4%? | 5 |
| C1: GPU batch validation | Medium | +0.5% | 6 |
| B2: GPU NFP precompute | High | enables B1 | 7 |

---

## File Structure

```
rust/src/
  evolved.rs              # Add squeeze, combined moves
  bin/
    export_packing.rs     # JSON export (done)
    parallel_generate.rs  # New: parallel generation

python/
  rust_hybrid.py          # Load/refine (done)
  parallel_refine.py      # Parallel pipeline (done)
  gen109_runner.py        # New: main orchestrator
  ilp_solver.py           # New: OR-Tools ILP
  nfp_compute.py          # New: NFP computation (GPU)
  gpu_batch_validate.py   # New: batch validation
```

---

## Success Metrics

- [ ] Rust squeeze move implemented and tested
- [ ] Rust combined move implemented and tested
- [ ] Best-of-N with improved Rust shows +1% over baseline
- [ ] ILP solver working for n≤5
- [ ] ILP finds better solution than Rust for at least one n≤10
- [ ] GPU batch validation faster than sequential
- [ ] Full pipeline score < 84 (2.5% improvement)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| ILP doesn't find better solutions | Set time limit, use Rust as fallback |
| GPU batch validation slower due to Python overhead | Benchmark early, skip if not faster |
| Rust changes break existing quality | Keep Gen91b as backup |
| Time budget exceeded | Prioritize by expected gain |

---

## Hardware Considerations

- **Apple M2 Pro**: 10 CPU cores, 19 GPU cores, 32GB RAM
- **Rust**: ~3000 SA iter/sec per core
- **Python/GPU**: ~500 collision checks/sec
- **ILP**: OR-Tools CP-SAT, ~1M decisions/sec

GPU is best used for:
1. Batch collision checking (many configs at once)
2. NFP precomputation (embarrassingly parallel)
3. NOT for individual SA moves (Python overhead dominates)

---

## Timeline Estimate

- Phase 1 (Rust improvements): 1-2 hours
- Phase 2 (ILP solver): 2-3 hours
- Phase 3 (GPU batch): 1-2 hours
- Phase 4 (Integration): 1 hour
- Full submission generation: 2-4 hours

Total: ~8-12 hours of work

---

## Continuation Prompt

See GEN109_CONTINUATION_PROMPT.md for the prompt to use after context clear.
