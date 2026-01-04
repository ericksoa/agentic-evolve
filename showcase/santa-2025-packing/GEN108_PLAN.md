# Gen108 Plan: Rust-Python Hybrid Pipeline

## Motivation

Gen107 validated that hybrid GPU/CPU collision works, but Python is 5-10x slower than Rust.
The key insight: **initial placement quality dominates final result**.

Rust greedy achieves ~3.2 side for n=20, while Python grid starts at 7.5.
Even with 40k SA iterations, Python only reaches 3.33 (vs Rust's 3.2).

## Strategy: Best of Both Worlds

Use Rust for what it's good at (fast greedy placement),
and Python for what it could add (different SA moves, GPU parallelism).

## Implementation Plan

### Phase 1: Rust JSON Export

Add JSON output to Rust evolved algorithm:

```rust
// rust/src/bin/export_packing.rs
fn export_packing_json(n: usize, output: &Path) {
    let mut packer = EvolvedPacker::new();
    let trees = packer.pack(n);

    let json = json!({
        "n": n,
        "trees": trees.iter().map(|t| {
            json!({
                "x": t.x,
                "y": t.y,
                "angle_deg": t.angle_deg
            })
        }).collect::<Vec<_>>(),
        "side_length": compute_side_length(&trees)
    });

    std::fs::write(output, serde_json::to_string_pretty(&json)?)?;
}
```

### Phase 2: Python Import & Refine

Load Rust packing and apply Python SA refinement:

```python
# python/rust_hybrid.py
def load_rust_packing(json_path: Path) -> np.ndarray:
    """Load packing from Rust JSON export."""
    with open(json_path) as f:
        data = json.load(f)

    configs = np.zeros((data['n'], 3), dtype=np.float64)
    for i, tree in enumerate(data['trees']):
        configs[i] = [tree['x'], tree['y'], tree['angle_deg']]

    return configs

def refine_with_sa(configs: np.ndarray, iterations: int = 10000) -> np.ndarray:
    """Apply Python SA refinement to Rust packing."""
    optimizer = MultiStageOptimizer()

    # Skip initial placement stages - Rust already did this
    config = PipelineConfig(
        rotation_enabled=False,
        compaction_enabled=False,
        squeeze_enabled=True,  # Light squeeze only
        sa_iterations=iterations,
        sa_restarts=3,
        polish_enabled=True,
    )

    return optimizer.optimize(configs, config)
```

### Phase 3: Parallel Refinement

Run multiple SA refinements in parallel:

```python
# python/parallel_refine.py
from concurrent.futures import ProcessPoolExecutor

def parallel_refine(n: int, n_workers: int = 6) -> np.ndarray:
    """Generate and refine packings in parallel."""

    # Generate N Rust packings
    rust_configs = []
    for i in range(n_workers):
        subprocess.run(['./rust/target/release/export_packing',
                       str(n), f'/tmp/packing_{i}.json'])
        rust_configs.append(load_rust_packing(f'/tmp/packing_{i}.json'))

    # Refine in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(refine_with_sa, cfg) for cfg in rust_configs]
        results = [f.result() for f in futures]

    # Return best
    return min(results, key=lambda r: r[1])  # (configs, side_length)
```

## Alternative: Improve Rust Directly

Based on Gen107 insights, try adding to Rust:

### 1. More Aggressive SA Moves

```rust
// Add squeeze move to Rust SA
fn sa_squeeze_move(trees: &mut Vec<PlacedTree>, factor: f64) {
    let (cx, cy) = compute_center(trees);
    for tree in trees.iter_mut() {
        tree.x = tree.x * factor + cx * (1.0 - factor);
        tree.y = tree.y * factor + cy * (1.0 - factor);
        tree.vertices = compute_vertices(tree.x, tree.y, tree.angle_deg);
    }
}
```

### 2. Combined Position+Rotation Moves

```rust
// 20% of moves: adjust both position and rotation together
fn sa_combined_move(tree: &mut PlacedTree, temp: f64) {
    let scale = (temp / INITIAL_TEMP).max(0.1);
    tree.x += rng.gen_range(-0.1..0.1) * scale;
    tree.y += rng.gen_range(-0.1..0.1) * scale;
    tree.angle_deg = (tree.angle_deg + rng.gen_range(-15.0..15.0) * scale) % 360.0;
    tree.vertices = compute_vertices(tree.x, tree.y, tree.angle_deg);
}
```

### 3. Longer SA with Restarts

```rust
// Current: 28k iterations, no restarts
// Try: 40k iterations, 6 restarts (like 70.1)
const SA_ITERATIONS: usize = 40_000;
const SA_RESTARTS: usize = 6;
```

## Experimental: ILP for Small N

For n ≤ 10, try integer linear programming:

```python
# python/ilp_solver.py
from ortools.sat.python import cp_model

def ilp_pack(n: int, grid_resolution: int = 100) -> np.ndarray:
    """
    ILP formulation for tree packing.

    Variables:
    - x[i], y[i]: discretized position of tree i
    - r[i]: rotation index (0-7 for 45° increments)
    - s: side length (minimize)

    Constraints:
    - All trees within [0, s] x [0, s]
    - No polygon overlaps (via precomputed NFP)
    """
    model = cp_model.CpModel()

    # ... ILP formulation

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    # Extract solution
    ...
```

## Expected Outcomes

| Approach | Expected Improvement | Confidence |
|----------|---------------------|------------|
| Rust export + Python SA | +1-2% | High |
| Rust SA improvements | +0.5-1% | Medium |
| Parallel refinement | +1-2% | High |
| ILP for small n | +2-5% for n≤10 | Low |

## Priority Order

1. **Rust export + Python refine** - Low effort, validates pipeline
2. **Rust SA improvements** - Medium effort, could help Rust directly
3. **Parallel refinement** - Scale up best-of-N approach
4. **ILP experimentation** - Only if time permits

## Success Criteria

- [ ] Rust JSON export working
- [ ] Python can load and refine Rust packings
- [ ] Refinement improves Rust baseline by >0.5%
- [ ] Parallel pipeline working for full n=1-200

## Files to Create

```
rust/src/bin/export_packing.rs    # JSON export
python/rust_hybrid.py             # Load and refine
python/parallel_refine.py         # Parallel processing
python/gen108_runner.py           # Main runner
```

## Hardware

- Apple M2 Pro (10 CPU cores, 19 GPU cores)
- 32 GB RAM
- Rust: ~3000 SA iter/sec
- Python: ~550 SA iter/sec
