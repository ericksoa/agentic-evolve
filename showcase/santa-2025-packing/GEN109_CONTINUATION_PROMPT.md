# Gen109 Continuation Prompt

Copy everything below this line after clearing context:

---

Continue working on the Santa 2025 packing competition. Read GEN109_PLAN.md for the full plan.

## Quick Context

**Competition**: Pack n Christmas tree polygons into smallest square (n=1 to 200)
**Score**: Sum of (side_length² / n) for all n
**Current best**: 86.17 (Rust Gen91b + best-of-20)
**Target**: <84 (2.5% improvement), ideally closer to #1 (~69)

## Gen108 Findings

- Python SA adds only 0-0.04% to Rust solutions (Rust is already near-optimal)
- Variance exploitation (best-of-N) gives 4% improvement
- Created hybrid pipeline (works but Python SA not valuable)

## Gen109 Strategy

Implement all approaches in priority order:

### Phase 1: Rust SA Improvements (Priority 1-3)
File: `rust/src/evolved.rs`

1. **Squeeze move** - Shrink all trees toward center (like 70.1 solution)
2. **Combined move** - Position + rotation together (20% of moves)
3. **Extended SA** - More iterations (40k), squeeze phases every 5k iters

Test with: `cargo build --release && ./target/release/best_of_n 50 5`

### Phase 2: ILP for Small n (Priority 5)
File: `python/ilp_solver.py` (create new)

- Use OR-Tools CP-SAT for n=1..10
- Discretize positions (200x200 grid) and rotations (8 angles)
- Precompute NFP (No-Fit Polygon) for overlap constraints
- 60 second timeout per n

### Phase 3: GPU Batch Validation (Priority 6)
File: `python/gpu_batch_validate.py` (create new)

- Batch collision checking for multiple configs
- Use for faster best-of-N selection
- Leverage existing `gpu_primitives.py` and `hybrid_collision.py`

### Phase 4: Integration
File: `python/gen109_runner.py` (create new)

- ILP results for n=1..10
- Improved Rust + best-of-N for n=11..200
- Generate final submission

## Key Files

```
rust/src/evolved.rs           # Main algorithm - ADD squeeze, combined moves
rust/src/lib.rs               # PlacedTree, overlap checking
rust/src/bin/export_packing.rs # JSON export (done in Gen108)
rust/src/bin/best_of_n.rs     # Best-of-N selection

python/hybrid_collision.py    # GPU+CPU collision (existing)
python/gpu_primitives.py      # GPU tree transforms (existing)
python/rust_hybrid.py         # Load Rust packings (Gen108)
python/ilp_solver.py          # ILP solver (TO CREATE)
python/gen109_runner.py       # Main runner (TO CREATE)
```

## Commands

```bash
# Build Rust
cd showcase/santa-2025-packing/rust
cargo build --release

# Test Rust improvements
./target/release/best_of_n 50 10

# Run Python
cd ../python
python3 gen109_runner.py --benchmark
python3 gen109_runner.py --ilp-only --max-n 10
python3 gen109_runner.py --full --output submission_gen109.csv
```

## Implementation Notes

1. **Rust squeeze move**: Must check for overlaps after squeeze (use has_any_overlap)
2. **ILP NFP**: Precompute for all 8x8=64 rotation pairs
3. **GPU batch**: Main benefit is evaluating many Rust outputs quickly
4. **Score formula**: Lower is better: sum(side² / n) for n=1..200

## Success Criteria

- [ ] Rust squeeze move: +0.3% improvement
- [ ] Rust combined move: +0.2% improvement
- [ ] ILP beats Rust for at least one n≤10
- [ ] Total score < 84 (2.5% better than 86.17)

Start with Phase 1 (Rust improvements) as it has best effort-to-reward ratio.
