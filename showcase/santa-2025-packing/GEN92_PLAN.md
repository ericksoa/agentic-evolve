# Gen92+ Evolution Plan: Refining Rotation-First

## Current State
- **Champion**: Gen91b (rotation-first optimization)
- **Best Score**: 87.29
- **Target**: ~69 (leaderboard top)
- **Gap**: ~26.5%

## What's Been Tried (Don't Repeat)

### Gen91 (Paradigm Shifts)
- Size-ordered placement: SKIPPED (all trees identical)
- **Rotation-first optimization: 87.29 - CHAMPION**
- BLF hybrid: 88.62 - REJECTED
- SA temp tuning (0.45→0.60): 87.64 - REJECTED
- SA iterations (28k→35k): 88.32 - REJECTED
- More search attempts (200→300): 87.99 - REJECTED

### Gen90 (Orthogonal Mutations)
- Wave phase reversal: 89.25 - REJECTED
- Boundary swap: 88.80 - REJECTED
- More SA iterations: 88.50 - REJECTED
- 3+2 wave split: 89.30 - REJECTED

## Gen92 Strategy: Refine Rotation-First

Since rotation-first improved results, explore variations:

### Phase 1: Finer Rotation Granularity
**Rationale**: Gen91b uses 8 rotations (0, 45, 90...). More angles might find better fits.

**Gen92a: 16 rotations** (every 22.5°)
- Try [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, ...]
- More positions to test, but potentially tighter packing

**Gen92b: 24 rotations** (every 15°)
- Even finer granularity
- May be overkill but worth testing

### Phase 2: Position Micro-Adjustment
**Rationale**: After finding best rotation, micro-adjust position.

**Gen92c: Post-rotation jitter**
- After finding best (position, rotation) pair
- Try small position adjustments (±0.01, ±0.02)
- Keep if improves bounding box

### Phase 3: Placement Scoring Refinement
**Rationale**: Current scoring may not optimally weight rotation benefits.

**Gen92d: Rotation-aware scoring**
- Add bonus for rotations that minimize extension
- Penalize rotations that create awkward gaps

### Phase 4: Adaptive Search Focus
**Rationale**: Spend more compute on difficult placements.

**Gen92e: N-scaled search**
- More search attempts for high-N trees (harder to place)
- Fewer for low-N (usually easy)

## Implementation Order

1. **Gen92a**: 16 rotations (simple change, quick test)
2. **Gen92b**: 24 rotations (if 16 helps, try more)
3. **Gen92c**: Position micro-adjustment
4. **Gen92d**: Rotation-aware scoring
5. **Gen92e**: N-scaled search attempts

## Benchmark Protocol

For each candidate:
1. Build: `cargo build --release`
2. Test: `./target/release/benchmark 200 3`
3. If best score < 87.29, run 5 more times to verify
4. If consistently better, update champion

## Key Code Section to Modify

In `find_placement_with_strategy()` at line ~430:
```rust
// GEN91b: All 8 standard rotations for exhaustive search
let all_rotations = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];
```

Change to 16 rotations:
```rust
// GEN92a: 16 rotations for finer granularity
let all_rotations = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
                     180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5];
```

## Success Criteria
- Score < 87.00: Minor improvement, continue refining
- Score < 86.00: Significant improvement, new champion
- Score < 85.00: Major breakthrough
- Score < 80.00: Exceptional progress
