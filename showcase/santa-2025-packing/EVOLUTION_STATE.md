# Evolution State - Gen87 Complete

## Current Champion
- **Gen87d** (greedy backtracking wave)
- **Score: 88.67** (lower is better)
- Location: `mutations/gen87d_greedy_backtrack.rs`
- Previous champion Gen84c scored 89.27

## Gen87 Results Summary
| Candidate | Score | Result |
|-----------|-------|--------|
| Gen84c (prev champion) | 89.27 | Baseline |
| Gen87a (rotation wave) | 90.45 | REJECTED |
| Gen87b (density order) | 89.39 | REJECTED |
| Gen87c (axis compress) | 100.89 | REJECTED |
| **Gen87d (greedy backtrack)** | **88.67** | **NEW CHAMPION** |

## Key Innovation in Gen87d
- Adds post-wave greedy pass targeting boundary-defining trees
- Identifies trees that define bounding box edges (left/right/top/bottom)
- Tries aggressive inward moves with rotation fallback
- 3 greedy passes after standard wave compaction

## Failed Approaches (Gen85-87)
- More waves (6 or 7) - worse
- Larger/adaptive step sizes - worse
- Stronger center pull - worse
- Phase-specific ordering - worse
- 8-directional diagonal - worse
- Rotation during wave compaction - worse
- Density-based ordering - neutral
- Axis-aligned only (no diagonal) - much worse

## Target
- Leaderboard top: ~69
- Current gap: 28.5%

## Next Steps for Gen88
Consider:
1. Combine Gen87d greedy backtrack with density ordering from 87b
2. Try different greedy pass counts (currently 3)
3. Target specific tree shapes during greedy pass
4. Try greedy pass BEFORE wave compaction as well as after
5. Adaptive greedy based on remaining gap to optimal

## File Locations
- Champion code: `rust/src/evolved.rs` (currently Gen84c, needs update to Gen87d)
- Mutations: `mutations/gen87*.rs`
- Benchmark: `cargo build --release --bin benchmark && ./target/release/benchmark 200 3`
