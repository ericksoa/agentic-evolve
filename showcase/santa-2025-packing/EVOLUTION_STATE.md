# Evolution State - Gen104 Complete (ML Ranking Failed)

## Continuation Prompt

```
Continue working on the Santa 2025 packing competition. Read EVOLUTION_STATE.md for full context.

Current status:
- Submission 86.17 accepted, passed validation
- Top leaderboard: ~69 (24% better)
- ML selection PROVEN to not help (post-hoc selection by min(side_length) is optimal)
- Pure best-of-N is already optimal for selection - need better algorithm

Gen104 key insight:
- Post-hoc selection cannot beat direct side_length comparison
- ML can only help DURING search (guiding placement), not after
- Closing the 24% gap requires fundamental algorithm improvement

Possible approaches to close the gap:
1. ILP solver for small n (Gurobi/CPLEX)
2. Simultaneous placement (not greedy incremental)
3. Tree-specific geometric insights (interlocking patterns)
4. Study winning solutions after competition ends
5. ML to guide search (not selection)

Current best submission:
- Score: 86.17 (Gen103 + safety margin + best-of-20)
- Generator: rust/src/bin/final_submission.rs
```

---

## Gen104 Results - Pairwise ML Ranking (Did Not Help)

### Approach
Created pairwise ranking model to select best packing from N candidates:
- Siamese network comparing two packings
- Trained on 5,498 pairs with 80.87% validation accuracy
- Round-robin comparison to rank candidates

### Benchmark Results (n=1-30, 10 runs)

| Selection Method | Score | Accuracy |
|-----------------|-------|----------|
| **Pure best-of-N** | **78.74** | N/A |
| ML ranking | 79.77 | 0% wins |

**Result: ML performs worse than simple min(side_length)**

### Root Cause Analysis

1. **Training accuracy**: 96.8% on training data
2. **Test accuracy**: ~50% on new candidates (random chance)
3. **Diagnosis**: Classic overfitting

The model memorized specific feature patterns from training runs rather than
learning what makes a packing "good". When given new packings from different
runs, it performs at random chance.

### Why Overfitting Occurred

- Features are raw positions (x, y) normalized by /10.0
- Different runs produce different absolute positions even for similar quality
- Model learned to match specific position patterns, not quality indicators
- Validation set was from same distribution as training (same runs)

### Files Added
- `ml/rerank_with_model.py` - Re-ranking script using pairwise model
- `rust/src/bin/generate_candidates.rs` - Generate candidates for ML testing

### Key Insight: Post-Hoc Selection is Optimal

For selecting from completed packings, **min(side_length) is provably optimal**:
- We want to select the packing with smallest side_length
- The selection criterion IS the optimization objective
- Therefore: no method can beat direct comparison of side_length

Where ML *could* help (not yet tried):
1. **During search**: Predict which partial packing leads to good final result
2. **Search guidance**: Prioritize which positions/rotations to try
3. **Early termination**: Predict if search is unlikely to beat current best

### Submission Status
- **Submission accepted**: 86.174381 (passed validation)
- **Top leaderboard**: ~69
- **Gap**: ~24% - requires fundamental algorithm improvement, not better selection

---

# Evolution State - Gen103 Complete (Overlap Fix Applied)

## Current Status
- **Champion: Gen91b + Best-of-20 selection + Safety Margin**
- **Score: ~87.6** (5-run with safety margin) / ~86 expected (20-run)
- **Target: ~70** (top leaderboard)
- **Gap: ~24%**
- **FIXED: Overlap validation issue resolved**

---

## Gen103 Overlap Fix (Completed)

### Problem
Kaggle rejected submissions with "Overlapping trees in group 104". Our local `has_overlaps()` passed but Kaggle's validation (using shapely) failed.

### Root Cause
- Floating-point precision loss when exporting to CSV (6 decimal places)
- Kaggle uses shapely for polygon intersection which detects tiny overlaps (~1e-12 area)
- Trees placed very close together create overlap artifacts after rounding

### Solution Applied
1. **Safety margin in overlap detection** (`lib.rs`):
   - Added `SAFETY_MARGIN = 1e-5` in `polygons_overlap()`
   - Trees must now be at least 1e-5 apart, not just non-overlapping
   - Uses `point_near_segment()` to check vertex-to-edge distances

2. **Increased CSV precision** (`final_submission.rs`):
   - Changed from 6 to 9 decimal places in CSV export
   - Format: `s{:.9}` instead of `s{:.6}`

3. **Created validation script** (`validate_submission.py`):
   - Uses shapely to match Kaggle's validation
   - Run before submitting: `python3 validate_submission.py rust/submission.csv`

### Results
- 5-run test submission: **VALID** (shapely validation passes)
- 20-run final submission: **VALID** (shapely validation passes)
- **Final Score: 86.1744** (20 runs with safety margin)
- Time to generate: 70.5 minutes
- Submitted to Kaggle: 2026-01-04

---

## Gen104 ML Improvements (In Progress)

### Pairwise Ranking Model (Completed)

The Gen102 ML model predicted well (MAE=0.04) but failed at selection (-0.98%). Root cause: trained for absolute prediction, but needed relative ranking.

**Solution: Pairwise ranking model**
- Train on pairs: (packing_A, packing_B) → "is A better?"
- Loss: binary cross-entropy on comparison
- Directly optimizes for the selection task

**Training Results:**
- Generated 5,498 training pairs from existing data
- Model: Siamese network with shared encoder (107K params)
- **Best validation accuracy: 80.87%** (vs 50% baseline)
- Training time: 7 seconds on MPS

**Files Added:**
- `ml/pairwise_model.py` - Siamese ranking network
- `ml/generate_pairs.py` - Pair generation script
- `ml/train_pairwise.py` - Training script
- `ml/pairwise_data.jsonl` - 5,498 training pairs
- `ml/ranking_model.pt` - Trained model

### TODO: Use Ranking Model for Selection

Next steps to improve score:
1. Create re-ranking script that uses pairwise model to select best from N candidates
2. Benchmark against pure best-of-N selection
3. If successful, integrate into submission pipeline

### Future Improvements

- **Hard pairs training**: Compare top 10% vs top 30% (not worst vs best)
- **Better features**: Min pairwise distance, boundary utilization, density variance
- **Guide SA search**: Use ML as value function during SA, not just selection

---

## Gen103 Results Summary - Best-of-N Optimization

### Approach Comparison

| Approach | Score | Improvement | Notes |
|----------|-------|-------------|-------|
| Single run | ~89 | baseline | |
| Best-of-5 | 86.55 | +3.52% | |
| **Best-of-20** | **85.89** | **+3.87%** | Best observed |
| Best-of-30 | 86.04 | +3.68% | |
| Stochastic-20 | 86.09 | +2.86% | Random param variation |
| Multi-strategy | 86.80 | +2.06% | 8 different configs |
| Ultimate combo | 86.38 | +3.42% | All approaches |

### Key Finding

Simple best-of-N with default config outperforms:
- Parameter variation (stochastic)
- Different strategy configurations
- Combined approaches

The evolved algorithm's default parameters are already well-tuned. Running multiple times and selecting best per N is the most effective improvement.

### Files Added
- `rust/src/bin/stochastic_best.rs` - Parameter variation
- `rust/src/bin/multi_strategy.rs` - Multi-config strategies
- `rust/src/bin/ultimate_submission.rs` - Combined approach
- `rust/src/bin/final_submission.rs` - Best-of-N submission generator

---

## Gen102 Results Summary - ML Value Function + Best-of-N

### ML Approach (Did Not Help)

Attempted to train a neural network to predict final side length from partial packing state:
- **Model**: MLP with 66K parameters
- **Training**: 6000 samples, 50 epochs, 7.8s on M2 MPS
- **Validation MAE**: 0.04 (quite accurate)

**Re-ranking results**: -0.98% (worse than baseline)
- Model predictions don't correlate well with actual best solutions
- Variance in predictions doesn't match variance in quality

### Best-of-N Selection (Works Well!)

Simple approach: run evolved N times, pick best for each n.

| Runs | Score (n=1-200) | Improvement | Time |
|------|-----------------|-------------|------|
| 1 | 89.71 | baseline | 3 min |
| 5 | **86.55** | **+3.52%** | 12 min |
| 10 (n=1-50) | 24.23 | +5.09% | 3 min |

**Key insight**: Evolved algorithm has stochastic elements that produce different quality solutions. Multiple runs with selection exploits this variance.

### Files Added
- `ml/model.py` - PyTorch value function models
- `ml/train.py` - Training pipeline with MPS support
- `ml/beam_search.py` - ML-guided beam search (too slow)
- `ml/rerank_strategies.py` - ML re-ranking (didn't help)
- `rust/src/bin/best_of_n.rs` - Best-of-N selection (works!)
- `rust/src/bin/generate_training_data.rs` - Training data generator

---

## Competition Status
- Competition ends: January 30, 2026
- Current #1: Rafbill - 69.99
- 2,412 teams participating

## Gen101 Results Summary - Combined Multi-Strategy Approach

Generation 101 tested **combinations of all previous approaches**, based on the insight that "best ideas often are combinations."

### Combined Strategy Architecture

```
For n <= 20:
  Strategy A: Diamond init → Sparrow explore → Wave compact → Local search
  Strategy B: Hexagonal init → Sparrow explore → Wave compact → Local search
  Strategy C: Evolved base → Extra local refinement (3x iterations)
  → Pick best result per N

For n > 20:
  Strategy C only (evolved + local refinement)
  → Pattern strategies don't scale well to large N
```

### Full Benchmark Results (n=1-200)

| Approach | Score | Time | Wins |
|----------|-------|------|------|
| Combined | 89.59 | 85 min | 76/200 |
| Evolved | 89.93 | 3 min | 123/200 |
| **Improvement** | **0.38%** | | |

### N-range Analysis

| Range | Combined Wins | Notes |
|-------|--------------|-------|
| n=1-20 | ~17/20 (85%) | Pattern strategies help |
| n=21-50 | ~15/30 (50%) | Mixed results |
| n=51-200 | ~44/150 (29%) | Evolved usually better |

### Key Findings

1. **Pattern-based init helps for small N**: Diamond/hexagonal patterns with Sparrow exploration beat evolved for most n <= 20

2. **Evolved is already optimal for large N**: The greedy + SA approach is hard to improve upon for n > 50

3. **Marginal gains require significant compute**: 0.38% improvement costs 30x more time

### Files Added
- `rust/src/combined.rs` - Multi-strategy combiner
- `rust/src/bin/benchmark_combined.rs` - Combined benchmark
- `rust/src/bin/submit_combined.rs` - Submission generator

---

## Gen100 Results Summary - Sparrow Algorithm

Generation 100 implemented the **Sparrow algorithm** from recent research (arxiv.org/html/2509.13329) - a state-of-the-art approach for 2D nesting problems.

### Sparrow Key Ideas
1. **Temporary overlap tolerance** - Allow collisions, use penetration depth as continuous metric
2. **Guided Local Search** - Dynamic weights on persistently colliding pairs
3. **Two-phase architecture** - Exploration (aggressive) then Compression (refinement)

### Results (n=1-20)

| Approach | Score | vs Champion | Notes |
|----------|-------|-------------|-------|
| Evolved (champion) | 10.88 | Baseline | Greedy + SA |
| Sparrow | 12.33 | +13% worse | Pure Sparrow approach |
| Hybrid | 11.20 | +3% worse | Evolved + Sparrow refinement |

### Why Sparrow Didn't Help

The Sparrow algorithm is designed for **strip packing** (infinite length, fixed width), not **square box minimization**. Our evolved algorithm already effectively:
- Navigates the "desert of infeasibility"
- Uses simulated annealing for escape from local optima
- Applies intensive local search via wave compaction

### Files Added
- `rust/src/sparrow.rs` - Sparrow-inspired algorithm
- `rust/src/hybrid.rs` - Evolved + Sparrow refinement

---

## Gen99 Results Summary - ILP/Optimization Analysis

Generation 99 explored **non-greedy optimization approaches** including ILP, global optimization, and pattern-based packing. **All experimental approaches performed worse than the champion.**

### Key Findings from Geometry Analysis

1. **Tree Properties**:
   - Area: 0.2456 (only 35% of bounding box)
   - Height: 1.0, Width: 0.7
   - 45° rotation gives smallest bbox: 0.813 x 0.813

2. **Packing Efficiency Analysis**:
   - Current efficiency: ~56% (area used / box area)
   - Target efficiency: ~70% (what top solutions achieve)
   - Gap: ~25% improvement needed in area efficiency

3. **Side Length Scaling**:
   - If side = k * sqrt(n): current k ≈ 0.64, target k ≈ 0.59
   - Top solutions are ~10% tighter per dimension
   - This translates to ~20% better score

### Experimental Approaches Tested

| Approach | Score (n=1-20) | vs Champion | Notes |
|----------|---------------|-------------|-------|
| NFP-guided greedy (Python) | 5.54 (n=1-8) | ~Similar | Scipy optimization, NFP constraints |
| Global optimizer (Rust) | 9.19 (n=1-15) | Worse | Differential evolution, population-based |
| Pattern-based (Rust) | 15.71 (n=1-20) | Much worse | Herringbone, diamond, spiral patterns |
| Champion (evolved) | ~12.0 (n=1-20) | Baseline | Greedy + SA |

### Why Alternative Approaches Failed

1. **Global optimization**: Too many local minima, slow convergence
2. **Pattern-based**: Fixed patterns don't adapt to n, suboptimal interlocking
3. **ILP/constraint**: Computationally intractable for n>10 with non-convex polygons

### Files Created

- `python/nfp_optimizer.py` - NFP-based optimization
- `python/analyze_tree_geometry.py` - Geometry analysis
- `rust/src/global_opt.rs` - Differential evolution optimizer
- `rust/src/pattern_based.rs` - Pattern-based packing

### What Would Actually Help (Hypothesis)

1. **Commercial ILP solvers** (Gurobi, CPLEX) for small N
2. **Machine learning** trained on good packings
3. **Study winning solutions** after competition ends
4. **Problem-specific insights** we're missing about tree interlocking

## Gen98 Results Summary

Generation 98 tried **multiple optimization approaches**. All mutations were rejected - none beat the champion.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen98 (relocation) | - | Remove boundary tree, re-place elsewhere | REJECTED - Too slow |
| Gen98b (16 rotations) | 96.85 | 22.5° rotation increments instead of 45° | REJECTED - Much worse |
| Gen98c (finer search) | 88.52 | 300 attempts, 0.0005 precision | REJECTED - Similar, 33% slower |
| Gen98d (5x SA iters) | 88.72 | 140k iterations instead of 28k | REJECTED - Similar, 2x slower |

## Gen98 Key Learnings

1. **Relocation moves are too expensive**: Removing a boundary tree and re-placing it requires full placement search, making SA too slow.

2. **Non-45° rotations hurt performance**: The tree shape has natural symmetry at 45° increments. Finer angles (22.5°) create suboptimal interlocking patterns.

3. **More search doesn't help**: Increasing search attempts (200→300) and precision (0.001→0.0005) gives marginal improvement at 33% time cost.

4. **More SA iterations don't help**: The SA is already well-tuned. 5x more iterations (28k→140k) with adjusted cooling doesn't improve results.

## Cumulative Plateau Analysis (Gen92-98)

After **seven full generations** of failed attempts, we've confirmed a fundamental plateau:

**Gen92 (Parameter Tuning) - All Failed**
**Gen93 (Algorithmic Changes) - All Failed**
**Gen94 (Paradigm Shifts within Greedy) - All Failed**
**Gen95 (Global Optimization) - All Failed**
**Gen96 (Paradigm Shifts) - All Failed**
**Gen97 (Winning Solution Techniques) - All Failed**
**Gen98 (Optimization Intensification) - All Failed**

## What's Working (Gen91b Champion)
- Exhaustive 8-rotation search at each position (45° increments)
- 5-pass wave compaction with bidirectional order
- Greedy backtracking for boundary trees
- Multi-strategy evaluation with cross-pollination (6 strategies)
- Well-tuned SA parameters (temp 0.45, cooling 0.99993, 28k iters)

## What Doesn't Help (Exhaustive List)
- More rotations (16 vs 8)
- Non-45° rotation angles
- More search attempts
- Finer binary search precision
- More SA iterations
- Slower SA cooling
- Relocation moves
- Continuous angle optimization
- Separation-based packing
- Global SA on complete configuration
- Re-centering and compression
- Different scoring functions
- Multi-start optimization
- Genetic algorithms
- Hexagonal grid patterns

## Gap to Target (20-26%)

The persistent gap to leaderboard (~70) confirms top solutions use **fundamentally different approaches**:

1. **Non-greedy global optimization**: ILP, constraint satisfaction, branch-and-bound
2. **Simultaneous placement**: Place all trees at once, not incrementally
3. **Problem-specific insights**: The tree shape may have exploitable geometric properties
4. **Learning-based methods**: Neural networks trained on good packings

## File Locations
- Champion code: `rust/src/evolved.rs` (Gen91b)
- Champion backup: `rust/src/evolved_champion.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`

## Recommendation

The **greedy incremental approach has reached its fundamental limit**. Seven generations of mutations have failed to improve on Gen91b. Further progress requires:

1. **ILP formulation** with commercial solvers (Gurobi, CPLEX)
2. **Complete algorithm redesign** (non-incremental placement)
3. **Wait for competition end** to study winning solutions

The evolution has plateaued at 20-26% gap to target.
