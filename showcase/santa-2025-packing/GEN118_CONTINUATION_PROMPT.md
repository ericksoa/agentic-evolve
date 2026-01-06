# Gen118 Continuation Prompt

We're continuing the Santa 2025 packing optimization. Current state:

- **Score**: 85.45 (unchanged since Gen115)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

## What's Been Tried (All Failed)

### Gen116
- CMA-ES on medium n (11-30) with 10 restarts, 20k evals
- Found "improvements" that failed strict segment-intersection validation
- Lesson: Shapely area-based checks miss edge crossings

### Gen117
- Pattern-based initialization (radial, hexagonal, spiral, grid)
- Local CMA-ES refinement with small sigma (0.02-0.1)
- Boundary-focused SA on large n (50-200)
- All produced same or worse results than current

### Previous Generations (92-115)
- Parameter tuning for SA, CMA-ES
- ML-based selection
- Global rotation optimization
- Various hybrid approaches

## Why Current Approach is Near-Optimal

The Rust solver (Gen91b) uses:
1. 6 parallel placement strategies
2. All 8 rotations tested at each position
3. 28,000 SA iterations with elite pool and hot restarts
4. 5-pass wave compaction
5. Best-of-20 selection

This is already extremely well-tuned.

## New Directions to Explore

### 1. Continuous Angles
Current: Only 8 discrete angles (0, 45, 90, ...)
Try: Continuous angle optimization within SA moves
Rationale: May find tighter packings with non-standard angles

### 2. Different Overlap Detection
Current: Polygon intersection with Shapely
Try: Use the Rust overlap checking which may have different tolerances
Rationale: Ensure we're using exact same validation as final checker

### 3. Per-Group Optimal Algorithms
For n=1: Trivial (single tree)
For n=2: Analytical solution possible (optimal angle difference)
For n=3-5: Exhaustive search over discrete positions feasible
Rationale: Small groups contribute disproportionately to score

### 4. Analyze Leaderboard Gap
Score of ~69 is 24% better than our 85.45
Questions:
- Are top submissions using same tree polygon?
- Are they interpreting overlap constraints differently?
- Do they have access to test cases we don't?

### 5. Genetic Algorithm on Configurations
Instead of CMA-ES on parameters, evolve entire configurations:
- Crossover: Exchange trees between two packings
- Mutation: Small position/angle perturbations
- Selection: Keep best valid configurations

## Priority for Gen118

Start with **continuous angles** since this is the simplest conceptual change with potential for improvement. Modify the Rust SA to allow small angle perturbations (±5°) around the standard 45° increments.

## Commands

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Validate before submitting
python3 python/validate_submission.py submission_best.csv

# Run Rust solver with more iterations
cd rust && cargo build --release
./target/release/best_of_n 200 30  # best of 30 runs

# Submit to Kaggle
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions submit \
  -c santa-2025 -f submission_best.csv -m "Gen118: [description]"
```
