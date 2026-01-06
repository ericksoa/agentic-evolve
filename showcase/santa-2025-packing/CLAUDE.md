# Santa 2025 Packing Challenge - Claude Workflow

## Pre-Submit Checklist

**ALWAYS run these checks before submitting to Kaggle:**

```bash
# 1. Validate submission format and check for overlaps
python3 python/validate_submission.py submission_best.csv

# 2. Only submit if validation passes
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions submit \
  -c santa-2025 \
  -f submission_best.csv \
  -m "Description of changes"
```

## Quick Validation Command

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing
python3 python/validate_submission.py submission_best.csv && echo "VALID - OK to submit" || echo "INVALID - FIX BEFORE SUBMITTING"
```

## Submission Format

The Kaggle submission requires:
- **id**: `{n:03d}_{tree_idx}` (e.g., `001_0`, `002_1`, `200_199`)
- **x, y, deg**: s-prefixed values (e.g., `s0.0`, `s90.0`)
- **Header**: `id,x,y,deg`

Example:
```
id,x,y,deg
001_0,s0.0,s0.0,s45.0
002_0,s0.1,s0.2,s90.0
002_1,s0.5,s-0.3,s135.0
```

## Common Issues

1. **"ID column id not found"**: Header must be `id,x,y,deg` (not `row_id`)
2. **"Solution and submission values for id do not match"**: Use `{n:03d}_{idx}` format
3. **"Overlapping trees in group NNN"**: Run validation before submit!

## Key Files

- `submission_best.csv` - Current best submission
- `python/validate_submission.py` - Pre-submit validation
- `python/create_hybrid_submission.py` - Merge Rust + Python solutions
- `python/analyze_submission.py` - Analyze score breakdown

## Generate New Submission

```bash
# Run Rust solver
cd rust && ./target/release/ultimate_submission 5 5 ../submission_rust.csv

# Create hybrid (merges best from Rust + Python)
python3 python/create_hybrid_submission.py \
  --rust-csv submission_rust.csv \
  --python-json python/optimized_small_n.json \
  --output submission_best.csv

# Validate before submitting
python3 python/validate_submission.py submission_best.csv
```

## Post-Improvement Workflow

**After finding improvements that update submission_best.csv:**

### 1. Update Visualizations
```bash
# Regenerate SVG visualizations for improved n values
./rust/target/release/visualize submission_best.csv

# View updated visualizations
open packing_n*.svg
```

### 2. Update README.md
The README needs updates when:
- Score improves (update leaderboard position and score)
- New generation (add to evolution journey table)
- New algorithm techniques (update "What Works" section)

Key sections to check:
- **Line 22**: Current best score (e.g., "our current best: **85.45** (Gen115)")
- **Line 55+**: Core algorithm description (should reflect current approach)
- **Evolution Journey table**: Add new generation milestone
- **Results Summary table**: Update final row with current score

### 3. Create GEN*_RESULTS.md
Document what was tried and what worked:
```markdown
# Gen116 Results

## Summary
- Starting score: 85.45
- Final score: XX.XX
- Improvement: X.XX points

## What Worked
- [List successful optimizations]

## What Didn't Work
- [List failed attempts with brief reason]

## Key Learnings
- [Technical insights for future generations]
```

### 4. Commit Checklist
Before committing:
```bash
# 1. Validate submission
python3 python/validate_submission.py submission_best.csv

# 2. Stage all related files
git add submission_best.csv README.md GEN*_RESULTS.md packing_n*.svg

# 3. Descriptive commit message
git commit -m "Gen116: [brief description] - score XX.XX -> YY.YY"
```

## Strict Validation Requirements

**CRITICAL**: The Kaggle validator uses strict segment-intersection overlap checking.

Always use `python/validate_submission.py` which matches Kaggle's checker:
- Segment intersection (not just area-based Shapely)
- Point-in-polygon for containment
- Zero tolerance for edge crossings

The CMA-ES optimizer should use `has_any_overlap_strict()` which combines:
1. Shapely fast filter (skip non-intersecting pairs)
2. Segment intersection for edge cases with zero-area overlaps
