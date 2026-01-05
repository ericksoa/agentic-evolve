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
