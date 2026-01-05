# Gen111 Continuation Prompt

Continue working on the Santa 2025 packing competition. Read this file for context.

## Quick Context

- **Competition**: Pack n Christmas tree polygons into smallest square (n=1 to 200)
- **Score formula**: Sum of (side_length² / n) for all n. Lower is better.
- **Current best**: 85.77 (Gen110)
- **Leader (#1)**: ~69
- **Gap**: 24% - still significant room for improvement

## Gen110 Results

**Score: 86.38 → 85.77 (0.71% improvement)**

What worked:
- Exhaustive search for small n found improvements for n=2,3,5,6
- n=2: 1.12 → 0.99 (biggest win)
- n=3: 1.35 → 1.14

What didn't work:
- NFP-based greedy placement (slower and worse than Rust)
- CMA-ES post-optimization (couldn't improve Rust SA solutions)
- Python solutions for n=4,7 had overlaps (validation caught them)

## Gen111 Strategy: Deeper Small-N Optimization

The analysis showed small n values (1-20) have the highest s²/n ratios, meaning they're farthest from optimal. Focus here.

### Priority 1: Intensive Search for n=11-20

The Python optimizer only ran for n=1-10. Extend to n=20 with:
- Finer grid search (0.02 step instead of 0.04)
- More angles (16 or 32 instead of 8)
- Longer local refinement (10000+ iterations)
- **Critical**: Use strict overlap validation before accepting

### Priority 2: Re-optimize n=4 and n=7

These had potential but failed validation. Try:
- Restart with different initial positions
- Use tighter placement (touching but not overlapping)
- More careful angle selection

### Priority 3: Beam Search Placement

Instead of greedy (keep 1 best), try beam search (keep top-k):
- For each tree placement, keep top 5-10 partial solutions
- Evaluate all and pick best complete packing
- May find better configurations than greedy

### Priority 4: Pattern Analysis

Study the best solutions to find patterns:
- What angles work best together?
- What placement patterns emerge?
- Can we learn rules from optimal small-n solutions?

## Key Files

```
rust/src/evolved.rs           # Rust packing algorithm
python/optimize_small_n.py    # Small n optimizer
python/create_hybrid_submission.py  # Merge best solutions
python/validate_submission.py # Pre-submit validation (ALWAYS USE)
submission_best.csv           # Current best (85.77)
CLAUDE.md                     # Pre-submit workflow
```

## Commands

```bash
# Build Rust
cd rust && cargo build --release

# Optimize small n (extend to n=20)
python3 python/optimize_small_n.py --max-n 20 --compare-csv submission_gen109.csv --output python/optimized_small_n_20.json

# Create hybrid submission
python3 python/create_hybrid_submission.py --rust-csv submission_gen109.csv --python-json python/optimized_small_n.json --output submission_best.csv

# ALWAYS validate before submitting
python3 python/validate_submission.py submission_best.csv

# Submit to Kaggle
/Users/aerickson/Library/Python/3.14/bin/kaggle competitions submit -c santa-2025 -f submission_best.csv -m "Gen111: description"
```

## Success Metrics

- [ ] Find valid improvements for n=11-20
- [ ] Re-optimize n=4 and n=7 without overlaps
- [ ] Total score < 85.0 (0.9% improvement)
- [ ] Stretch: score < 84.0 (2.1% improvement)

## Important Reminders

1. **ALWAYS validate before Kaggle submit** - use `python3 python/validate_submission.py`
2. **Strict overlap check** - use `has_overlap_strict()` not `has_overlap()`
3. **Submission format**: id=`{n:03d}_{idx}`, values=`s`-prefixed, column=`deg`

## Start Here

1. Run `python3 python/optimize_small_n.py --max-n 15` with extended search
2. Check which n values improved
3. Validate and create hybrid submission
4. If time permits, implement beam search for placement
