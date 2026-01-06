# Gen115 Continuation Prompt

Copy and paste this to start tomorrow's session:

---

We're continuing the Santa 2025 packing optimization. Current state:

- **Score**: 85.50 (Gen114)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

Read `GEN115_PLAN.md` for the full plan. Here's the priority order:

## Today's Goals

1. **Fix n=7**: CMA-ES found 5.2% improvement but had overlaps. Re-run with stricter constraints.

2. **Optimize n=3-6**: Run CMA-ES on these small n values that haven't been aggressively optimized.

3. **Medium n (11-30)**: This range is untouched - extract from CSV and run CMA-ES/SA optimization.

4. **Continuous angles**: If time permits, try optimizing with continuous rotation angles instead of discrete 45°.

## Key Files

- `submission_best.csv` - Current best (85.50)
- `python/optimized_small_n.json` - Optimized n=1-10 values
- `python/gen114_runner.py` - Reference for CMA-ES setup
- `python/validate_submission.py` - ALWAYS validate before submitting

## Important Lessons from Gen114

1. **Always use actual submission as baseline** - JSON files can become stale
2. **Strict validation after CMA-ES** - CMA-ES penalty function doesn't guarantee valid solutions
3. **CMA-ES escapes local optima** - Better than SA for global optimization

Start by reading GEN115_PLAN.md, then create gen115_runner.py targeting n=7 first with stricter overlap handling.
