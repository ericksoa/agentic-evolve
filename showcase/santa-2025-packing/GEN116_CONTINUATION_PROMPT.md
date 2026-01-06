# Gen116 Continuation Prompt

Copy this to continue after clearing context:

---

We're continuing the Santa 2025 packing optimization. Current state:

- Score: 85.45 (Gen115)
- Target: ~69 (top leaderboard)
- Gap: ~24%

Read GEN116_PLAN.md for the full plan. Here's the priority order:

1. **Large search budget for medium n (11-30)**: Run CMA-ES with 10 restarts, 20k evals on n=11-20. These contribute ~20% of score but haven't been Python-optimized.

2. **CMA-ES + SA hybrid**: Use CMA-ES for global search (2k evals, large sigma), then SA for local refinement (50k iterations). Test on n=10-15 first.

3. **Pattern-based optimization**: Initialize with radial/hexagonal/diagonal patterns instead of chaotic Rust output. May unlock improvements for n=20-50.

4. **Large n (50+) boundary optimization**: Focus SA on trees defining the bounding box. Even 0.5% improvement on n=50-200 is significant.

Key learnings from Gen115:
- Use strict segment-intersection validation (not just Shapely area)
- Start from known-valid solutions in submission_best.csv
- Multiple restarts with perturbation help escape local optima
- n=6 has edge-crossing issue even with zero Shapely area

Start with Priority 1: Create `python/gen116_medium_n.py` and run CMA-ES on n=11-20 with 10 restarts and 20k evals each.
