# Tomorrow's Prompt

Copy and paste this to continue:

---

Continue working on the Santa 2025 packing competition. Read `EVOLUTION_STATE.md` for full context.

**Current blocker**: Kaggle rejected our submission with "Overlapping trees in group 104". Our local `has_overlaps()` passes but Kaggle's validation fails.

**Today's tasks in priority order**:

1. **Fix overlap validation** (must do first)
   - Read `rust/src/lib.rs` to understand our `has_overlaps()` implementation
   - Extract and visualize n=104 from `rust/submission.csv` to see the problem
   - Compare our polygon intersection vs what Kaggle might use
   - Likely fix: add safety margin (shrink polygons by epsilon) or increase CSV precision beyond 6 decimals
   - Create a stricter local validator that catches what Kaggle catches
   - Generate new submission and test on Kaggle

2. **Implement pairwise ranking model** (Gen104 ML improvement)
   - The Gen102 model predicted well (MAE=0.04) but failed at selection
   - Root cause: trained for absolute scores, but needed relative ranking
   - New approach: train on pairs (packing_A, packing_B) → "is A better?"
   - Use existing best-of-N data to generate training pairs
   - Should improve on pure best-of-N selection

3. **If time permits**: Add better features to the ranking model
   - Minimum pairwise tree distance
   - Boundary utilization
   - Local density variance

Start by reading `EVOLUTION_STATE.md` and `rust/src/lib.rs`, then fix the overlap issue before moving to ML improvements.
