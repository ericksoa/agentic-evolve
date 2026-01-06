# Gen117 Continuation Prompt

Copy everything below the line to continue after clearing context:

---

We're continuing the Santa 2025 packing optimization. Current state:

- **Score**: 85.45 (Gen115, unchanged after Gen116)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

## Gen116 Key Finding

CMA-ES on medium n (11-30) found NO strictly-valid improvements. Initial results showed 2-6% gains, but these had **edge-crossing overlaps** that Kaggle rejects. With proper segment-intersection validation, Rust Best-of-20 is already near-optimal for medium n.

**Critical**: Always use `python/validate_submission.py` before accepting any "improvement" - it matches Kaggle's strict checker.

## What's Been Tried (Won't Work)

- CMA-ES on n=11-15 with 10 restarts, 20k evals (Gen116) - no valid improvements
- Shapely-only validation misses edge crossings - must use segment intersection
- Parameter tuning, ML selection, global rotation (Gen92-108) - all failed

## What Works

- Rust Best-of-20 greedy + SA (Gen103) - core algorithm
- CMA-ES for small n (2-10) with strict validation (Gen110-115)
- Pattern discovery: trees at 90° offsets work well for small n

## Priority for Gen117

From GEN116_PLAN.md, try these next:

1. **Pattern-based initialization** (Priority 2)
   - Initialize with radial/hexagonal/diagonal patterns instead of chaotic Rust output
   - May help escape local optima for n=20-50
   - See `radial_pattern()` pseudocode in GEN116_PLAN.md

2. **Large n (50+) boundary optimization** (Priority 3)
   - Focus SA on trees defining the bounding box
   - Even 0.5% improvement on n=50-200 is significant (~0.3 points)
   - Only optimize boundary trees, not interior

3. **CMA-ES + SA hybrid** (Priority 4)
   - CMA-ES for global search (2k evals, large sigma)
   - SA for local refinement (50k iterations)
   - Test on n=15-25 first

## Key Files

- `submission_best.csv` - Current best (validated, score 85.45)
- `python/gen116_medium_n.py` - CMA-ES optimizer with strict validation
- `python/validate_submission.py` - MUST use before any submission
- `CLAUDE.md` - Workflow documentation

## Quick Start

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Validate current best
python3 python/validate_submission.py submission_best.csv

# Check score breakdown
python3 python/analyze_submission.py submission_best.csv
```

Start with Priority 1: Create `python/gen117_patterns.py` with radial/hexagonal initialization, then run CMA-ES on n=20-30 to see if different starting points help escape local optima.
