# Gen119 Continuation Prompt

We're continuing the Santa 2025 packing optimization. Current state:

- **Score**: 85.41 (Gen118)
- **Target**: ~69 (top leaderboard)
- **Gap**: ~24%

## What's Been Tried

### Gen118 (Current)
- Post-SA continuous angle refinement (±10°, 0.5° steps)
- Improved 136 groups, 0.04 points total
- **Key insight**: Continuous refinement works AFTER discrete SA, not during

### Previous Generations
- Gen116-117: CMA-ES, pattern-based SA (no improvement)
- Gen114-115: CMA-ES for small n (worked for n≤10)
- Gen103: Best-of-20 selection (+3.87%)
- Gen91b: Evolved greedy + SA (foundation algorithm)

## Gen119 Priority: Combined Position + Angle Refinement

Gen118 only refined angles. Gen119 should refine **both position and angle** together.

### Implementation Approach

```python
# For each tree, search neighborhood:
for dx in range(-5, 6):  # ±0.05 position
    for dy in range(-5, 6):
        for da in range(-20, 21):  # ±10° angle
            new_x = tree.x + dx * 0.01
            new_y = tree.y + dy * 0.01
            new_angle = tree.angle + da * 0.5

            if no_overlap(new_x, new_y, new_angle) and smaller_bbox():
                accept()
```

### Optimization Order
1. **Boundary trees first** - they define the bounding box
2. **Greedy acceptance** - take any improvement immediately
3. **Multiple passes** - repeat until convergence

## Quick Commands

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/santa-2025-packing

# Validate
python3 python/validate_submission.py submission_best.csv

# Read Gen118 results
cat GEN118_RESULTS.md

# View current score breakdown
python3 -c "
import csv, math
from collections import defaultdict
TREE_VERTICES = [(0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25), (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0), (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5)]
groups = defaultdict(list)
with open('submission_best.csv') as f:
    for row in csv.DictReader(f):
        n = int(row['id'].split('_')[0])
        groups[n].append((float(row['x'][1:]), float(row['y'][1:]), float(row['deg'][1:])))
def side(trees):
    pts = [(vx*math.cos(math.radians(a))-vy*math.sin(math.radians(a))+x, vx*math.sin(math.radians(a))+vy*math.cos(math.radians(a))+y) for x,y,a in trees for vx,vy in TREE_VERTICES]
    return max(max(p[0] for p in pts)-min(p[0] for p in pts), max(p[1] for p in pts)-min(p[1] for p in pts))
print(f'Total: {sum(side(groups[n])**2/n for n in range(1,201)):.4f}')
"
```

## Key Files

- `submission_best.csv` - Current best (85.41)
- `python/gen118_continuous_refine.py` - Angle-only refinement (reference)
- `python/validate_submission.py` - Strict validation

## Post-Improvement Checklist

1. Validate: `python3 python/validate_submission.py submission_best.csv`
2. Update visualizations: Generate SVGs from submission_best.csv
3. Update README.md with new score
4. Create GEN119_RESULTS.md
5. Commit and push
6. Submit to Kaggle

Start by implementing combined position + angle refinement in `python/gen119_position_refine.py`.
