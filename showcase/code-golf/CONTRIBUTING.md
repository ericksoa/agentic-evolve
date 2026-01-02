# Code Golf Solution Standards

## Required Files

Every solved task MUST have these files in its directory:

```
<task_id>/
├── solution.py    # The golfed Python solution
└── README.md      # Documentation (REQUIRED)
```

---

## README.md Requirements

**Every solution MUST have a README.md** documenting the approach. This is non-negotiable.

### Minimum Required Sections

```markdown
# Task <task_id>

## Pattern
[One sentence describing what the transformation does]

## Algorithm
[2-4 sentences explaining the approach taken]

## Key Tricks
- [List golf tricks used]
- [e.g., "walrus operator for inline assignment"]
- [e.g., "1D array indexing instead of 2D"]

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | XXX | Initial solution |
| v2 | YYY | [what changed] |
```

### Example README

```markdown
# Task 10fcaaa3

## Pattern
Tile input 2x2, then add 8s diagonally adjacent to colored cells.

## Algorithm
First tile the input grid by doubling in both dimensions. Then identify all
non-zero cells in the tiled output. For each empty cell, check if it's
diagonally adjacent (distance √2) to any colored cell - if so, fill with 8.

## Key Tricks
- `g*2` to repeat list (tile rows)
- `r*2` to repeat row (tile columns)
- `(r-i)**2+(c-j)**2==2` for diagonal adjacency check
- `2in{...}` shorter than `any(...==2 for...)`

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 198 | Initial working solution |
| v2 | 176 | Inline S computation, use `2in{...}` trick |
```

---

## Solution Workflow

When solving a new task, follow this order:

### 1. Solve First
```python
# Get it working, don't worry about bytes yet
def solve(g):
    # ... working solution
```

### 2. Golf It
```python
# Minimize bytes while maintaining correctness
def solve(g):...  # one-liner if possible
```

### 3. Document Immediately
Create README.md **before** committing. Do not commit solutions without documentation.

### 4. Verify
```bash
python3 evaluator.py <task_id> <task_id>/solution.py
```

### 5. Commit Together
```bash
git add <task_id>/solution.py <task_id>/README.md
git commit -m "<task_id>: <pattern> (<bytes> bytes)"
```

---

## Pre-Commit Checklist

Before committing any solution:

- [ ] `solution.py` passes all train AND test examples
- [ ] `README.md` exists with all required sections
- [ ] Pattern description is clear and concise
- [ ] Algorithm explanation is understandable
- [ ] Key tricks are documented for future reference
- [ ] Byte history shows evolution (even if just v1)

---

## Why Documentation Matters

1. **Context rot** - Without docs, we forget how solutions work
2. **Re-golf opportunities** - Documented tricks help identify improvements
3. **Knowledge transfer** - Tricks discovered apply to other tasks
4. **Debugging** - Easier to fix broken solutions with documented intent

---

## Quick Reference: Common Sections

### For Simple Tasks (< 150 bytes)
- Pattern: 1 sentence
- Algorithm: 1-2 sentences
- Key Tricks: 2-3 bullets
- Byte History: 1-2 rows

### For Complex Tasks (300+ bytes)
- Pattern: 1-2 sentences
- Algorithm: 3-5 sentences, may include pseudocode
- Key Tricks: 4-6 bullets with explanations
- Byte History: multiple iterations
- Optional: "Failed Approaches" section

### For Very Hard Tasks (600+ bytes)
All of the above, plus:
- "Challenges" section explaining what made it hard
- "Potential Improvements" for future re-golf attempts
