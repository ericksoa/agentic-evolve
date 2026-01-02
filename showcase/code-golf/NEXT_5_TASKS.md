# Next 5 Tasks Plan

## Session Strategy

**ONE task per session** - Clear context after each task for maximum evolution effectiveness.

---

## Task Queue (Easiest First)

| # | Task ID | Grid Size | Pattern | Est. Difficulty | Est. Bytes |
|---|---------|-----------|---------|-----------------|------------|
| 1 | `25d8a9c8` | 3×3 | Row uniformity → 5 or 0 | Easy | 60-100 |
| 2 | `3c9b0459` | 3×3 | 180° rotation | Easy | 40-80 |
| 3 | `3aa6fb7a` | 7×7 | L-shaped 8s → add 1 at corner | Easy-Medium | 100-150 |
| 4 | `4258a5f9` | 9×9 | Draw 3×3 box around each 5 | Medium | 100-180 |
| 5 | `28e73c20` | Variable | Spiral maze pattern | Medium-Hard | 150-300 |

---

## Session Template

For each task, follow this workflow:

### 1. Understand (5 min)
```bash
# Read task
cat tasks/<task_id>.json | python3 -m json.tool

# Analyze examples manually
```

### 2. Solve (10-20 min)
- Write initial working solution
- Test with evaluator
```bash
python3 evaluator.py <task_id> <task_id>/solution.py
```

### 3. Golf (10-20 min)
- Apply known tricks
- Target: minimize bytes

### 4. Evolution (REQUIRED for 200+ bytes)
```bash
mkdir -p .evolve/<task_id>/mutations
```
- Run 5-10 generations, 3-4 mutations each
- Track all results
- Stop after 3 generations with no improvement

### 5. Document
Create `<task_id>/README.md` with:
- Pattern description
- Algorithm explanation
- Key tricks used
- Byte history
- Evolution summary (if applicable)

### 6. Commit & Push
```bash
git add <task_id>/
git commit -m "<task_id>: <pattern> (<bytes> bytes)"
git push
```

### 7. Provide Summary
After completing each task, output:
- Final byte count and score
- Key tricks discovered
- Prompt for next task

---

## Task Details

### Task 1: `25d8a9c8` (Easy)

**Pattern**: For each row in 3×3 grid, if all values are same → output [5,5,5], else [0,0,0]

**Examples**:
```
Input:  [[4,4,4],[2,3,2],[2,3,3]]
Output: [[5,5,5],[0,0,0],[0,0,0]]
        Row 0: all 4s → 5,5,5
        Row 1: mixed → 0,0,0
        Row 2: mixed → 0,0,0
```

**Approach**: `[[5,5,5]if len(set(r))==1 else[0,0,0]for r in g]`

**Target**: <100 bytes

---

### Task 2: `3c9b0459` (Easy)

**Pattern**: 180° rotation of 3×3 grid

**Examples**:
```
Input:  [[2,2,1],[2,1,2],[2,8,1]]
Output: [[1,8,2],[2,1,2],[1,2,2]]
```

**Approach**: `g[::-1]` for row reversal, `r[::-1]` for column reversal

**Target**: <80 bytes (potentially very short!)

---

### Task 3: `3aa6fb7a` (Easy-Medium)

**Pattern**: Find L-shaped 8 patterns (3 cells), add a 1 at the empty corner

**Examples**:
```
Input:  8 .     Output: 8 1
        8 8             8 8
```

**Approach**: For each L-shape of 8s, find the missing corner and place a 1

**Target**: 100-150 bytes

---

### Task 4: `4258a5f9` (Medium)

**Pattern**: Draw a 3×3 box of 1s around each 5 in the grid

**Examples**:
```
Input:  . . .     Output: 1 1 1
        . 5 .             1 5 1
        . . .             1 1 1
```

**Approach**: Find all 5s, fill surrounding 3×3 with 1s (keeping 5 in center)

**Target**: 100-180 bytes

---

### Task 5: `28e73c20` (Medium-Hard)

**Pattern**: Generate spiral maze pattern from all-zero grid

**Examples**:
- 6×6 grid → nested rectangles spiraling inward
- Output alternates between 3 (wall) and 0 (path)

**Approach**: Recursive or iterative spiral drawing

**Target**: 150-300 bytes (complex pattern generation)

---

## Resume Prompts

### For Task 1 (`25d8a9c8`):
```
Solve ARC task 25d8a9c8.

Pattern: 3×3 grid - for each row, output [5,5,5] if all values same, else [0,0,0].

Requirements:
1. Read tasks/25d8a9c8.json
2. Write working solution to 25d8a9c8/solution.py
3. Run evolution if >200 bytes
4. Document in 25d8a9c8/README.md
5. Test: python3 evaluator.py 25d8a9c8 25d8a9c8/solution.py
6. Commit and push

Target: <100 bytes
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

### For Task 2 (`3c9b0459`):
```
Solve ARC task 3c9b0459.

Pattern: 3×3 grid - 180° rotation.

Requirements:
1. Read tasks/3c9b0459.json
2. Write working solution to 3c9b0459/solution.py
3. Run evolution if >200 bytes
4. Document in 3c9b0459/README.md
5. Test: python3 evaluator.py 3c9b0459 3c9b0459/solution.py
6. Commit and push

Target: <80 bytes
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

### For Task 3 (`3aa6fb7a`):
```
Solve ARC task 3aa6fb7a.

Pattern: 7×7 grid - find L-shaped 8s, add 1 at the empty corner.

Requirements:
1. Read tasks/3aa6fb7a.json
2. Write working solution to 3aa6fb7a/solution.py
3. Run evolution if >200 bytes
4. Document in 3aa6fb7a/README.md
5. Test: python3 evaluator.py 3aa6fb7a 3aa6fb7a/solution.py
6. Commit and push

Target: 100-150 bytes
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

### For Task 4 (`4258a5f9`):
```
Solve ARC task 4258a5f9.

Pattern: 9×9 grid - draw 3×3 box of 1s around each 5.

Requirements:
1. Read tasks/4258a5f9.json
2. Write working solution to 4258a5f9/solution.py
3. Run evolution if >200 bytes
4. Document in 4258a5f9/README.md
5. Test: python3 evaluator.py 4258a5f9 4258a5f9/solution.py
6. Commit and push

Target: 100-180 bytes
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

### For Task 5 (`28e73c20`):
```
Solve ARC task 28e73c20.

Pattern: Variable-size grid (all zeros) - generate spiral maze pattern with 3s.

Requirements:
1. Read tasks/28e73c20.json
2. Write working solution to 28e73c20/solution.py
3. Run evolution (REQUIRED - likely >200 bytes)
4. Document in 28e73c20/README.md
5. Test: python3 evaluator.py 28e73c20 28e73c20/solution.py
6. Commit and push

Target: 150-300 bytes
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| All 5 tasks passing | 5/5 |
| Total bytes | <700 |
| Total points | +12,000+ |
| READMEs complete | 5/5 |

---

## After Completion

1. Update TOMORROW_PLAN.md with results
2. Update PROJECTION.md with new task stats
3. Calculate total points gained
4. Identify next batch of 5 tasks
