# Code Golf Workflow Prompt Template

**Copy and paste this prompt to start a new task session:**

---

## Standard Task Prompt

```
/evolve-size

## Task: Solve next ARC-AGI code golf task

### Instructions
1. Pick an UNSOLVED task from showcase/code-golf/tasks/ (check README.md Solved Problems table to avoid duplicates)
2. Read and understand the task JSON (train/test examples)
3. Create working solution, then golf it aggressively
4. For solutions 200+ bytes: run 5-10 generations of evolution mutations

### Required Deliverables
Create task directory with:
- `<task_id>/solution.py` - golfed Python solution
- `<task_id>/README.md` - with Pattern, Algorithm, Key Tricks, Byte History sections

### Documentation Updates (MANDATORY)
Update these files BEFORE committing:

**README.md:**
- Progress Summary table (Solved count, Total Score, Avg Score/Task, % of Winner)
- Solved Problems table (add entry, keep sorted by bytes ascending)
- Remove from Unsolved if listed
- Competition Status table (recalculate Est. Place)

**PROJECTION.md:**
- Current Status table
- Add task to appropriate tier in "Tasks by Difficulty"
- Update tier averages
- Recalculate Projected Final Standings

### Placement Formula
```
Conservative = 50 + (932,557 - (current_avg × 400)) / 500
Optimistic = 50 + (932,557 - tier_weighted_projection) / 500
```

### Commit Format
```
git add <task_id>/ showcase/code-golf/README.md showcase/code-golf/PROJECTION.md
git commit -m "<task_id>: <pattern description> (<bytes> bytes, +<score> pts)"
git push
```

### After Commit
Output this prompt template so user can copy it for the next task (context clearing).
```

---

## Quick Resume Prompt (shorter version)

```
Continue code golf workflow. Pick next unsolved ARC task from showcase/code-golf/tasks/.
Solve → Golf → Document → Update README.md & PROJECTION.md → Commit → Push → Output this prompt again.
Follow rules in CONTRIBUTING.md and CLAUDE.md.
```

---

## Re-Golf Existing Task Prompt

```
/evolve-size

Re-golf task <TASK_ID> in showcase/code-golf/<TASK_ID>/.
Run 5-10 generations of evolution mutations seeking byte savings.
Update solution.py, README.md (add evolution summary), and project files.
Commit with: "<task_id>: Re-golf <old>→<new> bytes (-X%)"
```

---

## Checklist (for Claude)

Before committing, verify:
- [ ] solution.py passes: `python3 evaluator.py <task_id> <task_id>/solution.py`
- [ ] README.md has: Pattern, Algorithm, Key Tricks, Byte History
- [ ] Main README.md updated: Progress Summary, Solved Problems table, Competition Status
- [ ] PROJECTION.md updated: Current Status, tier tables, Est. Place
- [ ] Commit message follows format
- [ ] Pushed to remote
- [ ] Output prompt template for next task

---

## Current Progress Reference

Check README.md for current stats. As of last update:
- 62 / 400 tasks solved
- 140,266 total points
- 2,262 avg pts/task (94.0% of winner)
- Winner: 962,070 pts (2,405 avg)
