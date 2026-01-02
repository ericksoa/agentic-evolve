# Claude Code Golf Instructions

## MANDATORY WORKFLOW

When working on ANY code golf task (new solution or re-golf), you MUST follow this workflow:

### 1. Understand the Task
- Read `tasks/<task_id>.json`
- Analyze train/test examples
- Identify the pattern

### 2. Initial Solution
- Create working solution first
- Apply known golf tricks
- Document in README.md

### 3. Evolution (AlphaEvolve-Inspired) (REQUIRED)

**For ANY solution over 200 bytes, you MUST run evolution.**

We use techniques *inspired by* AlphaEvolve (mutation, crossover, selection) but
implemented via our `/evolve` skill - not the actual AlphaEvolve system.

```
Run at least 5-10 generations of mutations
Each generation: 3-4 mutation variants
Track all results in .evolve/<task_id>/
```

#### Mutation Types Per Generation

1. **Variable elimination**: Can any variable be removed or inlined?
2. **Expression rewriting**: Can expressions be simplified algebraically?
3. **Operator substitution**: `//2` → `>>1`, `==0` → `<1`, etc.
4. **Algorithm restructuring**: Different approach entirely?
5. **Crossover**: Combine successful tricks from prior generations

#### Evolution Commands

```bash
# Create evolution directory
mkdir -p .evolve/<task_id>/mutations

# Test each mutation
python3 evaluator.py <task_id> .evolve/<task_id>/mutations/genNx_name.py

# Track results
# Update .evolve/<task_id>/evolution.md with each generation
```

#### Stop Conditions

- 3 generations with no improvement (plateau)
- Solution under 100 bytes
- All mutation categories exhausted

### 4. Document Everything

README.md must include:
- Pattern description
- Algorithm explanation
- Key tricks used
- **Evolution Summary** with generations and key discoveries
- Byte history table

### 5. Verify and Save

```bash
python3 evaluator.py <task_id> <task_id>/solution.py
```

---

## QUICK REFERENCE: Golf Tricks

| Trick | Bytes Saved | Example |
|-------|-------------|---------|
| `eval(str(g))` | 4 vs `[*map(list,g)]` | Deep copy |
| `x>>1` | 2 vs `(x)//2` | Division by 2 |
| `<1` | 1 vs `==0` | Zero check |
| `or` | varies | `x=x or y` for defaults |
| Variable elimination | 3-10 | `v=u-t` → inline |

---

## EXAMPLE: Task 11852cab Evolution (AlphaEvolve-Inspired)

This task went from **333 → 280 bytes** (-16%) through 10 generations:

| Gen | Discovery | Bytes |
|-----|-----------|-------|
| 4 | `v=u-t` instead of `v=(s-t)//2` | 284 |
| 5 | Eliminate v entirely | 282 |
| 6 | `>>1` instead of `//2` | 280 |

See `.evolve/11852cab/evolution.md` for full details.

---

## REMEMBER

- Evolution is NOT optional for 200+ byte solutions
- Failed mutations are valuable data - document them
- Crossover between generations often yields breakthroughs
- Always update README.md with evolution results
