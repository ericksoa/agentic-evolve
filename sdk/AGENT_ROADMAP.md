# Agent Roadmap

Minimal, focused plan for new agents. Marie Kondo rules: only what solves real problems.

## The Problems We're Solving

1. **Getting stuck** - plateau, local optima (most common failure)
2. **Wasted generations** - crashes, errors, broken mutations
3. **Blind optimization** - no understanding of *why* something works
4. **Premature convergence** - population becomes homogeneous too fast

## Current Agents (7)

| Agent | Role |
|-------|------|
| Initializer | Bootstrap initial population |
| Mutator | Single focused mutations |
| Evaluator | Measure fitness |
| Adversary | Trust validation, exploit detection |
| Crossover | Combine parent solutions |
| Arbitrator | Balanced analysis for human escalation |
| Reporter | Surface messages to human operator |

## New Agents to Build

### Phase 1: Fix What Hurts Now

#### Debugger Agent 🐛
**Problem:** Every failed mutation is a wasted API call

**Trigger:** Mutation fails (crash, error, timeout)

**Responsibilities:**
- Analyze error to identify root cause
- Compare with parent to find breaking change
- Suggest minimal fix
- Categorize failure type for learning

**Output:**
```json
{
  "failed_mutation": "gen3b.py",
  "error_type": "IndexError",
  "root_cause": "Off-by-one error in loop bound",
  "suggested_fix": "Change range(n) to range(n-1) on line 43",
  "failure_category": "boundary_condition",
  "lesson": "When mutating loop bounds, verify array access patterns"
}
```

**Messages:** Shares failure patterns to help mutators avoid same mistakes

---

#### Plateau Breaker Agent 🔨
**Problem:** Evolution gets stuck in local optima

**Trigger:** 3+ generations with <2% fitness improvement

**Responsibilities:**
- Diagnose why evolution is stuck
- Propose radical interventions (algorithm swaps, paradigm shifts)
- Coordinate with Mutator for high-risk/high-reward changes
- Use memory to avoid previously-failed dramatic changes

**Output:**
```json
{
  "diagnosis": "Stuck in local optimum - parameter tweaks exhausted",
  "proposed_interventions": [
    {"type": "algorithm_swap", "from": "backtracking", "to": "constraint_propagation"},
    {"type": "paradigm_shift", "description": "Switch from iterative to recursive with memoization"}
  ],
  "risk_level": "high"
}
```

**Messages:** Broadcasts strategy changes to all mutators

---

### Phase 2: Make Evolution Smarter

#### Meta-Strategist Agent 🎯
**Problem:** Manual tuning of evolution parameters

**Trigger:** Every N generations (default: 5)

**Responsibilities:**
- Analyze mutation success rates by type
- Compare crossover vs mutation effectiveness
- Track population diversity
- Adjust strategy mix based on what's working

**Output:**
```json
{
  "analysis": {
    "mutation_effectiveness": {"parameter_tweak": 0.12, "structural": 0.45},
    "crossover_contribution": 0.08,
    "diversity_index": 0.34
  },
  "recommendations": [
    {"action": "increase_structural_mutations", "rationale": "3x more effective"},
    {"action": "reduce_crossover_frequency", "rationale": "Low contribution"}
  ]
}
```

**Messages:** Broadcasts strategy updates to all agents

---

### Phase 3: Prevent Silent Failures

#### Diversity Guardian Agent 🌈
**Problem:** Population converges prematurely, killing exploration

**Trigger:** Continuous monitoring, alerts when diversity below threshold

**Responsibilities:**
- Compute genotypic diversity (code similarity via embeddings)
- Compute phenotypic diversity (fitness spread)
- Alert when population too homogeneous
- Can request injection of orthogonal solutions

**Output:**
```json
{
  "genotypic_diversity": 0.23,
  "phenotypic_diversity": 0.15,
  "alert": true,
  "recommendation": "Inject 2 orthogonal solutions or reduce selection pressure"
}
```

**Messages:** Urgent alerts when diversity critical

---

### Phase 4: Build Trust (Optional)

#### Ablation Agent 🔬
**Problem:** No understanding of why champion works

**Trigger:** New champion crowned (rare)

**Responsibilities:**
- Systematically remove components, measure fitness impact
- Identify dead code and unnecessary complexity
- Create minimal reproducing version
- Document which parts are load-bearing

**Output:**
```json
{
  "original_fitness": 20407,
  "ablation_results": [
    {"removed": "early_termination_check", "fitness_delta": -4200, "verdict": "critical"},
    {"removed": "debug_assertions", "fitness_delta": +50, "verdict": "dead_code"}
  ],
  "minimal_fitness": 20350,
  "insight": "95% of gains from early termination and inlining"
}
```

**Messages:** Shares discoveries about what's actually working

---

## What We're NOT Building

| Proposed | Decision | Reason |
|----------|----------|--------|
| Portfolio Manager | Cut | Merge into Diversity Guardian |
| Benchmark Forger | Cut | Adversary already does this |
| Simplifier | Cut | Ablation gives 80% of value |

---

## External Patterns to Study (Not Import)

- [obra/superpowers](https://github.com/obra/superpowers) - systematic-debugging, verification-before-completion
- [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) - import per-showcase, not into core

---

## Implementation Order

```
[x] Messaging system (done)
[x] Reporter agent (done)
[ ] Phase 1: Debugger Agent
[ ] Phase 1: Plateau Breaker Agent
[ ] Phase 2: Meta-Strategist Agent
[ ] Phase 3: Diversity Guardian Agent
[ ] Phase 4: Ablation Agent (optional)
```

---

## Principles

1. **Don't add agents that run every generation** - bloat
2. **Don't duplicate existing agent responsibilities** - Adversary/Mutator already do a lot
3. **Import skills per-showcase, not into core** - keep SDK lean
4. **If it's not obvious we need it, we don't** - add later if pain emerges
