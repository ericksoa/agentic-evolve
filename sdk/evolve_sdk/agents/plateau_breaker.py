"""Plateau Breaker agent - escapes local optima through radical intervention.

The Plateau Breaker is triggered when evolution stalls (e.g., 3+ generations
with <2% fitness improvement). Its role is to:
1. Diagnose WHY evolution is stuck
2. Propose radical interventions (algorithm swaps, paradigm shifts)
3. Coordinate with mutators for high-risk/high-reward changes
4. Use memory to avoid previously-failed dramatic changes

This is the most important agent for preventing the #1 failure mode: getting stuck.
"""

from ..skills import get_mode_guidance

PLATEAU_BREAKER_SYSTEM = """You are a plateau-breaking specialist for evolutionary algorithm discovery.

Your role: Diagnose WHY evolution is stuck and propose RADICAL interventions to escape.

## When You're Called

You're activated when evolution has stalled - typically 3+ generations with <2% fitness
improvement. This means incremental mutations aren't working anymore. The population
is stuck in a local optimum.

## Core Responsibilities

1. **Diagnose the Plateau**: Why are incremental changes failing?
   - Is the algorithm fundamentally limited?
   - Are we stuck in a local optimum?
   - Have we exhausted the parameter space?
   - Is there a structural barrier?

2. **Propose Radical Interventions**: Not tweaks - paradigm shifts
   - Algorithm swaps (replace the entire approach)
   - Data structure changes (fundamentally different representation)
   - Paradigm shifts (iterative → recursive, eager → lazy, etc.)
   - Hybrid approaches (combine ideas from different solutions)

3. **Prioritize by Expected Impact**: Rank interventions by potential gain
4. **Avoid Known Failures**: Check memory for previously-failed radical changes

## Intervention Types

### Algorithm Swap
Replace the entire algorithm with a different approach.
Example: "Replace bubble sort with quicksort" or "Switch from BFS to A*"

### Paradigm Shift
Change the fundamental computation model.
Example: "Switch from iterative to recursive with memoization"
Example: "Convert to constraint satisfaction approach"

### Data Structure Change
Use a fundamentally different data representation.
Example: "Replace adjacency matrix with adjacency list"
Example: "Use bitboards instead of arrays"

### Decomposition Change
Break down the problem differently.
Example: "Split into subproblems with dynamic programming"
Example: "Use divide-and-conquer instead of greedy"

### External Inspiration
Suggest looking at established algorithms from literature.
Example: "Consider constraint propagation techniques from CSP"
Example: "Look at branch-and-bound approaches"

## Risk Assessment

Every radical intervention has risk. Assess honestly:
- **High risk**: Complete algorithm replacement, might regress significantly
- **Medium risk**: Structural change, preserves some existing logic
- **Low risk**: Adding technique on top of existing approach

## Output Format

Always return valid JSON:
{
    "diagnosis": {
        "plateau_type": "local_optimum" | "parameter_exhaustion" | "structural_barrier" | "algorithmic_limit",
        "explanation": "Why evolution is stuck",
        "evidence": ["observation1", "observation2"],
        "generations_stalled": <number>
    },
    "interventions": [
        {
            "type": "algorithm_swap" | "paradigm_shift" | "data_structure" | "decomposition" | "external_inspiration",
            "description": "What to change",
            "from_approach": "Current approach",
            "to_approach": "Proposed approach",
            "expected_impact": "high" | "medium" | "low",
            "risk_level": "high" | "medium" | "low",
            "rationale": "Why this might work"
        }
    ],
    "recommended_intervention": <index of best intervention>,
    "exploration_generations": <suggested generations to explore before giving up>,
    "message_to_mutators": "Guidance for next generation mutators"
}

Be bold. If incremental changes were working, you wouldn't be called.
"""


def get_plateau_breaker_prompt(
    current_champion: dict,
    fitness_history: list[dict],
    recent_mutations: list[dict],
    failed_interventions: list[dict] | None = None,
    mode: str = "perf",
    generation: int = 0,
    stall_threshold: float = 0.02,
    stall_generations: int = 3,
) -> str:
    """Generate the prompt for a plateau breaker agent.

    Args:
        current_champion: Current best solution
            - file: path to champion
            - fitness: current fitness score
            - algorithm_description: what approach it uses (if known)
        fitness_history: List of {generation, best_fitness, avg_fitness} records
        recent_mutations: Recent mutation attempts with outcomes
            - mutation_type: what was tried
            - fitness_delta: improvement (or regression)
            - accepted: whether it was promoted
        failed_interventions: Previous radical changes that failed (from memory)
        mode: Evolution mode (size, perf, ml)
        generation: Current generation number
        stall_threshold: What % improvement counts as "stalled" (default 2%)
        stall_generations: How many generations of stall triggered this
    """

    # Build fitness history section
    fitness_lines = []
    for record in fitness_history[-10:]:  # Last 10 generations
        gen = record.get("generation", "?")
        best = record.get("best_fitness", 0)
        avg = record.get("avg_fitness", 0)
        delta = record.get("improvement_pct", 0)
        fitness_lines.append(f"  Gen {gen}: best={best:.4f}, avg={avg:.4f}, delta={delta:+.2f}%")

    fitness_section = "\n".join(fitness_lines) if fitness_lines else "  (no history available)"

    # Build recent mutations section
    mutation_lines = []
    for mut in recent_mutations[-15:]:  # Last 15 mutations
        mut_type = mut.get("mutation_type", "unknown")
        delta = mut.get("fitness_delta", 0)
        accepted = "✓" if mut.get("accepted", False) else "✗"
        mutation_lines.append(f"  {accepted} {mut_type}: {delta:+.4f}")

    mutations_section = "\n".join(mutation_lines) if mutation_lines else "  (no recent mutations)"

    # Build failed interventions section (from memory)
    failed_section = ""
    if failed_interventions:
        failed_lines = []
        for intervention in failed_interventions[-5:]:  # Last 5 failed radical changes
            int_type = intervention.get("type", "unknown")
            description = intervention.get("description", "")
            reason = intervention.get("failure_reason", "unknown")
            failed_lines.append(f"  - {int_type}: {description}\n    Failed because: {reason}")
        failed_section = f"""
## Previously Failed Radical Changes (AVOID THESE)
{chr(10).join(failed_lines)}
"""

    # Champion info
    champion_file = current_champion.get("file", "unknown")
    champion_fitness = current_champion.get("fitness", 0)
    champion_approach = current_champion.get("algorithm_description", "not analyzed")

    # Get mode-specific guidance if available
    skill_guidance = get_mode_guidance(mode, "plateau_breaker")
    mode_section = ""
    if skill_guidance:
        mode_section = f"""
## Mode-Specific Guidance ({mode})
{skill_guidance}
"""

    return f"""## Plateau Breaker - Generation {generation}

Evolution has STALLED. You've been called because incremental mutations
aren't working. The population is stuck in a local optimum.

## Stall Detection
- Threshold: <{stall_threshold*100:.0f}% improvement per generation
- Stalled for: {stall_generations} generations
- This means: Parameter tweaks and small changes have been exhausted

## Current Champion
- File: {champion_file}
- Fitness: {champion_fitness:.4f}
- Approach: {champion_approach}

## Fitness History (Last 10 Generations)
{fitness_section}

## Recent Mutation Attempts
{mutations_section}
{failed_section}
{mode_section}

## Your Task

1. **Read the champion solution** - understand the current approach
2. **Diagnose the plateau** - why are incremental changes failing?
3. **Propose radical interventions** - not tweaks, paradigm shifts
4. **Rank by impact/risk** - which intervention is most promising?
5. **Avoid known failures** - don't repeat what's already failed

## Return JSON

After your analysis, return:
```json
{{
    "diagnosis": {{
        "plateau_type": "<type>",
        "explanation": "<why stuck>",
        "evidence": ["<observation1>", "<observation2>"],
        "generations_stalled": {stall_generations}
    }},
    "interventions": [
        {{
            "type": "<intervention_type>",
            "description": "<what to change>",
            "from_approach": "<current>",
            "to_approach": "<proposed>",
            "expected_impact": "high" | "medium" | "low",
            "risk_level": "high" | "medium" | "low",
            "rationale": "<why this might work>"
        }}
    ],
    "recommended_intervention": <index>,
    "exploration_generations": <number>,
    "message_to_mutators": "<guidance for next generation>"
}}
```

Plateau types: local_optimum, parameter_exhaustion, structural_barrier, algorithmic_limit
Intervention types: algorithm_swap, paradigm_shift, data_structure, decomposition, external_inspiration

IMPORTANT: Be bold. Incremental changes have failed. Propose something different."""


def get_intervention_prompt(
    intervention: dict,
    champion_file: str,
    output_file: str,
    mode: str,
    generation: int,
) -> str:
    """Generate a prompt to execute a specific radical intervention.

    This is used to guide a mutator to implement the plateau breaker's recommendation.

    Args:
        intervention: The intervention to execute (from plateau breaker output)
        champion_file: Current champion to transform
        output_file: Where to save the new solution
        mode: Evolution mode
        generation: Current generation
    """

    int_type = intervention.get("type", "unknown")
    description = intervention.get("description", "")
    from_approach = intervention.get("from_approach", "current approach")
    to_approach = intervention.get("to_approach", "new approach")
    rationale = intervention.get("rationale", "")
    risk = intervention.get("risk_level", "high")

    return f"""## Execute Radical Intervention - Generation {generation}

The Plateau Breaker has diagnosed a stall and recommended this intervention.
Your job: Implement the radical change, not an incremental tweak.

## Intervention
- Type: {int_type}
- Description: {description}
- From: {from_approach}
- To: {to_approach}
- Rationale: {rationale}
- Risk Level: {risk}

## Source
- Champion: {champion_file}

## Target
- Output: {output_file}

## Instructions

1. **Read the champion** - understand the current implementation
2. **Implement the intervention** - make the RADICAL change described above
3. **This is not a tweak** - you're replacing/restructuring, not adjusting
4. **Preserve correctness** - the solution must still solve the problem
5. **Save to output file**

## Return JSON

```json
{{
    "file": "{output_file}",
    "intervention_type": "{int_type}",
    "description": "<what you actually changed>",
    "approach_before": "<how it worked before>",
    "approach_after": "<how it works now>",
    "confidence": "high" | "medium" | "low"
}}
```

IMPORTANT: This is a high-risk, high-reward mutation. Be bold but correct."""


def detect_plateau(
    fitness_history: list[dict],
    threshold: float = 0.02,
    window: int = 3,
) -> tuple[bool, int, float]:
    """Detect if evolution has plateaued.

    Args:
        fitness_history: List of {generation, best_fitness} records
        threshold: Maximum improvement % to count as stalled (default 2%)
        window: Number of generations to check (default 3)

    Returns:
        (is_stalled, generations_stalled, avg_improvement)
    """
    if len(fitness_history) < window + 1:
        return False, 0, 0.0

    recent = fitness_history[-(window + 1):]

    improvements = []
    for i in range(1, len(recent)):
        prev_fitness = recent[i - 1].get("best_fitness", 0)
        curr_fitness = recent[i].get("best_fitness", 0)

        if prev_fitness > 0:
            pct_improvement = (curr_fitness - prev_fitness) / prev_fitness
            improvements.append(pct_improvement)

    if not improvements:
        return False, 0, 0.0

    avg_improvement = sum(improvements) / len(improvements)
    is_stalled = avg_improvement < threshold
    generations_stalled = window if is_stalled else 0

    return is_stalled, generations_stalled, avg_improvement
