"""Adversary/Teacher agent - challenges candidate solutions for trust-aware evolution.

The Adversary agent acts as a skeptical reviewer that:
1. Challenges claimed fitness improvements
2. Detects evaluator exploitation or blind spots
3. Triggers validation escalation when suspicious
4. Provides trust-adjusted fitness scores
"""

from ..skills import get_mode_guidance

ADVERSARY_SYSTEM = """You are a skeptical Adversary/Teacher agent for evolutionary algorithm discovery.

Your role: Challenge candidate solutions before they're promoted, detecting potential
evaluator exploitation or suspiciously large improvements that may not generalize.

## Core Responsibilities

1. **Skeptical Review**: Question fitness claims, especially large jumps
2. **Exploit Detection**: Look for solutions gaming the evaluator (overfitting to test cases,
   hardcoding expected outputs, exploiting evaluation timing)
3. **Trust Scoring**: Rate confidence in the claimed fitness (0.0 to 1.0)
4. **Escalation Triggers**: Recommend higher-fidelity validation when suspicious

## Trust Heuristics

Red flags that LOWER trust:
- Fitness jump > 20% in a single mutation
- Solution size dropped dramatically (code golf cheats)
- Hardcoded constants matching test cases
- Suspiciously fast execution (cached/memoized answers)
- Pattern matching or lookup tables instead of algorithms
- No clear algorithmic improvement explaining the gain

Green flags that RAISE trust:
- Incremental improvements (< 10% per generation)
- Clear algorithmic innovation visible in diff
- Improvement consistent across multiple test cases
- Solution generalizes beyond training data
- Performance scales with input size as expected

## Escalation Levels

- Level 0: Basic evaluator (current) - default
- Level 1: Extended test suite - more edge cases
- Level 2: Out-of-distribution tests - unseen data
- Level 3: Gold standard validation - expensive but definitive

## Output Format

Always return valid JSON:
{
    "trust_score": 0.0-1.0,
    "escalation_level": 0-3,
    "flags": ["list", "of", "concerns"],
    "recommendation": "accept" | "challenge" | "reject",
    "adjusted_fitness": <original * trust_factor>,
    "analysis": "detailed reasoning"
}
"""


def get_adversary_prompt(
    candidate: dict,
    parent: dict | None,
    evaluation_result: dict,
    mode: str,
    generation: int,
    population_best: float,
) -> str:
    """Generate the prompt for an adversary agent to challenge a candidate.

    Args:
        candidate: The candidate solution being challenged
            - file: path to solution file
            - mutation_type: how it was created
            - hypothesis: claimed improvement reason
        parent: The parent solution it was mutated from (None if initial)
            - file: path
            - fitness: parent's fitness score
        evaluation_result: The evaluator's assessment
            - fitness: claimed fitness score
            - valid: whether it passed tests
            - notes: evaluator observations
        mode: Optimization mode (size, perf, ml)
        generation: Current generation number
        population_best: Best fitness in current population
    """

    candidate_file = candidate.get("file", "unknown")
    candidate_mutation = candidate.get("mutation_type", "unknown")
    candidate_hypothesis = candidate.get("hypothesis", "no hypothesis provided")

    eval_fitness = evaluation_result.get("fitness", 0)
    eval_valid = evaluation_result.get("valid", False)
    eval_notes = evaluation_result.get("notes", "")

    parent_section = ""
    fitness_jump = 0
    fitness_jump_pct = 0

    if parent:
        parent_file = parent.get("file", "unknown")
        parent_fitness = parent.get("fitness", 0)
        fitness_jump = eval_fitness - parent_fitness
        if parent_fitness > 0:
            fitness_jump_pct = (fitness_jump / parent_fitness) * 100
        parent_section = f"""
## Parent Solution
- File: {parent_file}
- Fitness: {parent_fitness:.4f}
- Fitness jump: {fitness_jump:+.4f} ({fitness_jump_pct:+.1f}%)
"""

    improvement_vs_best = ""
    if population_best > 0 and eval_fitness > population_best:
        improvement = ((eval_fitness - population_best) / population_best) * 100
        improvement_vs_best = f"\n**NOTE**: This would be a NEW CHAMPION - {improvement:.1f}% above current best ({population_best:.4f})"

    # Get mode-specific guidance if available
    skill_guidance = get_mode_guidance(mode, "adversary")
    mode_section = ""
    if skill_guidance:
        mode_section = f"""
## Mode-Specific Guidance ({mode})
{skill_guidance}
"""

    return f"""## Adversary Challenge - Generation {generation}

You are reviewing a candidate solution BEFORE it gets promoted to the population.
Your job: determine if the claimed fitness improvement is trustworthy.

## Candidate Solution
- File: {candidate_file}
- Mutation type: {candidate_mutation}
- Claimed hypothesis: {candidate_hypothesis}
{parent_section}

## Evaluator's Assessment
- Fitness: {eval_fitness:.4f}
- Valid: {eval_valid}
- Notes: {eval_notes}
{improvement_vs_best}
{mode_section}

## Your Task

1. **Read the candidate solution** - examine the actual code
2. **If parent exists, diff against parent** - understand what changed
3. **Assess the claimed fitness** - is the improvement real and generalizable?
4. **Look for red flags**:
   - Hardcoded answers or lookup tables
   - Overfitting to known test cases
   - Timing exploits (cached results)
   - Size tricks that break functionality
5. **Assign trust score and recommendation**

## Return JSON

After your analysis, return:
```json
{{
    "trust_score": <0.0 to 1.0>,
    "escalation_level": <0-3>,
    "flags": ["concern1", "concern2"],
    "recommendation": "accept" | "challenge" | "reject",
    "adjusted_fitness": <fitness * trust_factor>,
    "analysis": "Your detailed reasoning"
}}
```

Where:
- trust_score: 1.0 = fully trusted, 0.5 = suspicious, 0.0 = definite exploit
- escalation_level: 0 = no additional validation, 3 = full gold-standard check
- recommendation:
  - "accept" = promote with adjusted fitness
  - "challenge" = needs escalation before decision
  - "reject" = do not promote (fitness set to 0)

IMPORTANT: Read the actual solution code before making your assessment.
Skepticism is valuable, but don't reject legitimate improvements."""


def get_escalation_prompt(
    candidate: dict,
    trust_result: dict,
    escalation_level: int,
    mode: str,
    extended_test_command: str | None = None,
) -> str:
    """Generate prompt for escalated validation.

    Called when the adversary requests Level 1+ escalation.

    Args:
        candidate: The candidate solution
        trust_result: The adversary's initial assessment
        escalation_level: Requested escalation (1-3)
        mode: Optimization mode
        extended_test_command: Command for extended validation (if available)
    """

    candidate_file = candidate.get("file", "unknown")
    flags = trust_result.get("flags", [])
    flags_str = "\n".join(f"  - {f}" for f in flags) if flags else "  (none)"

    level_descriptions = {
        1: "Extended test suite with more edge cases and boundary conditions",
        2: "Out-of-distribution tests using unseen data patterns",
        3: "Gold-standard validation with highest-fidelity evaluation",
    }

    level_desc = level_descriptions.get(escalation_level, "Unknown escalation level")

    command_section = ""
    if extended_test_command:
        command_section = f"""
## Escalation Command
Run this for extended validation:
{extended_test_command}
"""

    return f"""## Escalated Validation - Level {escalation_level}

The Adversary flagged this candidate for additional validation.

## Candidate
- File: {candidate_file}
- Original claimed fitness: {trust_result.get('adjusted_fitness', 0):.4f}

## Adversary Concerns
{flags_str}

## Escalation Level {escalation_level}
{level_desc}
{command_section}

## Your Task

Perform the escalated validation:

### Level 1 - Extended Tests
- Run additional edge case tests
- Test boundary conditions
- Verify handling of empty/null/extreme inputs

### Level 2 - Out-of-Distribution
- Generate novel test cases unlike training data
- Test with different data distributions
- Verify generalization beyond known patterns

### Level 3 - Gold Standard
- Run the most comprehensive validation available
- Compare against reference implementation
- Measure on held-out test set

## Return JSON

```json
{{
    "escalation_passed": true/false,
    "revised_fitness": <updated fitness after escalation>,
    "revised_trust": <new trust score>,
    "tests_run": ["test1", "test2"],
    "failures": ["any failures"],
    "final_recommendation": "accept" | "reject"
}}
```

Be thorough - the adversary flagged this for a reason."""
