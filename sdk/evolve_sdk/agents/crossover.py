"""Crossover agent - combines multiple parent solutions."""

from ..skills import get_mode_guidance

CROSSOVER_SYSTEM = """You are a crossover specialist for evolutionary algorithm discovery.

Your role: Combine the best aspects of multiple parent solutions into a hybrid.

Key principles:
1. ANALYZE each parent to understand what makes it good
2. IDENTIFY complementary strengths that could combine well
3. CREATE a hybrid that inherits the best of each
4. VERIFY the hybrid is correct and potentially better

Crossover strategies:
- Function-level: Use functions/methods from different parents
- Algorithm-level: Combine algorithmic ideas (e.g., one parent's sort + another's search)
- Structural: Use one parent's structure with another's implementation details
- Parameter mixing: Blend numerical parameters from parents

Output format: Always return valid JSON with your results.
"""

# Fallback guidance if no skill file is found
_DEFAULT_MODE_GUIDANCE = {
    "size": """
SIZE crossover strategies:
- Take the shortest implementation of each function
- Combine golf tricks from different parents
- Use one parent's algorithm with another's variable naming""",
    "perf": """
PERFORMANCE crossover strategies:
- Combine fast paths from different parents
- Mix data structures (e.g., one's caching + another's algorithm)
- Blend optimization techniques""",
    "ml": """
ML crossover strategies:
- Ensemble: Combine predictions from multiple models
- Architecture mixing: Layers from different models
- Feature union: Combine feature engineering from parents""",
}


def get_crossover_prompt(
    parent_files: list[str],
    parent_fitnesses: list[float],
    output_file: str,
    mode: str,
    generation: int,
) -> str:
    """Generate the prompt for a crossover agent."""

    parents_info = "\n".join(
        f"  - {f} (fitness: {fit})"
        for f, fit in zip(parent_files, parent_fitnesses)
    )

    # Try to get guidance from skill file first, fall back to defaults
    skill_guidance = get_mode_guidance(mode, "crossover")
    if skill_guidance:
        mode_section = f"""## Mode Guidance (from /evolve-{mode} skill)
{skill_guidance}"""
    else:
        mode_section = _DEFAULT_MODE_GUIDANCE.get(mode, _DEFAULT_MODE_GUIDANCE["size"])

    return f"""Create a crossover hybrid from multiple parents.

Parents:
{parents_info}

Generation: {generation}
Mode: {mode}

{mode_section}

Instructions:
1. Read ALL parent solutions
2. Analyze what makes each one good
3. Design a hybrid that combines their strengths
4. Implement and save to: {output_file}
5. Verify correctness

Return JSON:
{{
    "file": "{output_file}",
    "parents_used": [<list of parent files actually used>],
    "from_each_parent": {{
        "<parent1>": "<what was taken>",
        "<parent2>": "<what was taken>",
        ...
    }},
    "description": "<how they were combined>"
}}

CRITICAL: The hybrid must be COMPLETE and CORRECT. Test it before returning."""
