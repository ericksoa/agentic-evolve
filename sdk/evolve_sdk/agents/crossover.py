"""Crossover agent - combines multiple parent solutions."""

import json

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


def get_workspace_crossover_prompt(
    parent_workspaces: list[str],
    parent_fitnesses: list[float],
    baseline_workspace: str,
    output_workspace: str,
    generation: int,
) -> str:
    """Generate prompt for workspace-mode crossover (directory-level evolution).

    Merges changes from two parent workspace directories against the
    baseline repository.

    Args:
        parent_workspaces: List of parent workspace directory paths
        parent_fitnesses: List of parent fitness scores
        baseline_workspace: Path to the unmodified baseline repository
        output_workspace: Path to save hybrid workspace
        generation: Current generation number
    """
    parents_info = "\n".join(
        f"  - {d} (fitness: {fit:.4f})"
        for d, fit in zip(parent_workspaces, parent_fitnesses)
    )

    return f"""Create a crossover hybrid from multiple parent workspaces.

You are working in: {output_workspace}
(This is a copy of the baseline. Apply the best changes from the parents.)

Parents (read-only, for reference):
{parents_info}

Baseline (unmodified): {baseline_workspace}
Generation: {generation}

## Instructions
1. Compare each parent workspace against the baseline to identify what changed
2. Analyze which changes in each parent contribute to passing tests
3. Apply non-conflicting changes from both parents into your workspace
4. For conflicting changes, prefer the approach from the higher-fitness parent
5. Do NOT modify the parent workspaces — only modify files in {output_workspace}

## Strategy
- Use `diff` or file comparison to identify changes from baseline
- Merge changes file-by-file
- Prioritize changes that fix different tests (complementary changes)
- When parents modify the same file differently, analyze which change is more correct

Return JSON:
{{
    "parents_used": {json.dumps(parent_workspaces)},
    "from_each_parent": {{
        "<parent1>": "<what was taken>",
        "<parent2>": "<what was taken>"
    }},
    "description": "<how they were combined>",
    "conflicts_resolved": [<list of files with conflicts and how resolved>]
}}

CRITICAL: The hybrid must preserve all passing tests from both parents."""
