"""Initializer agent - bootstraps the initial population."""

from ..skills import get_mode_guidance

INITIALIZER_SYSTEM = """You are a population initializer for evolutionary algorithm discovery.

Your role: Create diverse initial solutions that will serve as the foundation for evolution.

Key principles:
1. DIVERSITY is critical - use different algorithms, approaches, data structures
2. Each solution must be VALID and pass basic correctness tests
3. Include both simple baselines and more sophisticated attempts
4. Document what approach each solution uses

Output format: Always return valid JSON with your results.
"""

# Fallback guidance if no skill file is found
_DEFAULT_MODE_GUIDANCE = {
    "size": """
Focus on CODE SIZE optimization:
- Include a naive/readable solution as baseline
- Include solutions using different language features (comprehensions, lambdas, etc.)
- Include solutions with different algorithms (even if slower, might be shorter)
- Measure size in BYTES (use: wc -c <file>)""",
    "perf": """
Focus on PERFORMANCE optimization:
- Include a naive O(n²) baseline for comparison
- Include solutions with different time complexities
- Include solutions using different data structures
- Measure performance with timing benchmarks""",
    "ml": """
Focus on MODEL ACCURACY optimization:
- Include a simple baseline model (logistic regression, decision tree)
- Include solutions with different architectures
- Include solutions with different feature engineering
- Measure accuracy with validation metrics (F1, accuracy, etc.)""",
}


def get_initializer_prompt(
    problem: str,
    mode: str,
    output_dir: str,
    population_size: int = 5,
    task_file: str | None = None,
) -> str:
    """Generate the prompt for the initializer agent."""

    # Try to get guidance from skill file first, fall back to defaults
    skill_guidance = get_mode_guidance(mode, "initializer")
    if skill_guidance:
        mode_section = f"""## Mode Guidance (from /evolve-{mode} skill)
{skill_guidance}"""
    else:
        mode_section = _DEFAULT_MODE_GUIDANCE.get(mode, _DEFAULT_MODE_GUIDANCE["size"])

    task_context = ""
    if task_file:
        task_context = f"""
Task specification file: {task_file}
Read this file first to understand the exact requirements."""

    return f"""Initialize population for: {problem}

Mode: {mode}
{mode_section}
{task_context}

Instructions:
1. Understand the problem requirements
2. Create {population_size} diverse initial solutions
3. Save each to: {output_dir}/gen0_<variant>.py (variants: a, b, c, d, e, ...)
4. Test each solution for basic correctness
5. Measure initial fitness for each

Return JSON:
{{
    "solutions": [
        {{"file": "<path>", "approach": "<description>", "fitness": <score>}},
        ...
    ],
    "best": {{"file": "<path>", "fitness": <score>}},
    "diversity_notes": "<how solutions differ>"
}}

IMPORTANT: Each solution must be COMPLETE and RUNNABLE. No placeholders or TODOs."""
