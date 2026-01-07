"""Mutator agent - creates mutations of existing solutions."""

MUTATOR_SYSTEM = """You are a mutation specialist for evolutionary algorithm discovery.

Your role: Create ONE focused mutation of a parent solution to explore the fitness landscape.

Key principles:
1. ONE mutation at a time - don't combine multiple changes
2. Maintain CORRECTNESS - the mutation must still solve the problem
3. Be CREATIVE - try unusual approaches, not just obvious tweaks
4. DOCUMENT what you changed and why

Mutation strategies (pick ONE per invocation):
- Parameter tweaks: Change constants, thresholds, sizes
- Algorithm swap: Replace one algorithm with another
- Code golf tricks: Apply language-specific size optimizations
- Structural change: Reorganize loops, conditions, data flow
- Library swap: Use different built-in functions or approaches

Output format: Always return valid JSON with your results.
"""


def get_mutator_prompt(
    parent_file: str,
    parent_fitness: float,
    output_file: str,
    mode: str,
    generation: int,
    variant: str,
    mutation_hint: str | None = None,
) -> str:
    """Generate the prompt for a mutator agent."""

    mode_guidance = {
        "size": """
SIZE optimization mutations to try:
- Remove whitespace, shorten variable names
- Use list comprehensions instead of loops
- Combine statements with semicolons or commas
- Use operator tricks (x or y, x and y, -~x for x+1)
- Replace function definitions with lambdas
- Use string multiplication, slice tricks
- Eliminate unnecessary parentheses""",
        "perf": """
PERFORMANCE optimization mutations to try:
- Change algorithm complexity (sort method, search strategy)
- Add/remove caching or memoization
- Change data structure (list vs set vs dict)
- Vectorize operations (numpy, SIMD-friendly patterns)
- Reduce memory allocations
- Optimize hot loops (move invariants out, reduce calls)""",
        "ml": """
ML accuracy optimization mutations to try:
- Adjust hyperparameters (learning rate, regularization)
- Add/remove features
- Change model architecture (layers, neurons)
- Try different preprocessing
- Adjust class weights or sampling
- Try ensemble techniques""",
    }

    hint_section = ""
    if mutation_hint:
        hint_section = f"""
Suggested mutation direction: {mutation_hint}
(You may follow this hint or try something different if you have a better idea)"""

    return f"""Create ONE mutation of the parent solution.

Parent: {parent_file}
Parent fitness: {parent_fitness}
Generation: {generation}, Variant: {variant}
Mode: {mode}

{mode_guidance.get(mode, mode_guidance["size"])}
{hint_section}

Instructions:
1. Read the parent solution carefully
2. Choose ONE mutation strategy
3. Apply the mutation
4. Save to: {output_file}
5. Verify the mutation is still correct (quick test)

Return JSON:
{{
    "file": "{output_file}",
    "mutation_type": "<strategy used>",
    "description": "<what specifically changed>",
    "hypothesis": "<why this might improve fitness>"
}}

CRITICAL: The mutated solution must be COMPLETE and CORRECT. Test it before returning."""
