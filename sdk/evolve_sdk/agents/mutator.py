"""Mutator agent - creates mutations of existing solutions."""

from ..skills import get_mode_guidance

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

# Fallback guidance if no skill file is found
_DEFAULT_MODE_GUIDANCE = {
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


def get_mutator_prompt(
    parent_file: str,
    parent_fitness: float,
    output_file: str,
    mode: str,
    generation: int,
    variant: str,
    mutation_hint: str | None = None,
    memory_context: str | None = None,
) -> str:
    """Generate the prompt for a mutator agent.

    Args:
        parent_file: Path to parent solution
        parent_fitness: Parent's fitness score
        output_file: Path to save mutation
        mode: Evolution mode (size, perf, ml)
        generation: Current generation number
        variant: Mutation variant (a, b, c, etc.)
        mutation_hint: Optional hint for mutation direction
        memory_context: Optional context from evolution memory (similar mutations, failed approaches)
    """

    # Try to get guidance from skill file first, fall back to defaults
    skill_guidance = get_mode_guidance(mode, "mutator")
    if skill_guidance:
        mode_section = f"""## Mode Guidance (from /evolve-{mode} skill)
{skill_guidance}"""
    else:
        mode_section = _DEFAULT_MODE_GUIDANCE.get(mode, _DEFAULT_MODE_GUIDANCE["size"])

    hint_section = ""
    if mutation_hint:
        hint_section = f"""
Suggested mutation direction: {mutation_hint}
(You may follow this hint or try something different if you have a better idea)"""

    memory_section = ""
    if memory_context:
        memory_section = f"""
## Memory Context (from previous evolution runs)
{memory_context}
"""

    return f"""Create ONE mutation of the parent solution.

Parent: {parent_file}
Parent fitness: {parent_fitness}
Generation: {generation}, Variant: {variant}
Mode: {mode}

{mode_section}
{hint_section}
{memory_section}
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


def get_workspace_mutator_prompt(
    workspace_dir: str,
    failing_tests: list[str],
    test_output: str,
    parent_fitness: float,
    output_workspace: str,
    generation: int,
    variant: str,
    memory_context: str | None = None,
) -> str:
    """Generate prompt for workspace-mode mutation (directory-level evolution).

    Instead of mutating a single file, the agent modifies files within
    a workspace directory to fix failing tests.

    Args:
        workspace_dir: Path to parent workspace directory
        failing_tests: List of test names that are failing
        test_output: Last pytest output (truncated)
        parent_fitness: Parent's fitness score (tests_passed / tests_total)
        output_workspace: Path to save mutated workspace
        generation: Current generation number
        variant: Mutation variant label
        memory_context: Optional context from evolution memory
    """
    failing_list = "\n".join(f"  - {t}" for t in failing_tests[:20])

    # Truncate test output
    test_snippet = test_output[-1500:] if len(test_output) > 1500 else test_output

    memory_section = ""
    if memory_context:
        memory_section = f"""
## Memory Context (from previous evolution runs)
{memory_context}
"""

    return f"""Fix failing tests in a multi-file Python codebase.

You are working in: {output_workspace}
(This is a copy of the parent workspace. Modify files in place.)

Parent fitness: {parent_fitness:.4f}
Generation: {generation}, Variant: {variant}

## Failing Tests
{failing_list}

## Last Test Output
```
{test_snippet}
```
{memory_section}
## Instructions
1. Read the failing test assertions to understand what the tests expect
2. Explore the workspace to understand the current state of the code
3. Use Grep to find ALL references to functions/classes being modified
4. Make targeted changes to fix as many failing tests as possible
5. Do NOT modify test files — only modify source code

## Strategy
- Focus on ONE or TWO specific failing tests per mutation
- Search exhaustively for all references before making changes
- Preserve all passing tests (don't break what works)
- Make minimal, targeted changes

Return JSON:
{{
    "mutation_type": "<strategy used>",
    "description": "<what specifically changed>",
    "hypothesis": "<why this might fix the failing tests>",
    "targeted_tests": [<list of test names targeted>]
}}

CRITICAL: The mutation must preserve all currently passing tests."""
