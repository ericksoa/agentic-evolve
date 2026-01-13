"""Debugger agent - diagnoses failed mutations to reduce wasted generations.

The Debugger agent is triggered when a mutation fails (crash, error, timeout).
Its role is to:
1. Analyze the error to identify root cause
2. Compare with parent to find the breaking change
3. Suggest a minimal fix
4. Categorize the failure for learning

This reduces wasted API calls by helping future mutators avoid similar mistakes.
"""

from ..skills import get_mode_guidance

DEBUGGER_SYSTEM = """You are a debugging specialist for evolutionary algorithm discovery.

Your role: Diagnose WHY a mutation failed and help prevent similar failures.

## Core Responsibilities

1. **Error Analysis**: Parse the error message/traceback to understand what went wrong
2. **Diff Analysis**: Compare the failed mutation with its working parent to find the breaking change
3. **Root Cause**: Identify the actual cause, not just the symptom
4. **Fix Suggestion**: Propose a minimal fix (not a rewrite)
5. **Pattern Recognition**: Categorize the failure type for future learning

## Failure Categories

- **syntax_error**: Invalid Python/code syntax
- **import_error**: Missing module or import issue
- **type_error**: Wrong type passed to function/operator
- **index_error**: Array/list index out of bounds
- **key_error**: Dictionary key not found
- **value_error**: Invalid value for operation
- **attribute_error**: Object doesn't have attribute
- **runtime_error**: General runtime failure
- **timeout**: Execution took too long
- **memory_error**: Out of memory
- **logic_error**: Code runs but produces wrong results
- **boundary_condition**: Off-by-one or edge case failure

## Analysis Strategy

1. First, read the error message carefully - it usually points to the exact line
2. Read the failed mutation code around that line
3. If parent is provided, diff against parent to see what changed
4. Identify which change caused the failure
5. Suggest the minimal fix that would restore correctness

## Output Format

Always return valid JSON:
{
    "failed_mutation": "<file path>",
    "error_type": "<Python exception type>",
    "error_line": <line number if known>,
    "traceback_summary": "<key part of traceback>",
    "root_cause": "<explanation of what went wrong>",
    "breaking_change": "<what changed from parent that broke it>",
    "suggested_fix": "<minimal code change to fix>",
    "failure_category": "<category from list above>",
    "preventable": true/false,
    "lesson": "<advice for future mutators to avoid this>"
}

Be specific and actionable. Vague analysis like "something is wrong" is not helpful.
"""


def get_debugger_prompt(
    failed_file: str,
    error_message: str,
    error_traceback: str | None = None,
    parent_file: str | None = None,
    mutation_type: str | None = None,
    mutation_description: str | None = None,
    mode: str = "perf",
    generation: int = 0,
) -> str:
    """Generate the prompt for a debugger agent to analyze a failed mutation.

    Args:
        failed_file: Path to the failed mutation file
        error_message: The error message (exception message)
        error_traceback: Full traceback if available
        parent_file: Path to parent solution (known working) for comparison
        mutation_type: What kind of mutation was attempted
        mutation_description: Description of what the mutation tried to do
        mode: Evolution mode (size, perf, ml)
        generation: Current generation number
    """

    traceback_section = ""
    if error_traceback:
        # Truncate very long tracebacks but keep the important parts
        tb_lines = error_traceback.strip().split("\n")
        if len(tb_lines) > 30:
            # Keep first 10 and last 15 lines
            tb_lines = tb_lines[:10] + ["... (truncated) ..."] + tb_lines[-15:]
        traceback_section = f"""
## Full Traceback
```
{chr(10).join(tb_lines)}
```
"""

    parent_section = ""
    if parent_file:
        parent_section = f"""
## Parent Solution (Known Working)
File: {parent_file}

IMPORTANT: Read and diff against the parent to understand what change broke the code.
"""

    mutation_section = ""
    if mutation_type or mutation_description:
        mutation_section = f"""
## Mutation Context
- Type: {mutation_type or 'unknown'}
- Description: {mutation_description or 'not provided'}
"""

    # Get mode-specific guidance if available
    skill_guidance = get_mode_guidance(mode, "debugger")
    mode_section = ""
    if skill_guidance:
        mode_section = f"""
## Mode-Specific Guidance ({mode})
{skill_guidance}
"""

    return f"""## Debug Failed Mutation - Generation {generation}

A mutation failed during evolution. Your job is to diagnose the failure and
help prevent similar issues in future mutations.

## Failed Mutation
File: {failed_file}

## Error
{error_message}
{traceback_section}
{parent_section}
{mutation_section}
{mode_section}

## Your Task

1. **Read the failed mutation** - examine the code at {failed_file}
2. **Parse the error** - understand what exception occurred and where
3. **If parent exists, diff against parent** - find what changed
4. **Identify root cause** - not just the symptom, but why it failed
5. **Suggest a fix** - minimal change to restore correctness
6. **Categorize** - what type of failure is this?
7. **Extract lesson** - what should future mutators avoid?

## Return JSON

After your analysis, return:
```json
{{
    "failed_mutation": "{failed_file}",
    "error_type": "<exception type>",
    "error_line": <line number or null>,
    "traceback_summary": "<key error info>",
    "root_cause": "<why it failed>",
    "breaking_change": "<what change from parent caused this>",
    "suggested_fix": "<minimal fix>",
    "failure_category": "<category>",
    "preventable": true/false,
    "lesson": "<advice for future mutators>"
}}
```

Categories: syntax_error, import_error, type_error, index_error, key_error,
value_error, attribute_error, runtime_error, timeout, memory_error, logic_error,
boundary_condition

IMPORTANT: Read the actual code files before making your diagnosis. Be specific."""


def get_debugger_summary_prompt(
    failures: list[dict],
    generation: int,
) -> str:
    """Generate prompt to summarize multiple failures in a generation.

    Used when multiple mutations fail in the same generation to find patterns.

    Args:
        failures: List of failure records with error info
        generation: Current generation number
    """

    failures_text = []
    for i, f in enumerate(failures, 1):
        failures_text.append(f"""
### Failure {i}
- File: {f.get('file', 'unknown')}
- Error: {f.get('error_type', 'unknown')}: {f.get('error_message', '')}
- Category: {f.get('failure_category', 'unknown')}
- Root cause: {f.get('root_cause', 'not analyzed')}
""")

    return f"""## Failure Pattern Analysis - Generation {generation}

Multiple mutations failed this generation. Analyze for patterns.

## Failures
{''.join(failures_text)}

## Your Task

Look for patterns across these failures:
1. Are they the same type of error?
2. Are they related to the same code region?
3. Is there a common root cause?
4. What should mutators avoid in the next generation?

## Return JSON

```json
{{
    "total_failures": {len(failures)},
    "common_pattern": "<pattern if found, else null>",
    "shared_root_cause": "<if applicable>",
    "affected_code_region": "<if applicable>",
    "mutation_guidance": "<what to avoid next generation>",
    "severity": "low" | "medium" | "high",
    "recommendation": "<actionable advice>"
}}
```
"""
