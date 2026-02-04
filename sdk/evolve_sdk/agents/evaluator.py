"""Evaluator agent - measures fitness of solutions."""

from ..skills import get_mode_guidance

EVALUATOR_SYSTEM = """You are an evaluation specialist for evolutionary algorithm discovery.

Your role: Objectively measure the fitness of solutions by running the provided benchmark command.

Key principles:
1. USE THE PROVIDED BENCHMARK COMMAND - this is mandatory
2. Parse the JSON output from the benchmark to get fitness scores
3. CONSISTENT measurement - same method for all solutions
4. Return structured JSON results

Output format: Always return valid JSON with your results.
"""

# Fallback guidance if no skill file is found
_DEFAULT_MODE_GUIDANCE = {
    "size": """
SIZE evaluation:
1. First verify correctness against test cases
2. If correct, measure size in BYTES: wc -c <file>
3. Fitness = 1000 - bytes (higher is better, invalid = 0)

Commands:
- Size: wc -c <file>
- Test: python3 <file> (or run specific test)""",
    "perf": """
PERFORMANCE evaluation:
1. First verify correctness against test cases
2. If correct, run benchmark multiple times for statistical validity
3. Fitness = ops/second or 1/latency (higher is better, invalid = 0)

Commands:
- Benchmark: time python3 <file> or custom benchmark
- Run multiple iterations and average""",
    "ml": """
ML evaluation:
1. Train/load the model
2. Evaluate on validation set
3. Fitness = primary metric (F1, accuracy, etc.)

Commands:
- Train: python3 <file> --train
- Evaluate: python3 <file> --eval""",
}


def get_evaluator_prompt(
    solution_files: list[str],
    mode: str,
    test_cases: str | None = None,
    benchmark_command: str | None = None,
) -> str:
    """Generate the prompt for an evaluator agent."""

    files_list = "\n".join(f"  - {f}" for f in solution_files)

    # If we have a custom benchmark command, use it exclusively
    if benchmark_command:
        return f"""Evaluate the fitness of these solutions using the provided benchmark command.

Solutions to evaluate:
{files_list}

BENCHMARK COMMAND (YOU MUST USE THIS):
{benchmark_command}

For each solution file, replace {{solution}} with the actual file path and run the command.
For example: {benchmark_command.replace('{solution}', solution_files[0]) if solution_files else benchmark_command}

The command outputs JSON with these fields:
- "valid": boolean - whether the solution is correct
- "fitness": number - the fitness score (higher is better)
- "error": string - error message if invalid

INSTRUCTIONS:
1. For EACH solution file listed above:
   a. Run the benchmark command with that file
   b. Parse the JSON output
   c. Record the fitness score
2. After evaluating ALL solutions, return the combined results

Return JSON:
{{
    "evaluations": [
        {{
            "file": "<path>",
            "valid": true/false,
            "fitness": <score from benchmark>,
            "error": "<if invalid, why>",
            "notes": "<any observations>"
        }},
        ...
    ],
    "best": {{"file": "<path>", "fitness": <score>}},
    "ranking": ["<file1>", "<file2>", ...]
}}

CRITICAL: You MUST run the benchmark command for each solution. Do not skip this step."""

    # Fallback to generic evaluation - try skill guidance first
    skill_guidance = get_mode_guidance(mode, "evaluator")
    if skill_guidance:
        mode_section = f"""## Mode Guidance (from /evolve-{mode} skill)
{skill_guidance}"""
    else:
        mode_section = _DEFAULT_MODE_GUIDANCE.get(mode, _DEFAULT_MODE_GUIDANCE["size"])

    test_section = ""
    if test_cases:
        test_section = f"""
Test cases to verify:
{test_cases}"""

    return f"""Evaluate the fitness of these solutions.

Solutions to evaluate:
{files_list}

Mode: {mode}
{mode_section}
{test_section}

Instructions:
1. For EACH solution:
   a. Verify correctness (run tests)
   b. If correct, measure fitness
   c. If incorrect, fitness = 0
2. Return results for ALL solutions

Return JSON:
{{
    "evaluations": [
        {{
            "file": "<path>",
            "valid": true/false,
            "fitness": <score>,
            "error": "<if invalid, why>",
            "notes": "<observations about this solution>"
        }},
        ...
    ],
    "best": {{"file": "<path>", "fitness": <score>}},
    "ranking": ["<file1>", "<file2>", ...]
}}

IMPORTANT: Evaluate ALL solutions listed. Be thorough but efficient."""


def get_workspace_evaluator_prompt(
    workspace_dirs: list[str],
    test_command: str,
) -> str:
    """Generate prompt for workspace-mode evaluation (directory-level evolution).

    Runs pytest against workspace directories and returns granular fitness.

    Args:
        workspace_dirs: List of workspace directory paths to evaluate
        test_command: Command to run tests (with {workspace} placeholder)
    """
    dirs_list = "\n".join(f"  - {d}" for d in workspace_dirs)

    return f"""Evaluate the fitness of these workspace solutions by running tests.

Workspaces to evaluate:
{dirs_list}

TEST COMMAND (YOU MUST USE THIS):
{test_command}

For each workspace, replace {{workspace}} with the actual workspace path and run the command.

## Instructions
1. For EACH workspace listed above:
   a. Run the test command with that workspace path
   b. Parse the test output to count passed/failed tests
   c. Calculate fitness = tests_passed / tests_total
2. After evaluating ALL workspaces, return the combined results

Return JSON:
{{
    "evaluations": [
        {{
            "file": "<workspace path>",
            "valid": true,
            "fitness": <tests_passed / tests_total>,
            "tests_passed": <number>,
            "tests_total": <number>,
            "failing_tests": [<list of failing test names>],
            "error": "<if evaluation failed, why>",
            "notes": "<any observations>"
        }},
        ...
    ],
    "best": {{"file": "<workspace path>", "fitness": <score>}},
    "ranking": ["<workspace1>", "<workspace2>", ...]
}}

CRITICAL: You MUST run the test command for each workspace. Do not skip this step."""
