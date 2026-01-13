"""Tests for the Debugger agent prompt generation."""

import pytest
from evolve_sdk.agents.debugger import (
    DEBUGGER_SYSTEM,
    get_debugger_prompt,
    get_debugger_summary_prompt,
)


class TestDebuggerSystem:
    """Tests for the DEBUGGER_SYSTEM prompt."""

    def test_system_prompt_exists(self):
        """System prompt should be a non-empty string."""
        assert isinstance(DEBUGGER_SYSTEM, str)
        assert len(DEBUGGER_SYSTEM) > 100

    def test_system_prompt_contains_failure_categories(self):
        """System prompt should document all failure categories."""
        categories = [
            "syntax_error",
            "import_error",
            "type_error",
            "index_error",
            "key_error",
            "value_error",
            "attribute_error",
            "runtime_error",
            "timeout",
            "memory_error",
            "logic_error",
            "boundary_condition",
        ]
        for category in categories:
            assert category in DEBUGGER_SYSTEM, f"Missing category: {category}"

    def test_system_prompt_contains_json_format(self):
        """System prompt should document the expected JSON output format."""
        required_fields = [
            "failed_mutation",
            "error_type",
            "error_line",
            "root_cause",
            "suggested_fix",
            "failure_category",
            "lesson",
        ]
        for field in required_fields:
            assert field in DEBUGGER_SYSTEM, f"Missing JSON field: {field}"


class TestGetDebuggerPrompt:
    """Tests for the get_debugger_prompt function."""

    def test_minimal_prompt(self):
        """Should generate prompt with just required parameters."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError: list index out of range",
        )

        assert isinstance(prompt, str)
        assert "/path/to/gen1a.py" in prompt
        assert "IndexError: list index out of range" in prompt
        assert "Debug Failed Mutation" in prompt

    def test_prompt_with_traceback(self):
        """Should include traceback when provided."""
        traceback = """Traceback (most recent call last):
  File "gen1a.py", line 42, in solve
    result = arr[n]
IndexError: list index out of range"""

        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError: list index out of range",
            error_traceback=traceback,
        )

        assert "Full Traceback" in prompt
        assert "line 42" in prompt
        assert "arr[n]" in prompt

    def test_prompt_with_parent(self):
        """Should include parent file when provided."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError: list index out of range",
            parent_file="/path/to/gen0_a.py",
        )

        assert "Parent Solution" in prompt
        assert "/path/to/gen0_a.py" in prompt
        assert "diff against the parent" in prompt.lower()

    def test_prompt_with_mutation_context(self):
        """Should include mutation context when provided."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError: list index out of range",
            mutation_type="parameter_tweak",
            mutation_description="Changed loop bound from n to n+1",
        )

        assert "Mutation Context" in prompt
        assert "parameter_tweak" in prompt
        assert "loop bound" in prompt

    def test_prompt_with_all_parameters(self):
        """Should handle all parameters together."""
        traceback = """Traceback (most recent call last):
  File "gen1a.py", line 42, in solve
    result = arr[n]
IndexError: list index out of range"""

        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError: list index out of range",
            error_traceback=traceback,
            parent_file="/path/to/gen0_a.py",
            mutation_type="parameter_tweak",
            mutation_description="Changed loop bound from n to n+1",
            mode="perf",
            generation=5,
        )

        assert "Generation 5" in prompt
        assert "/path/to/gen1a.py" in prompt
        assert "/path/to/gen0_a.py" in prompt
        assert "parameter_tweak" in prompt
        assert "line 42" in prompt

    def test_prompt_contains_json_template(self):
        """Prompt should include the expected JSON output template."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="IndexError",
        )

        assert '"failed_mutation"' in prompt
        assert '"error_type"' in prompt
        assert '"root_cause"' in prompt
        assert '"suggested_fix"' in prompt
        assert '"failure_category"' in prompt
        assert '"lesson"' in prompt

    def test_long_traceback_truncation(self):
        """Very long tracebacks should be truncated."""
        # Create a traceback with 50 lines
        long_traceback = "\n".join([f"  Line {i}: some code" for i in range(50)])

        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="Error",
            error_traceback=long_traceback,
        )

        # Should contain truncation marker
        assert "(truncated)" in prompt

    def test_prompt_includes_task_instructions(self):
        """Prompt should include clear task instructions."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="Error",
        )

        assert "Read the failed mutation" in prompt
        assert "Parse the error" in prompt
        assert "Identify root cause" in prompt
        assert "Suggest a fix" in prompt


class TestGetDebuggerSummaryPrompt:
    """Tests for the get_debugger_summary_prompt function."""

    def test_summary_with_single_failure(self):
        """Should handle a single failure."""
        failures = [
            {
                "file": "gen1a.py",
                "error_type": "IndexError",
                "error_message": "list index out of range",
                "failure_category": "index_error",
                "root_cause": "Off-by-one in loop",
            }
        ]

        prompt = get_debugger_summary_prompt(failures, generation=3)

        assert "Generation 3" in prompt
        assert "gen1a.py" in prompt
        assert "IndexError" in prompt
        assert "index_error" in prompt

    def test_summary_with_multiple_failures(self):
        """Should handle multiple failures."""
        failures = [
            {
                "file": "gen1a.py",
                "error_type": "IndexError",
                "error_message": "list index out of range",
                "failure_category": "index_error",
            },
            {
                "file": "gen1b.py",
                "error_type": "TypeError",
                "error_message": "unsupported operand type",
                "failure_category": "type_error",
            },
            {
                "file": "gen1c.py",
                "error_type": "IndexError",
                "error_message": "tuple index out of range",
                "failure_category": "index_error",
            },
        ]

        prompt = get_debugger_summary_prompt(failures, generation=3)

        assert "Failure 1" in prompt
        assert "Failure 2" in prompt
        assert "Failure 3" in prompt
        assert "gen1a.py" in prompt
        assert "gen1b.py" in prompt
        assert "gen1c.py" in prompt
        assert '"total_failures": 3' in prompt

    def test_summary_contains_pattern_analysis_request(self):
        """Summary prompt should ask for pattern analysis."""
        failures = [
            {"file": "gen1a.py", "error_type": "Error", "error_message": "msg"},
        ]

        prompt = get_debugger_summary_prompt(failures, generation=1)

        assert "pattern" in prompt.lower()
        assert "common" in prompt.lower()
        assert "avoid" in prompt.lower()

    def test_summary_json_template(self):
        """Summary prompt should include expected JSON template."""
        failures = [
            {"file": "gen1a.py", "error_type": "Error", "error_message": "msg"},
        ]

        prompt = get_debugger_summary_prompt(failures, generation=1)

        assert '"common_pattern"' in prompt
        assert '"shared_root_cause"' in prompt
        assert '"mutation_guidance"' in prompt
        assert '"severity"' in prompt


class TestDebuggerIntegration:
    """Integration tests for debugger with other agent patterns."""

    def test_debugger_output_matches_memory_schema(self):
        """Debugger output fields should be compatible with memory storage."""
        # The debugger output should include fields that can be stored in memory
        # as a 'failed_mutation' frame type
        expected_fields = [
            "failed_mutation",  # file path
            "error_type",  # for categorization
            "failure_category",  # for pattern matching
            "lesson",  # for future mutator guidance
        ]

        for field in expected_fields:
            assert field in DEBUGGER_SYSTEM

    def test_prompt_references_file_paths(self):
        """Prompts should use file paths that can be read by the agent."""
        prompt = get_debugger_prompt(
            failed_file="/path/to/gen1a.py",
            error_message="Error",
            parent_file="/path/to/gen0_a.py",
        )

        # Should instruct to read actual files
        assert "Read" in prompt
        assert "/path/to/gen1a.py" in prompt
        assert "/path/to/gen0_a.py" in prompt
