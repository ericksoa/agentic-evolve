"""Tests for reporter.py cleanup — no ANSI escape sequences."""

import pytest
from evolve_sdk.agents.reporter import ReporterAgent, report_to_operator


class MockMemory:
    """Mock EvolutionMemory for testing."""

    def __init__(self, messages=None):
        self._messages = messages or []

    def get_human_messages(self, limit=50):
        return self._messages


class TestDisplayMessageNoAnsi:
    """Verify _display_message produces no ANSI escape sequences."""

    def test_status_message_no_ansi(self, capsys):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "from_agent": "mutator",
            "message_type": "status",
            "priority": "info",
            "title": "Working on mutation",
            "content": "Some details",
            "generation": 3,
            "related_fitness": 15.5,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "[MUTATOR]" in captured.out
        assert "Working on mutation" in captured.out

    def test_milestone_message_no_ansi(self, capsys):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "from_agent": "runner",
            "message_type": "milestone",
            "priority": "important",
            "title": "New champion found!",
            "content": "Fitness improved by 50%",
            "generation": 5,
            "related_fitness": 25.0,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "New champion found!" in captured.out

    def test_warning_message_no_ansi(self, capsys):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "from_agent": "adversary",
            "message_type": "warning",
            "priority": "urgent",
            "title": "Suspicious fitness jump detected",
            "content": "",
            "generation": 4,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "!" in captured.out  # urgent priority marker
        assert "Suspicious" in captured.out

    def test_critical_priority_marker(self, capsys):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "from_agent": "system",
            "message_type": "error",
            "priority": "critical",
            "title": "System failure",
            "content": "",
            "generation": 0,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "!!" in captured.out  # critical priority marker

    def test_content_truncation(self, capsys):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "from_agent": "evaluator",
            "message_type": "result",
            "priority": "info",
            "title": "Evaluation complete",
            "content": "x" * 200,
            "generation": 1,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "..." in captured.out
        # Should not have full 200 chars
        assert "x" * 151 not in captured.out

    def test_no_agent_prefix(self, capsys):
        reporter = ReporterAgent(MockMemory(), show_agent_prefix=False)
        msg = {
            "from_agent": "mutator",
            "message_type": "status",
            "priority": "info",
            "title": "Test",
            "content": "",
            "generation": 0,
        }
        reporter._display_message(msg)
        captured = capsys.readouterr()
        assert "[MUTATOR]" not in captured.out
        assert "Test" in captured.out


class TestPriorityFiltering:
    """Verify priority filtering still works after cleanup."""

    def test_filters_low_priority(self):
        reporter = ReporterAgent(MockMemory(), min_priority="important")
        assert not reporter._should_display({
            "priority": "debug",
            "message_type": "status",
        })
        assert not reporter._should_display({
            "priority": "info",
            "message_type": "status",
        })
        assert reporter._should_display({
            "priority": "important",
            "message_type": "status",
        })
        assert reporter._should_display({
            "priority": "urgent",
            "message_type": "status",
        })

    def test_filters_quiet_types(self):
        reporter = ReporterAgent(MockMemory(), quiet_types=["status"])
        assert not reporter._should_display({
            "priority": "info",
            "message_type": "status",
        })
        assert reporter._should_display({
            "priority": "info",
            "message_type": "milestone",
        })


class TestDeduplication:
    """Verify deduplication still works after cleanup."""

    def test_message_id_generation(self):
        reporter = ReporterAgent(MockMemory())
        msg = {
            "timestamp": "2024-01-01T00:00:00",
            "from_agent": "mutator",
            "title": "Test",
        }
        msg_id = reporter._get_message_id(msg)
        assert "2024-01-01T00:00:00" in msg_id
        assert "mutator" in msg_id
        assert "Test" in msg_id

    def test_duplicate_messages_filtered(self):
        reporter = ReporterAgent(MockMemory())
        msg_id = "2024-01-01:mutator:Test"
        reporter._displayed_ids.add(msg_id)
        # Same ID should be in the set
        assert msg_id in reporter._displayed_ids


class TestSystemMessage:
    """Test _print_system_message without ANSI."""

    def test_system_message_no_ansi(self, capsys):
        reporter = ReporterAgent(MockMemory())
        reporter._print_system_message("Reporter started")
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "[reporter]" in captured.out
        assert "Reporter started" in captured.out

    def test_error_system_message(self, capsys):
        reporter = ReporterAgent(MockMemory())
        reporter._print_system_message("Something broke", is_error=True)
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "ERROR:" in captured.out


class TestReportToOperator:
    """Test the standalone report_to_operator function."""

    def test_no_ansi(self, capsys):
        report_to_operator("Test message", content="Details", from_agent="system")
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "[SYSTEM]" in captured.out
        assert "Test message" in captured.out
        assert "Details" in captured.out

    def test_no_content(self, capsys):
        report_to_operator("Just a title")
        captured = capsys.readouterr()
        assert "Just a title" in captured.out
