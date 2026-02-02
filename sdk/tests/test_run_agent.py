"""Tests for _run_agent message handling in the evolution runner."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Mock SDK types for testing without SDK installed
class MockTextBlock:
    def __init__(self, text):
        self.text = text


class MockToolUseBlock:
    def __init__(self, name, block_id="tool_1"):
        self.name = name
        self.id = block_id


class MockThinkingBlock:
    def __init__(self, thinking="..."):
        self.thinking = thinking


class MockAssistantMessage:
    __name__ = "AssistantMessage"

    def __init__(self, content):
        self.content = content

    @property
    def __class__(self):
        return type("AssistantMessage", (), {"__name__": "AssistantMessage"})


class MockStreamEvent:
    __name__ = "StreamEvent"

    def __init__(self, event):
        self.event = event

    @property
    def __class__(self):
        return type("StreamEvent", (), {"__name__": "StreamEvent"})


class MockResultMessage:
    __name__ = "ResultMessage"

    def __init__(self, result=None, is_error=False, num_turns=3,
                 total_cost_usd=0.01, structured_output=None):
        self.result = result
        self.is_error = is_error
        self.num_turns = num_turns
        self.total_cost_usd = total_cost_usd
        self.structured_output = structured_output

    @property
    def __class__(self):
        return type("ResultMessage", (), {"__name__": "ResultMessage"})


@pytest.fixture
def runner():
    """Create a minimal EvolutionRunner for testing _run_agent."""
    with patch.dict("sys.modules", {
        "claude_agent_sdk": MagicMock(),
    }):
        from evolve_sdk.runner import EvolutionRunner
        with patch.object(EvolutionRunner, "__init__", lambda self, *a, **kw: None):
            r = EvolutionRunner.__new__(EvolutionRunner)
            r.config = MagicMock()
            r.config.model = "test-model"
            r.config.trust = MagicMock()
            r.config.trust.enabled = False
            r.cwd = "/tmp"
            r.mutations_dir = MagicMock()
            r.work_dir = MagicMock()
            r.generation = 1
            return r


class TestRunAgentMessageHandling:
    """Test that _run_agent correctly processes different SDK message types."""

    @pytest.mark.asyncio
    async def test_extracts_text_from_assistant_message(self, runner, capsys):
        """AssistantMessage with TextBlock should be extracted to result."""
        messages = [
            MockAssistantMessage([MockTextBlock("hello world")]),
            MockResultMessage(result="hello world", num_turns=1),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                result = await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="test-agent",
                )

        # ResultMessage.result should be preferred
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_prints_phase_from_assistant_message(self, runner, capsys):
        """ToolUseBlock should print semantic phase to stdout."""
        messages = [
            MockAssistantMessage([MockToolUseBlock("Read")]),
            MockResultMessage(result="done", num_turns=1),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="test-agent",
                )

        captured = capsys.readouterr()
        assert "[test-agent] Analyzing..." in captured.out

    @pytest.mark.asyncio
    async def test_prints_agent_start_and_completion(self, runner, capsys):
        """Should print starting and done messages."""
        messages = [
            MockResultMessage(result="ok", num_turns=2, total_cost_usd=0.005),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="my-agent",
                )

        captured = capsys.readouterr()
        assert "[my-agent] Starting..." in captured.out
        assert "[my-agent] Done (2 turns)" in captured.out

    @pytest.mark.asyncio
    async def test_handles_error_result(self, runner, capsys):
        """ResultMessage with is_error should print ERROR status."""
        messages = [
            MockResultMessage(result="something broke", is_error=True, num_turns=1),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                result = await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="err-agent",
                )

        captured = capsys.readouterr()
        assert "[err-agent] ERROR" in captured.out

    @pytest.mark.asyncio
    async def test_prefers_structured_output(self, runner, capsys):
        """structured_output should be preferred over result text."""
        structured = {"fitness": 42.0, "valid": True}
        messages = [
            MockResultMessage(
                result="some text",
                structured_output=structured,
                num_turns=1,
            ),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                result = await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="struct-agent",
                )

        parsed = json.loads(result)
        assert parsed["fitness"] == 42.0

    @pytest.mark.asyncio
    async def test_handles_stream_event_tool_start(self, runner, capsys):
        """StreamEvent with content_block_start for tool_use should print phase."""
        messages = [
            MockStreamEvent({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Bash"},
            }),
            MockResultMessage(result="done", num_turns=1),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Bash"],
                    agent_name="stream-agent",
                )

        captured = capsys.readouterr()
        assert "[stream-agent] Exploring..." in captured.out

    @pytest.mark.asyncio
    async def test_phase_deduplication(self, runner, capsys):
        """Repeated same-phase tools should only print the phase once."""
        messages = [
            MockAssistantMessage([MockToolUseBlock("Read", "t1")]),
            MockAssistantMessage([MockToolUseBlock("Read", "t2")]),
            MockAssistantMessage([MockToolUseBlock("Glob", "t3")]),
            MockResultMessage(result="done", num_turns=3),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read", "Glob"],
                    agent_name="dedup-agent",
                )

        captured = capsys.readouterr()
        # "Analyzing..." should appear only once despite 3 tools in that phase
        assert captured.out.count("Analyzing...") == 1

    @pytest.mark.asyncio
    async def test_bash_after_write_shows_testing(self, runner, capsys):
        """Bash after Write should show 'Testing' phase, not 'Exploring'."""
        messages = [
            MockAssistantMessage([MockToolUseBlock("Read", "t1")]),
            MockAssistantMessage([MockToolUseBlock("Write", "t2")]),
            MockAssistantMessage([MockToolUseBlock("Bash", "t3")]),
            MockResultMessage(result="done", num_turns=3),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read", "Write", "Bash"],
                    agent_name="phase-agent",
                )

        captured = capsys.readouterr()
        assert "[phase-agent] Analyzing..." in captured.out
        assert "[phase-agent] Implementing..." in captured.out
        assert "[phase-agent] Testing..." in captured.out

    @pytest.mark.asyncio
    async def test_phase_no_bouncing(self, runner, capsys):
        """Alternating tool phases should not re-print already-seen phases."""
        messages = [
            MockAssistantMessage([MockToolUseBlock("Read", "t1")]),
            MockAssistantMessage([MockToolUseBlock("Bash", "t2")]),
            MockAssistantMessage([MockToolUseBlock("Read", "t3")]),
            MockAssistantMessage([MockToolUseBlock("Bash", "t4")]),
            MockAssistantMessage([MockToolUseBlock("Read", "t5")]),
            MockResultMessage(result="done", num_turns=5),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read", "Bash"],
                    agent_name="bounce-agent",
                )

        captured = capsys.readouterr()
        # Each phase should appear exactly once despite alternation
        assert captured.out.count("Analyzing...") == 1
        assert captured.out.count("Exploring...") == 1

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, runner, capsys):
        """Agent exception should return error JSON and print FAILED."""
        async def mock_query(**kwargs):
            raise RuntimeError("connection failed")
            yield  # Make it an async generator

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                result = await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="fail-agent",
                )

        captured = capsys.readouterr()
        assert "[fail-agent] FAILED:" in captured.out
        assert "error" in result

    @pytest.mark.asyncio
    async def test_raises_without_sdk(self, runner):
        """Should raise RuntimeError when SDK is not available."""
        with patch("evolve_sdk.runner.SDK_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Claude Agent SDK not installed"):
                await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                )

    @pytest.mark.asyncio
    async def test_handles_plain_string_messages(self, runner, capsys):
        """Plain string messages should be accumulated."""
        messages = [
            "plain text response",
            MockResultMessage(result=None, num_turns=1),
        ]

        async def mock_query(**kwargs):
            for m in messages:
                yield m

        with patch("evolve_sdk.runner.query", mock_query):
            with patch("evolve_sdk.runner.SDK_AVAILABLE", True):
                result = await runner._run_agent(
                    system="test", prompt="test", tools=["Read"],
                    agent_name="str-agent",
                )

        assert "plain text response" in result
