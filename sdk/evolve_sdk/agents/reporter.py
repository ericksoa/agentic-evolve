"""
Reporter Agent - Surfaces messages from the evolution memory to the human operator.

The reporter runs as a background task during evolution, polling the message
queue and displaying important updates in real-time. This ensures the operator
stays informed even when subagents are running in parallel.

Design:
- Polls memory.get_human_messages() at configurable intervals
- Tracks displayed message IDs to avoid duplicates
- Filters by priority level (default: important+)
- Formats messages with clean text for terminal visibility
- Can be paused/resumed during critical operations

Usage:
    reporter = ReporterAgent(memory, poll_interval=2.0)
    await reporter.start()
    # ... evolution runs ...
    await reporter.stop()
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory import EvolutionMemory


class ReporterAgent:
    """
    Background agent that surfaces messages to the human operator.

    Runs as an async task, polling the message queue and printing
    important updates to the terminal.
    """

    # Priority levels in order (for filtering)
    PRIORITY_LEVELS = ["debug", "info", "important", "urgent", "critical"]

    def __init__(
        self,
        memory: "EvolutionMemory",
        poll_interval: float = 2.0,
        min_priority: str = "info",
        show_agent_prefix: bool = True,
        quiet_types: list[str] | None = None,
        quiet_agents: list[str] | None = None,
    ):
        """
        Initialize the reporter agent.

        Args:
            memory: EvolutionMemory instance to poll for messages
            poll_interval: Seconds between polls (default: 2.0)
            min_priority: Minimum priority level to display (default: info)
            show_agent_prefix: Whether to show agent name prefix
            quiet_types: Message types to suppress (e.g., ["status"])
            quiet_agents: Agent names to suppress (e.g., ["runner"])
        """
        self.memory = memory
        self.poll_interval = poll_interval
        self.min_priority = min_priority
        self.show_agent_prefix = show_agent_prefix
        self.quiet_types = quiet_types or []
        self.quiet_agents = quiet_agents or []

        # Track displayed messages to avoid duplicates
        self._displayed_ids: set[str] = set()
        self._last_poll_time: str = ""

        # Task management
        self._task: asyncio.Task | None = None
        self._running = False
        self._paused = False

        # Stats
        self.messages_displayed = 0
        self.messages_filtered = 0

    async def start(self):
        """Start the reporter background task."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._poll_loop())

        # Announce startup
        self._print_system_message("Reporter agent started - monitoring message queue")

    async def stop(self):
        """Stop the reporter background task."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Announce shutdown
        self._print_system_message(
            f"Reporter agent stopped - displayed {self.messages_displayed} messages, "
            f"filtered {self.messages_filtered}"
        )

    def pause(self):
        """Pause message display (e.g., during human escalation prompts)."""
        self._paused = True

    def resume(self):
        """Resume message display."""
        self._paused = False

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                if not self._paused:
                    await self._check_messages()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't let errors crash the reporter
                self._print_system_message(f"Reporter error: {e}", is_error=True)
                await asyncio.sleep(self.poll_interval * 2)

    async def _check_messages(self):
        """Check for new messages and display them."""
        try:
            # Get recent messages for humans
            messages = self.memory.get_human_messages(limit=50)

            for msg in messages:
                msg_id = self._get_message_id(msg)

                # Skip already displayed
                if msg_id in self._displayed_ids:
                    continue

                # Check filters
                if not self._should_display(msg):
                    self._displayed_ids.add(msg_id)
                    self.messages_filtered += 1
                    continue

                # Display the message
                self._display_message(msg)
                self._displayed_ids.add(msg_id)
                self.messages_displayed += 1

        except Exception:
            # Silently ignore errors during polling
            pass

    def _get_message_id(self, msg: dict) -> str:
        """Generate a unique ID for a message to track duplicates."""
        # Use timestamp + agent + title as unique key
        return f"{msg.get('timestamp', '')}:{msg.get('from_agent', '')}:{msg.get('title', '')}"

    def _should_display(self, msg: dict) -> bool:
        """Check if a message should be displayed based on filters."""
        # Priority filter
        msg_priority = msg.get("priority", "info")
        if self.PRIORITY_LEVELS.index(msg_priority) < self.PRIORITY_LEVELS.index(self.min_priority):
            return False

        # Type filter
        msg_type = msg.get("message_type", "status")
        if msg_type in self.quiet_types:
            return False

        # Agent filter
        from_agent = msg.get("from_agent", "")
        if from_agent in self.quiet_agents:
            return False

        return True

    def _display_message(self, msg: dict):
        """Format and print a message to the terminal."""
        from_agent = msg.get("from_agent", "unknown")
        msg_type = msg.get("message_type", "status")
        priority = msg.get("priority", "info")
        title = msg.get("title", "")
        content = msg.get("content", "")
        generation = msg.get("generation", 0)
        fitness = msg.get("related_fitness")

        # Priority indicator
        priority_marker = ""
        if priority == "urgent":
            priority_marker = "! "
        elif priority == "critical":
            priority_marker = "!! "

        # Build the output
        lines = []

        # Header line
        if self.show_agent_prefix:
            header = f"\n  [{from_agent.upper()}] {priority_marker}{title}"
        else:
            header = f"\n  {priority_marker}{title}"
        lines.append(header)

        # Content (if different from title and not too long)
        if content and content != title:
            # Truncate long content
            content_preview = content[:150]
            if len(content) > 150:
                content_preview += "..."
            lines.append(f"     {content_preview}")

        # Context line (generation, fitness if relevant)
        context_parts = []
        if generation > 0:
            context_parts.append(f"Gen {generation}")
        if fitness is not None:
            context_parts.append(f"Fitness: {fitness:.2f}")

        if context_parts:
            lines.append(f"     [{' | '.join(context_parts)}]")

        # Print all lines
        for line in lines:
            print(line)

    def _print_system_message(self, text: str, is_error: bool = False):
        """Print a system message from the reporter itself."""
        prefix = "ERROR: " if is_error else ""
        print(f"\n  [reporter] {prefix}{text}")

    def inject_message(self, title: str, content: str = "", priority: str = "info"):
        """
        Manually inject a message to be displayed.

        Useful for external systems to send notifications through the reporter.
        """
        msg = {
            "from_agent": "reporter",
            "message_type": "status",
            "priority": priority,
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "generation": 0,
        }
        self._display_message(msg)


# Convenience function for one-shot message display
def report_to_operator(
    title: str,
    content: str = "",
    from_agent: str = "system",
    message_type: str = "status",
    priority: str = "info",
):
    """
    Display a one-shot message to the operator.

    Use this when you don't have a running ReporterAgent but need
    to surface something to the human.
    """
    print(f"\n  [{from_agent.upper()}] {title}")
    if content and content != title:
        print(f"     {content[:150]}")
