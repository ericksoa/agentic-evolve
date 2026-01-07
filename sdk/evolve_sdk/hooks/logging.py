"""Logging hooks - track all tool usage during evolution."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Global log file path (set by runner)
_log_file: Path | None = None
_generation: int = 0


def set_log_context(log_file: Path, generation: int):
    """Set the logging context."""
    global _log_file, _generation
    _log_file = log_file
    _generation = generation


async def log_tool_use(
    input_data: dict[str, Any],
    tool_use_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Log all tool usage for analysis and debugging.

    Logs:
    - Tool name and inputs
    - Timestamp
    - Generation number
    - Success/failure (for PostToolUse)

    Args:
        input_data: Hook input containing tool info
        tool_use_id: ID of the tool call
        context: Additional context

    Returns:
        Empty dict (logging hook never blocks)
    """
    global _log_file, _generation

    if _log_file is None:
        return {}

    event_name = input_data.get("hook_event_name", "")
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "generation": _generation,
        "event": event_name,
        "tool": tool_name,
        "tool_use_id": tool_use_id,
    }

    if event_name == "PreToolUse":
        # Log what's about to happen
        log_entry["action"] = "invoke"
        # Truncate large inputs
        if tool_name in ("Write", "Edit"):
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            log_entry["input"] = {
                "file_path": tool_input.get("file_path"),
                "content_length": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
            }
        else:
            log_entry["input"] = tool_input

    elif event_name == "PostToolUse":
        # Log result
        log_entry["action"] = "complete"
        tool_response = input_data.get("tool_response", {})
        if isinstance(tool_response, dict):
            # Truncate large outputs
            output = str(tool_response.get("output", ""))
            log_entry["output_length"] = len(output)
            log_entry["output_preview"] = output[:500] + "..." if len(output) > 500 else output
            log_entry["success"] = "error" not in output.lower()
        else:
            log_entry["output"] = str(tool_response)[:500]

    # Append to log file
    try:
        with open(_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't fail evolution due to logging errors

    return {}


def create_logging_hook(log_file: Path, generation: int = 0):
    """
    Create a logging hook with specific file target.

    Args:
        log_file: Path to write logs
        generation: Current generation number

    Returns:
        Configured logging hook function
    """
    set_log_context(log_file, generation)
    return log_tool_use
