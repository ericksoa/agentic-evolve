"""Validation hooks - block unsafe mutations before execution."""

from typing import Any


async def validate_mutation(
    input_data: dict[str, Any],
    tool_use_id: str,
    context: dict[str, Any],
    blocked_patterns: list[str] | None = None,
    allowed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate mutations before they execute.

    Blocks:
    - Dangerous imports (os.system, subprocess, eval, etc.)
    - Writes outside allowed directories
    - Suspicious patterns

    Args:
        input_data: Hook input containing tool info
        tool_use_id: ID of the tool call
        context: Additional context
        blocked_patterns: List of patterns to block in content
        allowed_paths: List of path prefixes that are allowed

    Returns:
        Hook response (empty dict to allow, or deny decision)
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return {}

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Default blocked patterns
    if blocked_patterns is None:
        blocked_patterns = [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "urllib",
            "requests.get",
            "requests.post",
            "socket",
            "shutil.rmtree",
            "os.remove",
        ]

    # Check Write/Edit tools for dangerous content
    if tool_name in ("Write", "Edit"):
        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        file_path = tool_input.get("file_path", "")

        # Check for blocked patterns
        for pattern in blocked_patterns:
            if pattern in content:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Blocked: '{pattern}' not allowed in evolved solutions",
                    }
                }

        # Check path restrictions
        if allowed_paths:
            path_allowed = any(file_path.startswith(p) for p in allowed_paths)
            if not path_allowed:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Blocked: writes only allowed in {allowed_paths}",
                    }
                }

    # Check Bash for dangerous commands
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        dangerous_commands = [
            "rm -rf",
            "rm -r /",
            "sudo",
            "chmod 777",
            "> /dev/",
            "mkfs",
            "dd if=",
            ":(){ :|:",  # Fork bomb
        ]
        for dangerous in dangerous_commands:
            if dangerous in command:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Blocked: dangerous command pattern '{dangerous}'",
                    }
                }

    return {}


def create_validation_hook(
    blocked_patterns: list[str] | None = None,
    allowed_paths: list[str] | None = None,
):
    """
    Create a validation hook with custom configuration.

    Args:
        blocked_patterns: Patterns to block in code
        allowed_paths: Allowed write paths

    Returns:
        Configured validation hook function
    """

    async def hook(input_data: dict, tool_use_id: str, context: dict) -> dict:
        return await validate_mutation(
            input_data,
            tool_use_id,
            context,
            blocked_patterns=blocked_patterns,
            allowed_paths=allowed_paths,
        )

    return hook
