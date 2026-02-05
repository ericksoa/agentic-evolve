"""
Engineering Notebook for RALPH loop iterations on SWE-Refactor.

Records per-iteration observations: agent reasoning, tool calls, file diffs,
costs, and compilation outcomes. Produces both machine-readable JSON and
human-readable Markdown notebooks for study and publication.

Adapted from RefactorBench notebook.py for Java projects.
"""

import difflib
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# File snapshot & diff utilities
# ---------------------------------------------------------------------------

class FileSnapshot:
    """Captures MD5 hashes of source files for change detection."""

    EXCLUDED_FRAGMENTS = {
        "target/", "build/", ".gradle/", "progress.json",
        "__pycache__/", ".class", "notebook/", ".git/",
        ".mvn/", "gradlew", "mvnw",
    }

    def __init__(self, workdir: Path, extensions: tuple[str, ...] = (".java",)):
        self.hashes: dict[str, str] = {}
        self.workdir = workdir
        for ext in extensions:
            for f in workdir.rglob(f"*{ext}"):
                rel = str(f.relative_to(workdir))
                if any(excl in rel for excl in self.EXCLUDED_FRAGMENTS):
                    continue
                try:
                    self.hashes[rel] = hashlib.md5(f.read_bytes()).hexdigest()
                except OSError:
                    pass

    def diff_against(
        self,
        after: "FileSnapshot",
        before_dir: Path,
        after_dir: Path,
    ) -> dict[str, str]:
        """Return {rel_path: unified_diff_text} for files that changed."""
        diffs: dict[str, str] = {}
        all_paths = sorted(set(self.hashes) | set(after.hashes))

        for path in all_paths:
            if self.hashes.get(path) == after.hashes.get(path):
                continue

            old_lines: list[str] = []
            new_lines: list[str] = []

            if path in self.hashes:
                try:
                    old_lines = (before_dir / path).read_text(
                        errors="replace"
                    ).splitlines(keepends=True)
                except OSError:
                    pass

            if path in after.hashes:
                try:
                    new_lines = (after_dir / path).read_text(
                        errors="replace"
                    ).splitlines(keepends=True)
                except OSError:
                    pass

            diff = "".join(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{path}", tofile=f"b/{path}",
            ))
            if diff:
                # Truncate very large diffs per file
                if len(diff) > 8000:
                    diff = diff[:8000] + "\n... (truncated)\n"
                diffs[path] = diff

        return diffs


def compute_solution_diff(original_repo: Path, workdir: Path) -> str:
    """Unified diff from original repo to solved working state."""
    original = FileSnapshot(original_repo)
    current = FileSnapshot(workdir)
    diffs = original.diff_against(current, original_repo, workdir)
    return "\n".join(diffs.values())


def compute_diff_stats(file_diffs: dict[str, str]) -> dict:
    """Count files changed, lines added, lines removed from diff dict."""
    files_changed = len(file_diffs)
    lines_added = 0
    lines_removed = 0
    for diff_text in file_diffs.values():
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                lines_removed += 1
    return {
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
    }


# ---------------------------------------------------------------------------
# Agent output parsing (stream-json format)
# ---------------------------------------------------------------------------

def build_tool_summary(tool_name: str, tool_input: dict) -> str:
    """Human-readable one-line summary of a tool call."""
    if tool_name == "Read":
        fp = tool_input.get("file_path", "")
        return Path(fp).name if fp else "(unknown)"

    if tool_name in ("Write", "Edit"):
        fp = tool_input.get("file_path", "")
        name = Path(fp).name if fp else "(unknown)"
        if tool_name == "Edit":
            old = tool_input.get("old_string", "")[:60]
            new = tool_input.get("new_string", "")[:60]
            return f"{name}: '{old}' -> '{new}'"
        return name

    if tool_name == "Grep":
        pat = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        glob = tool_input.get("glob", "")
        parts = [f"pattern='{pat}'"]
        if glob:
            parts.append(f"glob='{glob}'")
        if path:
            parts.append(f"path='{Path(path).name}'")
        return " ".join(parts)

    if tool_name == "Glob":
        pat = tool_input.get("pattern", "")
        return f"pattern='{pat}'"

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:120]

    # Fallback: show first 120 chars of the input
    return str(tool_input)[:120]


def parse_stream_json(raw_output: str) -> dict:
    """Parse stream-json JSONL from claude -p --output-format stream-json --verbose.

    Returns dict with:
        reasoning: list[str]       - agent text per turn
        tool_calls: list[dict]     - [{tool, input_summary, turn}]
        files_read: list[str]
        files_created: list[str]
        grep_patterns: list[str]
        cost_usd: float
        input_tokens: int
        output_tokens: int
        num_turns: int
    """
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    files_read: set[str] = set()
    files_created: set[str] = set()
    grep_patterns: set[str] = set()
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    num_turns = 0

    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "assistant":
            num_turns += 1
            content = msg.get("message", {}).get("content", [])
            for block in content:
                block_type = block.get("type")

                if block_type == "text":
                    text = block.get("text", "").strip()
                    if text:
                        reasoning.append(text[:2000])

                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    summary = build_tool_summary(tool_name, tool_input)
                    tool_calls.append({
                        "tool": tool_name,
                        "input_summary": summary,
                        "turn": num_turns,
                    })

                    # Track specific tool usage
                    if tool_name == "Read":
                        fp = tool_input.get("file_path", "")
                        if fp:
                            files_read.add(fp)
                    elif tool_name == "Write":
                        fp = tool_input.get("file_path", "")
                        if fp:
                            files_created.add(fp)
                    elif tool_name == "Grep":
                        pat = tool_input.get("pattern", "")
                        if pat:
                            grep_patterns.add(pat)

        elif msg_type == "result":
            cost_usd = msg.get("total_cost_usd", 0) or 0
            usage = msg.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            num_turns = msg.get("num_turns", num_turns)

    return {
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "files_read": sorted(files_read),
        "files_created": sorted(files_created),
        "grep_patterns": sorted(grep_patterns),
        "cost_usd": cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_turns": num_turns,
    }


# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------

class NotebookWriter:
    """Incrementally builds an engineering notebook (JSON + Markdown)."""

    def __init__(
        self,
        notebook_dir: Path,
        task_id: str,
        chain_id: int,
        model: str,
        task_description: str = "",
    ):
        self.notebook_dir = notebook_dir
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = notebook_dir / "notebook.json"
        self.md_path = notebook_dir / "notebook.md"
        self.task_id = task_id
        self.chain_id = chain_id
        self.model = model
        self.task_description = task_description
        self.entries: list[dict] = []
        self.metadata: dict = {
            "task_id": task_id,
            "chain_id": chain_id,
            "model": model,
            "started_at": datetime.now().isoformat(),
            "final_status": "in_progress",
            "total_cost_usd": 0,
            "total_iterations": 0,
        }

        # Resume from existing notebook if present
        if self.json_path.exists():
            try:
                data = json.loads(self.json_path.read_text())
                self.entries = data.get("entries", [])
                self.metadata = data.get("metadata", self.metadata)
            except (json.JSONDecodeError, OSError):
                pass

    def add_entry(self, entry: dict) -> None:
        """Append an iteration entry and rewrite both files."""
        self.entries.append(entry)
        self.metadata["total_iterations"] = len(self.entries)
        self.metadata["total_cost_usd"] = sum(
            e.get("cost_usd", 0) for e in self.entries
        )
        self._write_json()
        self._write_markdown()

    def finalize(
        self,
        status: str,
        solution_diff: str = "",
        results_dir: Path | None = None,
    ) -> None:
        """Mark notebook complete, add solution diff, copy to results."""
        self.metadata["final_status"] = status
        self.metadata["completed_at"] = datetime.now().isoformat()
        self.metadata["solution_diff"] = solution_diff

        self._write_json()
        self._write_markdown(solution_diff=solution_diff)

        # Copy to results directory
        if results_dir:
            results_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"chain_{self.chain_id}"
            shutil.copy2(self.json_path, results_dir / f"{prefix}_notebook.json")
            shutil.copy2(self.md_path, results_dir / f"{prefix}_notebook.md")
            if solution_diff:
                diff_path = results_dir / f"{prefix}_solution.diff"
                diff_path.write_text(solution_diff)

    def _write_json(self) -> None:
        """Write notebook.json with entries + metadata."""
        data = {
            "metadata": self.metadata,
            "entries": self.entries,
        }
        self.json_path.write_text(json.dumps(data, indent=2))

    def _write_markdown(self, solution_diff: str = "") -> None:
        """Regenerate notebook.md from entries."""
        lines: list[str] = []

        # Header
        status = self.metadata.get("final_status", "in_progress").upper()
        lines.append(f"# Engineering Notebook: {self.task_id}\n")
        lines.append(
            f"**Chain:** {self.chain_id} | **Model:** {self.model} | "
            f"**Status:** {status}"
        )
        started = self.metadata.get("started_at", "")
        completed = self.metadata.get("completed_at", "")
        lines.append(
            f"**Started:** {started}"
            + (f" | **Completed:** {completed}" if completed else "")
        )
        lines.append(f"**Total iterations:** {len(self.entries)}")

        # Task description
        if self.task_description:
            lines.append("\n## Task Description\n")
            lines.append(f"> {self.task_description}\n")

        lines.append("\n---\n")

        # Iteration entries
        for entry in self.entries:
            lines.append(self._format_entry(entry))
            lines.append("\n---\n")

        # Solution diff
        if solution_diff:
            lines.append("## Solution Diff (original -> solved)\n")
            lines.append("```diff")
            lines.append(solution_diff)
            lines.append("```\n")

        # Summary table
        lines.append(self._format_summary())

        self.md_path.write_text("\n".join(lines))

    def _format_entry(self, entry: dict) -> str:
        """Format a single iteration entry as Markdown."""
        lines: list[str] = []
        iteration = entry.get("iteration", "?")
        compiled = entry.get("compiled", False)
        wall = entry.get("wall_clock_seconds", 0)

        # Header
        status_marker = " (COMPILED)" if compiled else " (FAILED)"
        lines.append(
            f"## Iteration {iteration}{status_marker} ({wall:.0f}s)"
        )

        # Agent reasoning
        reasoning = entry.get("agent_reasoning", [])
        if reasoning:
            lines.append("\n### Agent Reasoning\n")
            for text in reasoning:
                # Truncate long reasoning blocks for readability
                display = text[:500]
                if len(text) > 500:
                    display += " ..."
                lines.append(f"> {display}\n")

        # Tool calls
        tool_calls = entry.get("tool_calls", [])
        if tool_calls:
            lines.append("\n### Key Actions\n")
            lines.append("| # | Tool | Summary |")
            lines.append("|---|------|---------|")
            for i, tc in enumerate(tool_calls, 1):
                tool = tc.get("tool", "?")
                summary = tc.get("input_summary", "")
                # Escape pipes in summary for table
                summary = summary.replace("|", "\\|")[:100]
                lines.append(f"| {i} | {tool} | {summary} |")
            lines.append("")

        # Files changed
        file_diffs = entry.get("file_diffs", {})
        diff_stats = entry.get("diff_stats", {})
        if file_diffs:
            fc = diff_stats.get("files_changed", 0)
            la = diff_stats.get("lines_added", 0)
            lr = diff_stats.get("lines_removed", 0)
            lines.append(f"\n### Files Changed ({fc} files, +{la} -{lr})\n")
            for path, diff_text in file_diffs.items():
                lines.append(f"**{path}**")
                lines.append("```diff")
                lines.append(diff_text)
                lines.append("```\n")

        # Compile results
        error = entry.get("error_summary", "")
        lines.append(f"\n### Compilation Result\n")
        lines.append(f"- **Status:** {'SUCCESS' if compiled else 'FAILED'}")
        if error:
            lines.append(f"- **Error:** {error[:500]}")

        return "\n".join(lines)

    def _format_summary(self) -> str:
        """Summary statistics table."""
        lines: list[str] = []
        lines.append("## Summary Statistics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        total_wall = sum(e.get("wall_clock_seconds", 0) for e in self.entries)
        total_turns = sum(e.get("num_turns", 0) for e in self.entries)
        total_input = sum(e.get("input_tokens", 0) for e in self.entries)
        total_output = sum(e.get("output_tokens", 0) for e in self.entries)

        lines.append(f"| Iterations | {len(self.entries)} |")
        lines.append(f"| Total wall clock | {total_wall:.0f}s |")
        lines.append(f"| Total turns | {total_turns} |")
        lines.append(f"| Total input tokens | {total_input:,} |")
        lines.append(f"| Total output tokens | {total_output:,} |")

        # Compilation progression
        statuses = [
            "✅" if e.get("compiled") else "❌"
            for e in self.entries
        ]
        lines.append(f"| Compilation progression | {' → '.join(statuses)} |")

        return "\n".join(lines)
