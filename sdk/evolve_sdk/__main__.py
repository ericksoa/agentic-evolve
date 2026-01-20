"""
CLI entry point for evolve_sdk.

Usage:
    python -m evolve_sdk "shortest Python sort" --mode=size
    python -m evolve_sdk "faster string search" --mode=perf
    python -m evolve_sdk --resume
    python -m evolve_sdk memory stats
    python -m evolve_sdk memory search "fingerprint reduction"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .runner import EvolutionRunner
from .config import EvolutionConfig
from .progress import print_final_results

# Optional memory import
try:
    from .memory import EvolutionMemory
    from .memory.queries import (
        get_mutation_context,
        get_breakthrough_patterns,
        get_trust_calibration,
        get_exploit_patterns,
        get_meta_evolution_insights,
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        prog="evolve_sdk",
        description="Evolve algorithms using Claude Agent SDK with hierarchical agents",
    )

    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem description (what to evolve)",
    )
    parser.add_argument(
        "--mode",
        choices=["size", "perf", "ml"],
        default="size",
        help="Optimization mode (default: size)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume most recent evolution",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=50,
        help="Maximum generations to run (default: 50)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size (default: 10)",
    )
    parser.add_argument(
        "--plateau",
        type=int,
        default=5,
        help="Stop after N generations without improvement (default: 5)",
    )
    parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Directory for evolution state (default: .evolve-sdk)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run mutations sequentially instead of in parallel",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-5-20251101",
        help="Model to use for subagents (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to evolve_config.json file (loads problem, mode, and evaluation settings)",
    )

    args = parser.parse_args()

    # Handle resume
    if args.resume:
        result = resume_evolution(args.evolve_dir)
        if result is None:
            print("No evolution found to resume")
            sys.exit(1)
        problem, mode = result
        print(f"Resuming: {problem} (mode: {mode})")
        # Create runner for resume
        runner = EvolutionRunner(
            problem=problem,
            mode=mode,
            max_generations=args.max_generations,
            population_size=args.population_size,
            plateau_threshold=args.plateau,
            evolve_dir=args.evolve_dir,
            parallel_mutations=not args.no_parallel,
            model=args.model,
        )
    elif args.config:
        # Load from config file
        from .config import EvolutionConfig
        # Build overrides dict, only including non-None values
        overrides = {
            "max_generations": args.max_generations,
            "population_size": args.population_size,
            "plateau_threshold": args.plateau,
            "evolve_dir": args.evolve_dir,
            "model": args.model,
        }
        if args.problem:  # Only override problem if explicitly provided
            overrides["problem"] = args.problem
        config = EvolutionConfig.from_config_file(args.config, **overrides)
        # Override mode if explicitly provided
        if args.mode != "size":  # size is default, so only override if explicitly set
            config.mode = args.mode
        print(f"Loaded config: {config.problem} (mode: {config.mode})")
        if config.test_command:
            print(f"Benchmark: {config.test_command}")
        runner = EvolutionRunner(
            problem=config.problem,
            mode=config.mode,
            max_generations=config.max_generations,
            population_size=config.population_size,
            plateau_threshold=config.plateau_threshold,
            evolve_dir=config.evolve_dir,
            parallel_mutations=not args.no_parallel,
            model=config.model,
            test_command=config.test_command,
            starter_solutions=config.starter_solutions,
            optimization_strategies=config.optimization_strategies,
            cwd=getattr(config, 'cwd', None),
        )
    else:
        if not args.problem:
            parser.error("problem is required unless --resume or --config is specified")
        problem = args.problem
        mode = args.mode

        # Create runner
        runner = EvolutionRunner(
            problem=problem,
            mode=mode,
            max_generations=args.max_generations,
            population_size=args.population_size,
            plateau_threshold=args.plateau,
            evolve_dir=args.evolve_dir,
            parallel_mutations=not args.no_parallel,
            model=args.model,
        )

    # Run evolution
    try:
        result = asyncio.run(runner.run())
        print_final_results(result)
    except KeyboardInterrupt:
        print("\n[!] Evolution interrupted. State saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] Evolution failed: {e}")
        sys.exit(1)


def resume_evolution(evolve_dir: Path) -> tuple[str, str] | None:
    """Find and resume the most recent evolution."""
    from datetime import datetime

    if not evolve_dir.exists():
        return None

    # Find all evolution.json files
    evolutions = []
    for state_file in evolve_dir.glob("*/evolution.json"):
        try:
            state = json.loads(state_file.read_text())
            updated = state.get("updated_at", "")
            evolutions.append((updated, state_file, state))
        except Exception:
            continue

    if not evolutions:
        return None

    # Get most recent
    evolutions.sort(reverse=True)
    _, _, state = evolutions[0]

    return state.get("problem", "unknown"), state.get("mode", "size")


def memory_main():
    """CLI for memory operations."""
    if not MEMORY_AVAILABLE:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="evolve_sdk memory",
        description="Query and manage evolution memory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Memory commands")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )
    stats_parser.add_argument(
        "--problem",
        help="Specific problem to show stats for (default: all)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memory")
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )
    search_parser.add_argument(
        "--problem",
        help="Specific problem to search (default: all)",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum results (default: 10)",
    )

    # Breakthroughs command
    breakthrough_parser = subparsers.add_parser(
        "breakthroughs",
        help="Show patterns that broke plateaus"
    )
    breakthrough_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )
    breakthrough_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results (default: 5)",
    )

    # Trust command
    trust_parser = subparsers.add_parser("trust", help="Show trust calibration data")
    trust_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )

    # Exploits command
    exploits_parser = subparsers.add_parser("exploits", help="Show detected exploit patterns")
    exploits_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )

    # Messages command
    messages_parser = subparsers.add_parser("messages", help="Show inter-agent message feed")
    messages_parser.add_argument(
        "--evolve-dir",
        type=Path,
        default=Path(".evolve-sdk"),
        help="Evolution directory",
    )
    messages_parser.add_argument(
        "--problem",
        help="Specific problem to show messages for (default: all)",
    )
    messages_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum messages to show (default: 20)",
    )
    messages_parser.add_argument(
        "--priority",
        choices=["debug", "info", "important", "urgent", "critical"],
        help="Filter by minimum priority",
    )
    messages_parser.add_argument(
        "--type",
        dest="msg_type",
        choices=["status", "discovery", "warning", "question", "guidance", "strategy", "result", "milestone", "error"],
        help="Filter by message type",
    )
    messages_parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for new messages (poll every 2 seconds)",
    )

    # DRIs command - list and search Don't Repeat Incidents
    dris_parser = subparsers.add_parser("dris", help="List and search Don't Repeat Incidents")
    dris_parser.add_argument(
        "--relevant",
        metavar="TASK",
        help="Find DRIs relevant to a task (semantic search)",
    )
    dris_parser.add_argument(
        "--action",
        metavar="TYPE",
        help="Filter by action type (e.g., submission, huggingface_upload)",
    )
    dris_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum DRIs to show (default: 20)",
    )
    dris_parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project directory (default: current)",
    )

    # Add DRI command
    add_dri_parser = subparsers.add_parser("add-dri", help="Add a new Don't Repeat Incident")
    add_dri_parser.add_argument(
        "--id",
        required=True,
        help="DRI ID (e.g., DRI-015)",
    )
    add_dri_parser.add_argument(
        "--title",
        required=True,
        help="Short title describing the incident",
    )
    add_dri_parser.add_argument(
        "--root-cause",
        required=True,
        help="WHY it failed (most important field)",
    )
    add_dri_parser.add_argument(
        "--submissions-wasted",
        type=int,
        default=0,
        help="Number of submissions wasted",
    )
    add_dri_parser.add_argument(
        "--rules",
        nargs="+",
        help="Rules to follow to avoid this incident",
    )
    add_dri_parser.add_argument(
        "--verify",
        nargs="+",
        help="Commands to verify the incident won't repeat",
    )
    add_dri_parser.add_argument(
        "--tags",
        nargs="+",
        help="Tags for categorization",
    )
    add_dri_parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project directory (default: current)",
    )

    # Submissions command
    submissions_parser = subparsers.add_parser("submissions", help="List competition submissions")
    submissions_parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Number of recent submissions to show (default: 10)",
    )
    submissions_parser.add_argument(
        "--status",
        choices=["succeeded", "failed", "pending", "timeout"],
        help="Filter by status",
    )
    submissions_parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project directory (default: current)",
    )

    # Evaluations command
    evals_parser = subparsers.add_parser("evaluations", help="List model evaluations")
    evals_parser.add_argument(
        "--model",
        metavar="NAME",
        help="Filter by model name",
    )
    evals_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum evaluations to show (default: 10)",
    )
    evals_parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project directory (default: current)",
    )

    # Pre-check command - run before major actions
    precheck_parser = subparsers.add_parser("precheck", help="Run pre-action safety checks")
    precheck_parser.add_argument(
        "action",
        help="Action type (e.g., submission, huggingface_upload, evaluation)",
    )
    precheck_parser.add_argument(
        "--description",
        default="",
        help="Description of the specific action",
    )
    precheck_parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Project directory (default: current)",
    )

    # Migrate command - migrate existing memory.json to new format
    migrate_parser = subparsers.add_parser("migrate", help="Migrate existing memory.json to new format")
    migrate_parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the memory.json file to migrate",
    )
    migrate_parser.add_argument(
        "--project",
        type=Path,
        help="Project directory (default: inferred from json_path)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without making changes",
    )
    migrate_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # New project-based commands (use MemoryManager, not EvolutionMemory)
    if args.command == "dris":
        cmd_dris(
            project=args.project,
            relevant=args.relevant,
            action=args.action,
            limit=args.limit,
        )
        return
    elif args.command == "add-dri":
        cmd_add_dri(
            project=args.project,
            dri_id=args.id,
            title=args.title,
            root_cause=args.root_cause,
            submissions_wasted=args.submissions_wasted,
            rules=args.rules,
            verify=args.verify,
            tags=args.tags,
        )
        return
    elif args.command == "submissions":
        cmd_submissions(
            project=args.project,
            recent=args.recent,
            status=args.status,
        )
        return
    elif args.command == "evaluations":
        cmd_evaluations(
            project=args.project,
            model=args.model,
            limit=args.limit,
        )
        return
    elif args.command == "precheck":
        cmd_precheck(
            project=args.project,
            action=args.action,
            description=args.description,
        )
        return
    elif args.command == "migrate":
        cmd_migrate(
            json_path=args.json_path,
            project=args.project,
            dry_run=args.dry_run,
            yes=args.yes,
        )
        return

    # Legacy evolution-based commands (use EvolutionMemory)
    evolve_dir = args.evolve_dir

    # Find memory stores
    memory_stores = find_memory_stores(evolve_dir, getattr(args, 'problem', None))

    if not memory_stores:
        print(f"No evolution memory found in {evolve_dir}")
        sys.exit(1)

    if args.command == "stats":
        cmd_stats(memory_stores)
    elif args.command == "search":
        cmd_search(memory_stores, args.query, args.limit)
    elif args.command == "breakthroughs":
        cmd_breakthroughs(memory_stores, args.limit)
    elif args.command == "trust":
        cmd_trust(memory_stores)
    elif args.command == "exploits":
        cmd_exploits(memory_stores)
    elif args.command == "messages":
        cmd_messages(
            memory_stores,
            limit=args.limit,
            priority=args.priority,
            msg_type=args.msg_type,
            watch=args.watch,
        )


def find_memory_stores(evolve_dir: Path, problem: str | None = None) -> list[EvolutionMemory]:
    """Find all memory stores in the evolve directory."""
    stores = []

    if not evolve_dir.exists():
        return stores

    for problem_dir in evolve_dir.iterdir():
        if not problem_dir.is_dir():
            continue

        if problem and problem not in problem_dir.name:
            continue

        # Look for memory file (JSON fallback)
        memory_file = problem_dir / "evolution.json"
        if memory_file.exists():
            try:
                store = EvolutionMemory(
                    store_path=memory_file,
                    problem_id=problem_dir.name,
                )
                stores.append(store)
            except Exception:
                continue

    return stores


def cmd_stats(stores: list[EvolutionMemory]):
    """Show memory statistics."""
    print("=" * 60)
    print("EVOLUTION MEMORY STATISTICS")
    print("=" * 60)

    total_frames = 0
    total_by_type = {}

    for store in stores:
        stats = store.stats()
        print(f"\n{store.problem_id}:")
        print(f"  Store: {stats['store_path']}")
        print(f"  Frames: {stats['total_frames']}")
        print(f"  Embeddings: {'enabled' if stats['embeddings_enabled'] else 'disabled'}")

        if stats['frame_types']:
            print("  By type:")
            for ft, count in sorted(stats['frame_types'].items()):
                print(f"    {ft}: {count}")
                total_by_type[ft] = total_by_type.get(ft, 0) + count

        total_frames += stats['total_frames']

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_frames} frames across {len(stores)} problem(s)")
    if total_by_type:
        print("By type:")
        for ft, count in sorted(total_by_type.items()):
            print(f"  {ft}: {count}")


def cmd_search(stores: list[EvolutionMemory], query: str, limit: int):
    """Search memory for matching frames."""
    print(f"Searching for: '{query}'")
    print("=" * 60)

    total_results = 0
    for store in stores:
        results = store.search_text(query, limit=limit)
        if results:
            print(f"\n{store.problem_id} ({len(results)} results):")
            for r in results:
                ft = r.get("frame_type", "unknown")
                gen = r.get("generation", "?")
                print(f"  [{ft}] Gen {gen}")

                # Show relevant info based on type
                if ft == "mutation":
                    delta = r.get("fitness_delta_pct", 0)
                    print(f"    Fitness change: {delta:+.1f}%")
                    tags = r.get("tags", [])
                    if tags:
                        print(f"    Tags: {', '.join(tags)}")
                elif ft == "champion":
                    fitness = r.get("fitness", 0)
                    print(f"    Fitness: {fitness:.4f}")
                elif ft == "exploit":
                    pattern = r.get("pattern_type", "unknown")
                    print(f"    Pattern: {pattern}")

                total_results += 1

    print("\n" + "-" * 60)
    print(f"Total: {total_results} results")


def cmd_breakthroughs(stores: list[EvolutionMemory], limit: int):
    """Show breakthrough patterns."""
    print("BREAKTHROUGH PATTERNS")
    print("=" * 60)

    for store in stores:
        patterns = get_breakthrough_patterns(store, limit=limit)
        if patterns:
            print(f"\n{store.problem_id}:")
            for i, p in enumerate(patterns, 1):
                gen = p.get("generation", "?")
                improvement = p.get("fitness_improvement", 0)
                tags = p.get("tags", [])
                print(f"  {i}. Gen {gen}: +{improvement:.1f}%")
                if tags:
                    print(f"     Tags: {', '.join(tags)}")
                diff_preview = p.get("diff_preview", "")
                if diff_preview:
                    preview = diff_preview[:100].replace("\n", " ")
                    print(f"     Preview: {preview}...")


def cmd_trust(stores: list[EvolutionMemory]):
    """Show trust calibration data."""
    print("TRUST CALIBRATION")
    print("=" * 60)

    for store in stores:
        calibration = get_trust_calibration(store)
        print(f"\n{store.problem_id}:")
        print(f"  Total decisions: {calibration['total_trust_decisions']}")
        print(f"  Human overrides: {calibration['total_human_overrides']}")
        print(f"  Override rate: {calibration['human_override_rate']:.1%}")

        breakdown = calibration.get("recommendation_breakdown", {})
        if breakdown:
            print("  Recommendations:")
            for rec, count in breakdown.items():
                print(f"    {rec}: {count}")

        suggestions = calibration.get("suggestions", [])
        if suggestions:
            print("  Suggestions:")
            for s in suggestions:
                print(f"    - {s}")


def cmd_exploits(stores: list[EvolutionMemory]):
    """Show detected exploit patterns."""
    print("DETECTED EXPLOIT PATTERNS")
    print("=" * 60)

    for store in stores:
        exploits = get_exploit_patterns(store)
        print(f"\n{store.problem_id}:")
        print(f"  Total exploits: {exploits['total_exploits']}")

        patterns = exploits.get("pattern_types", {})
        if patterns:
            print("  By pattern:")
            for pt, count in patterns.items():
                print(f"    {pt}: {count}")

        flags = exploits.get("common_flags", {})
        if flags:
            top_flags = sorted(flags.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Common flags:")
            for flag, count in top_flags:
                print(f"    {flag}: {count}")


def cmd_messages(
    stores: list[EvolutionMemory],
    limit: int = 20,
    priority: str | None = None,
    msg_type: str | None = None,
    watch: bool = False,
):
    """Show inter-agent message feed."""
    import time

    # Priority ordering for filtering
    priority_order = ["debug", "info", "important", "urgent", "critical"]

    def get_messages():
        """Get messages from all stores."""
        all_messages = []
        for store in stores:
            messages = store.get_human_messages(limit=limit * 2)  # Get extra for filtering
            for msg in messages:
                msg["_problem"] = store.problem_id
            all_messages.extend(messages)
        return all_messages

    def filter_messages(messages):
        """Filter messages by priority and type."""
        filtered = []
        for msg in messages:
            # Priority filter
            if priority:
                msg_priority = msg.get("priority", "info")
                if priority_order.index(msg_priority) < priority_order.index(priority):
                    continue

            # Type filter
            if msg_type and msg.get("message_type") != msg_type:
                continue

            filtered.append(msg)
        return filtered[:limit]

    def display_messages(messages, clear=False):
        """Display messages in a formatted feed."""
        if clear:
            print("\033[2J\033[H", end="")  # Clear screen

        print("=" * 70)
        print("EVOLUTION MESSAGE FEED")
        print("=" * 70)

        if not messages:
            print("\nNo messages found.")
            return

        for msg in messages:
            problem = msg.get("_problem", "unknown")
            from_agent = msg.get("from_agent", "unknown")
            emoji = msg.get("agent_emoji", "")
            msg_type = msg.get("message_type", "status")
            priority = msg.get("priority", "info")
            title = msg.get("title", "")
            content = msg.get("content", "")
            gen = msg.get("generation", 0)
            timestamp = msg.get("timestamp", "")[:19]  # Trim to seconds

            # Priority indicator
            priority_indicators = {
                "debug": "  ",
                "info": "  ",
                "important": "* ",
                "urgent": "! ",
                "critical": "!!",
            }
            indicator = priority_indicators.get(priority, "  ")

            # Color based on message type
            type_colors = {
                "milestone": "\033[1;33m",  # Bold yellow
                "discovery": "\033[1;32m",  # Bold green
                "warning": "\033[1;31m",    # Bold red
                "error": "\033[1;31m",      # Bold red
                "result": "\033[1;36m",     # Bold cyan
                "status": "\033[0m",        # Normal
                "strategy": "\033[1;35m",   # Bold magenta
                "question": "\033[1;34m",   # Bold blue
                "guidance": "\033[1;34m",   # Bold blue
            }
            color = type_colors.get(msg_type, "\033[0m")
            reset = "\033[0m"

            # Format: [time] indicator emoji Agent: title
            header = f"\n{indicator}[Gen {gen}] {emoji} {from_agent}: {color}{title}{reset}"
            print(header)

            if content and content != title:
                # Indent content
                for line in content.split("\n")[:3]:  # Max 3 lines
                    print(f"   {line}")

            print(f"   [{problem}] {timestamp}")

        print("\n" + "-" * 70)
        print(f"Showing {len(messages)} message(s)")

    # Initial display
    messages = filter_messages(get_messages())
    display_messages(messages)

    # Watch mode
    if watch:
        print("\nWatching for new messages (Ctrl+C to exit)...")
        seen_count = len(messages)
        try:
            while True:
                time.sleep(2)
                new_messages = filter_messages(get_messages())
                if len(new_messages) != seen_count:
                    display_messages(new_messages, clear=True)
                    seen_count = len(new_messages)
                    print("\nWatching for new messages (Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print("\nStopped watching.")


# ==================== Project Memory Commands (DRIs, Submissions, etc.) ====================

def cmd_dris(
    project: Path,
    relevant: str | None = None,
    action: str | None = None,
    limit: int = 20,
):
    """List and search Don't Repeat Incidents."""
    try:
        from .memory import MemoryManager
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    try:
        memory = MemoryManager(project)
    except Exception as e:
        print(f"Error: Could not load project memory: {e}")
        sys.exit(1)

    print("=" * 70)
    print("DON'T REPEAT INCIDENTS (DRIs)")
    print("=" * 70)

    if relevant:
        # Semantic search
        print(f"\nSearching for DRIs relevant to: '{relevant}'\n")
        dris = memory.get_relevant_dris(relevant, limit=limit)
    elif action:
        # Filter by action type
        print(f"\nDRIs for action type: '{action}'\n")
        dris = memory.get_dris_for_action(action)[:limit]
    else:
        # Show all
        dris = memory.get_all_dris()[:limit]

    if not dris:
        print("\nNo DRIs found.")
        print("\nTo add a DRI:")
        print("  evolve-sdk memory add-dri --id DRI-001 --title '...' --root-cause '...'")
        return

    for dri in dris:
        wasted = f" [WASTED {dri.submissions_wasted} submissions]" if dri.submissions_wasted > 0 else ""
        print(f"\n{dri.id}: {dri.title}{wasted}")
        print(f"  Root cause: {dri.root_cause}")

        if dri.rules_to_follow:
            print("  Rules:")
            for rule in dri.rules_to_follow[:3]:
                print(f"    - {rule}")

        if dri.verification_commands:
            print("  Verify:")
            for cmd in dri.verification_commands[:2]:
                print(f"    $ {cmd}")

        if dri.tags:
            print(f"  Tags: {', '.join(dri.tags)}")

    print("\n" + "-" * 70)
    print(f"Total: {len(dris)} DRI(s)")
    memory.close()


def cmd_add_dri(
    project: Path,
    dri_id: str,
    title: str,
    root_cause: str,
    submissions_wasted: int = 0,
    rules: list[str] | None = None,
    verify: list[str] | None = None,
    tags: list[str] | None = None,
):
    """Add a new Don't Repeat Incident."""
    try:
        from .memory import MemoryManager, DRI
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    try:
        memory = MemoryManager(project)
    except Exception as e:
        print(f"Error: Could not load project memory: {e}")
        sys.exit(1)

    dri = DRI(
        id=dri_id,
        title=title,
        root_cause=root_cause,
        submissions_wasted=submissions_wasted,
        rules_to_follow=rules or [],
        verification_commands=verify or [],
        tags=tags or [],
    )

    try:
        frame_id = memory.add_dri(dri)
        print(f"Added DRI: {dri.id}")
        print(f"  Title: {dri.title}")
        print(f"  Root cause: {dri.root_cause}")
        if submissions_wasted > 0:
            print(f"  Submissions wasted: {submissions_wasted}")
        print(f"  Frame ID: {frame_id}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        memory.close()


def cmd_submissions(
    project: Path,
    recent: int = 10,
    status: str | None = None,
):
    """List competition submissions."""
    try:
        from .memory import MemoryManager
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    try:
        memory = MemoryManager(project)
    except Exception as e:
        print(f"Error: Could not load project memory: {e}")
        sys.exit(1)

    print("=" * 70)
    print("SUBMISSIONS")
    print("=" * 70)

    submissions = memory.get_recent_submissions(limit=recent)

    if status:
        submissions = [s for s in submissions if s.status == status]

    if not submissions:
        print("\nNo submissions found.")
        return

    # Group by status
    succeeded = [s for s in submissions if s.status == "succeeded"]
    failed = [s for s in submissions if s.status == "failed"]
    other = [s for s in submissions if s.status not in ("succeeded", "failed")]

    if succeeded:
        print(f"\n[+] SUCCEEDED ({len(succeeded)}):")
        for s in succeeded[:5]:
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(s.metrics.items())[:3])
            print(f"  {s.id}: {s.model_path}")
            if metrics_str:
                print(f"    Metrics: {metrics_str}")

    if failed:
        print(f"\n[x] FAILED ({len(failed)}):")
        for s in failed[:5]:
            print(f"  {s.id}: {s.model_path}")
            if s.failure_reason:
                print(f"    Reason: {s.failure_reason[:60]}...")

    if other:
        print(f"\n[?] OTHER ({len(other)}):")
        for s in other[:5]:
            print(f"  {s.id}: {s.model_path} ({s.status})")

    print("\n" + "-" * 70)
    print(f"Total: {len(submissions)} submission(s)")
    memory.close()


def cmd_evaluations(
    project: Path,
    model: str | None = None,
    limit: int = 10,
):
    """List model evaluations."""
    try:
        from .memory import MemoryManager
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    try:
        memory = MemoryManager(project)
    except Exception as e:
        print(f"Error: Could not load project memory: {e}")
        sys.exit(1)

    print("=" * 70)
    print("MODEL EVALUATIONS")
    print("=" * 70)

    evals = memory.get_model_evaluations(model_name=model)[:limit]

    if not evals:
        print("\nNo evaluations found.")
        return

    for eval in evals:
        print(f"\n{eval.model_name}:")
        print(f"  Date: {eval.eval_date[:10]}")
        print(f"  Sample size: {eval.sample_size}")

        metrics_str = ", ".join(f"{k}={v}" for k, v in list(eval.metrics.items())[:5])
        print(f"  Metrics: {metrics_str}")

        print(f"  Verdict: {eval.verdict}")

        if eval.sample_size < 100:
            print(f"  WARNING: Sample size {eval.sample_size} < 100 - conclusions may be unreliable")

    print("\n" + "-" * 70)
    print(f"Total: {len(evals)} evaluation(s)")
    memory.close()


def cmd_precheck(
    project: Path,
    action: str,
    description: str = "",
):
    """Run pre-action safety checks."""
    try:
        from .memory import MemoryManager
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    try:
        memory = MemoryManager(project)
    except Exception as e:
        print(f"Error: Could not load project memory: {e}")
        sys.exit(1)

    print("=" * 70)
    print(f"PRE-ACTION CHECK: {action}")
    print("=" * 70)

    check = memory.pre_action_check(action, description)

    if check["warnings"]:
        print("\n!! WARNINGS - Review before proceeding !!")
        print("-" * 40)
        for warning in check["warnings"]:
            print(f"  - {warning}")

    if check["relevant_dris"]:
        print(f"\nRelevant DRIs ({len(check['relevant_dris'])}):")
        print("-" * 40)
        for dri in check["relevant_dris"]:
            print(f"\n  {dri.id}: {dri.title}")
            print(f"    Root cause: {dri.root_cause}")
            if dri.verification_commands:
                print(f"    Verify: {dri.verification_commands[0]}")

    if not check["warnings"] and not check["relevant_dris"]:
        print("\nNo relevant DRIs or warnings found.")
        print("Proceed with caution - this doesn't mean there are no risks.")

    print("\n" + "-" * 70)
    if check["warnings"]:
        print("RECOMMENDATION: Review the warnings above before proceeding.")
    else:
        print("Pre-check complete.")

    memory.close()


def cmd_migrate(
    json_path: Path,
    project: Path | None = None,
    dry_run: bool = False,
    yes: bool = False,
):
    """Migrate existing memory.json to new format."""
    try:
        from .memory.migration import (
            migrate_memory_json,
            analyze_memory_json,
            print_migration_report,
        )
    except ImportError:
        print("Error: Memory module not available. Install with: pip install evolve-sdk[memory]")
        sys.exit(1)

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Analyze first
    print("Analyzing memory file...")
    try:
        analysis = analyze_memory_json(json_path)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)

    print(f"\nSource: {analysis['file_path']}")
    print(f"Size: {analysis['file_size_bytes']:,} bytes (~{analysis['estimated_tokens']:,} tokens)")
    print(f"\nFound:")
    print(f"  - {analysis['dris']['count']} DRIs")
    print(f"  - {analysis['submissions']['count']} submissions ({analysis['submissions']['succeeded']} succeeded, {analysis['submissions']['failed']} failed)")
    print(f"  - {analysis['evaluations']['count']} evaluations")

    if analysis["other_data"]:
        print(f"\nOther data (won't be migrated): {', '.join(analysis['other_data'][:5])}")

    if dry_run:
        print("\n[DRY RUN] No changes will be made.")
        return

    # Confirm
    if not yes:
        response = input("\nProceed with migration? [y/N] ")
        if response.lower() != "y":
            print("Migration cancelled.")
            return

    # Run migration
    print("\nMigrating...")
    try:
        stats = migrate_memory_json(json_path, project, dry_run=False)
    except Exception as e:
        print(f"Error during migration: {e}")
        sys.exit(1)

    # Print results
    print_migration_report(stats, analysis)


if __name__ == "__main__":
    # Check if this is a memory subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "memory":
        # Remove "memory" from argv for subparser
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        memory_main()
    else:
        main()
