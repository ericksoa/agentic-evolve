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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

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


if __name__ == "__main__":
    # Check if this is a memory subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "memory":
        # Remove "memory" from argv for subparser
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        memory_main()
    else:
        main()
