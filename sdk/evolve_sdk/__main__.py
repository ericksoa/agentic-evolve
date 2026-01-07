"""
CLI entry point for evolve_sdk.

Usage:
    python -m evolve_sdk "shortest Python sort" --mode=size
    python -m evolve_sdk "faster string search" --mode=perf
    python -m evolve_sdk --resume
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .runner import EvolutionRunner
from .config import EvolutionConfig
from .progress import print_final_results


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
        default="claude-sonnet-4-20250514",
        help="Model to use for subagents (default: claude-sonnet-4-20250514)",
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
    import json
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


if __name__ == "__main__":
    main()
