#!/usr/bin/env python3
"""
Benchmark Discovery for /evolve

Searches the benchmark registry to find known benchmarks matching a problem description.
"""

import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Try tomllib (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Need tomllib (Python 3.11+) or tomli package", file=sys.stderr)
        print("Install with: pip install tomli", file=sys.stderr)
        sys.exit(1)


@dataclass
class BenchmarkSource:
    name: str
    repo: str
    bench_dir: Optional[str] = None
    language: str = "rust"
    notes: Optional[str] = None
    ffi_required: bool = False


@dataclass
class BenchmarkMatch:
    problem_id: str
    description: str
    score: float
    sources: list[BenchmarkSource]
    trait_signature: Optional[str] = None


def load_registry(registry_path: Path) -> dict:
    """Load the benchmark registry from TOML file."""
    with open(registry_path, "rb") as f:
        return tomllib.load(f)


def extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from text."""
    # Simple tokenization: split on non-alphanumeric, lowercase
    import re
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return set(words)


def score_match(problem_keywords: set[str], benchmark_keywords: list[str]) -> float:
    """
    Score how well problem keywords match benchmark keywords.
    Returns 0.0 to 1.0.
    """
    if not benchmark_keywords:
        return 0.0

    benchmark_set = set(k.lower() for k in benchmark_keywords)
    matches = problem_keywords & benchmark_set

    if not matches:
        return 0.0

    # Score based on proportion of benchmark keywords matched
    # Weight by number of matches to prefer more specific matches
    score = len(matches) / len(benchmark_set)

    # Bonus for matching multiple keywords
    if len(matches) >= 2:
        score = min(1.0, score + 0.2)

    return score


def find_matching_benchmarks(
    problem_description: str,
    registry_path: Path,
    threshold: float = 0.3
) -> list[BenchmarkMatch]:
    """
    Find benchmarks matching the problem description.

    Returns list of matches sorted by score (highest first).
    """
    registry = load_registry(registry_path)
    problem_keywords = extract_keywords(problem_description)

    matches = []

    for problem_id, config in registry.items():
        # Skip meta sections
        if problem_id.startswith("_"):
            continue

        if not isinstance(config, dict):
            continue

        keywords = config.get("keywords", [])
        score = score_match(problem_keywords, keywords)

        if score >= threshold:
            sources = []
            for src in config.get("sources", []):
                sources.append(BenchmarkSource(
                    name=src.get("name", "unknown"),
                    repo=src.get("repo", ""),
                    bench_dir=src.get("bench_dir"),
                    language=src.get("language", "rust"),
                    notes=src.get("notes"),
                    ffi_required=src.get("ffi_required", False),
                ))

            matches.append(BenchmarkMatch(
                problem_id=problem_id,
                description=config.get("description", ""),
                score=score,
                sources=sources,
                trait_signature=config.get("trait_signature"),
            ))

    # Sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def clone_benchmark(source: BenchmarkSource, target_dir: Path) -> bool:
    """Clone a benchmark repository to target directory."""
    if target_dir.exists():
        print(f"  Directory exists, skipping clone: {target_dir}")
        return True

    print(f"  Cloning {source.repo}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", source.repo, str(target_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Clone failed: {result.stderr}", file=sys.stderr)
        return False

    print(f"  Cloned to {target_dir}")
    return True


def format_output(matches: list[BenchmarkMatch]) -> dict:
    """Format matches as JSON for Claude to consume."""
    return {
        "found": len(matches) > 0,
        "matches": [
            {
                "problem_id": m.problem_id,
                "description": m.description,
                "score": round(m.score, 2),
                "trait_signature": m.trait_signature,
                "sources": [
                    {
                        "name": s.name,
                        "repo": s.repo,
                        "bench_dir": s.bench_dir,
                        "language": s.language,
                        "notes": s.notes,
                        "ffi_required": s.ffi_required,
                    }
                    for s in m.sources
                ]
            }
            for m in matches
        ]
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Find benchmarks for a problem")
    parser.add_argument("problem", help="Problem description")
    parser.add_argument(
        "--registry",
        default=Path(__file__).parent / "benchmarks.toml",
        type=Path,
        help="Path to benchmark registry",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum score threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--clone",
        type=Path,
        help="Clone best match to this directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    if not args.registry.exists():
        print(f"Registry not found: {args.registry}", file=sys.stderr)
        sys.exit(1)

    matches = find_matching_benchmarks(
        args.problem,
        args.registry,
        args.threshold,
    )

    if args.json:
        print(json.dumps(format_output(matches), indent=2))
        return

    if not matches:
        print("No matching benchmarks found.")
        print("\nConsider contributing to the registry if you know of one!")
        sys.exit(0)

    print(f"Found {len(matches)} matching benchmark(s):\n")

    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.problem_id} (score: {match.score:.0%})")
        print(f"   {match.description}")
        if match.trait_signature:
            print(f"   Signature: {match.trait_signature}")
        print(f"   Sources:")
        for src in match.sources:
            lang_note = f" [{src.language}]" if src.language != "rust" else ""
            ffi_note = " (FFI required)" if src.ffi_required else ""
            print(f"     - {src.name}{lang_note}{ffi_note}: {src.repo}")
            if src.notes:
                print(f"       {src.notes}")
        print()

    if args.clone and matches:
        best = matches[0]
        if best.sources:
            source = best.sources[0]
            clone_benchmark(source, args.clone)


if __name__ == "__main__":
    main()
