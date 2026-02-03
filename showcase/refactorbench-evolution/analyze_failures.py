#!/usr/bin/env python3
"""
Analyze failure patterns from evaluation results.

Usage:
    python3 analyze_failures.py results.json
    python3 evaluate.py strategy.json --json > results.json && python3 analyze_failures.py results.json
"""

import argparse
import json
import sys
from collections import Counter


def categorize_failure(detail: dict) -> str:
    """Categorize a failure into a failure mode."""
    error = detail.get("error", "")
    turns = detail.get("turns_used", 0)

    if "API error" in error:
        return "api_error"
    if "not found" in error.lower():
        return "file_not_found"
    if "timeout" in error.lower():
        return "timeout"
    if turns >= 15:
        return "exhausted_turns"
    if turns == 0:
        return "setup_error"
    if turns <= 2:
        return "early_failure"
    return "test_failure"


def analyze(results: dict):
    """Analyze evaluation results and print failure breakdown."""
    details = results.get("task_details", [])
    if not details:
        print("No task details found in results.")
        return

    total = len(details)
    passed = sum(1 for d in details if d["passed"])
    failed = [d for d in details if not d["passed"]]

    print(f"Overall: {passed}/{total} passed ({passed / total * 100:.1f}%)")
    print(f"SOTA comparison: 35% (Claude 3.5 Sonnet descriptive)")
    print()

    # Per-repo breakdown
    repo_stats = {}
    for d in details:
        repo = d["repo"]
        if repo not in repo_stats:
            repo_stats[repo] = {"passed": 0, "total": 0, "failures": []}
        repo_stats[repo]["total"] += 1
        if d["passed"]:
            repo_stats[repo]["passed"] += 1
        else:
            repo_stats[repo]["failures"].append(d)

    print("Per-repo breakdown:")
    print(f"  {'Repo':<25} {'Passed':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'-' * 51}")
    for repo, stats in sorted(repo_stats.items()):
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {repo:<25} {stats['passed']:>8} {stats['total']:>8} {rate:>7.1f}%")
    print()

    # Failure mode analysis
    if failed:
        print(f"Failure modes ({len(failed)} failures):")
        categories = Counter(categorize_failure(d) for d in failed)
        for cat, count in categories.most_common():
            pct = count / len(failed) * 100
            print(f"  {cat:<25} {count:>4} ({pct:.0f}%)")
        print()

    # Turns distribution
    all_turns = [d["turns_used"] for d in details]
    pass_turns = [d["turns_used"] for d in details if d["passed"]]
    fail_turns = [d["turns_used"] for d in details if not d["passed"]]

    print("Turns distribution:")
    if pass_turns:
        print(f"  Passed: avg={sum(pass_turns)/len(pass_turns):.1f}, "
              f"min={min(pass_turns)}, max={max(pass_turns)}")
    if fail_turns:
        print(f"  Failed: avg={sum(fail_turns)/len(fail_turns):.1f}, "
              f"min={min(fail_turns)}, max={max(fail_turns)}")
    print()

    # Hardest repos
    print("Hardest repos (by failure rate):")
    sorted_repos = sorted(repo_stats.items(),
                         key=lambda x: x[1]["passed"] / x[1]["total"])
    for repo, stats in sorted_repos[:5]:
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {repo}: {rate:.0f}% pass rate")


def main():
    parser = argparse.ArgumentParser(description="Analyze RefactorBench evaluation failures")
    parser.add_argument("results", help="Path to JSON results file")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    analyze(results)


if __name__ == "__main__":
    main()
