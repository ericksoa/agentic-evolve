#!/usr/bin/env python3
"""
Workspace test wrapper for RefactorBench.

Runs AST tests against a workspace directory and outputs JSON fitness.
Used by the evolve-sdk evaluator agent via workspace_test_command.

Usage:
    python3 workspace_test.py <workspace_dir> --repo <repo_name> --task <task_name>

Output (JSON to stdout):
    {"fitness": 0.75, "valid": true, "tests_passed": 6, "tests_total": 8, ...}
"""

import argparse
import json
import re
import sys
from pathlib import Path

from refactor_agent import get_task_info, run_test


def compute_granular_fitness(test_result: dict, num_tests: int = 0) -> dict:
    """Parse pytest -v output to count individual test passes/failures."""
    stdout = test_result.get("stdout", "")

    test_lines = re.findall(
        r'([\w/\.\-]+::[\w\[\]\-]+)\s+(PASSED|FAILED|ERROR)', stdout
    )

    if test_lines:
        tests_passed = sum(1 for _, status in test_lines if status == "PASSED")
        failing_tests = [name for name, status in test_lines if status != "PASSED"]
    else:
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        failing_tests = []

    tests_total = num_tests if num_tests > 0 else max(len(test_lines), 1)

    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "failing_tests": failing_tests,
    }


def main():
    parser = argparse.ArgumentParser(description="Test a workspace and output JSON fitness")
    parser.add_argument("workspace", help="Path to workspace directory")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--task", required=True, help="Task name")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_dir():
        print(json.dumps({"fitness": 0, "valid": False, "error": f"Not a directory: {workspace}"}))
        sys.exit(0)

    try:
        task_info = get_task_info(args.repo, args.task)
        test_result = run_test(task_info["test_file"], workspace, task_info["repo_name"])
        granular = compute_granular_fitness(test_result, num_tests=task_info.get("num_tests", 0))

        tests_total = max(granular["tests_total"], 1)
        fitness = granular["tests_passed"] / tests_total

        output = {
            "fitness": round(fitness, 4),
            "valid": True,
            "tests_passed": granular["tests_passed"],
            "tests_total": tests_total,
            "failing_tests": granular["failing_tests"],
            "test_output": test_result.get("stdout", "")[-2000:],
            "passed": test_result["passed"],
        }
    except Exception as e:
        output = {"fitness": 0, "valid": False, "error": str(e)}

    print(json.dumps(output))


if __name__ == "__main__":
    main()
