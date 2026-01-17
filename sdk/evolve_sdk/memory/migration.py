"""
Migration tools for evolve-sdk memory.

Provides utilities to migrate existing memory.json files to the new
MemoryManager format, extracting DRIs, submissions, and evaluations.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .project_schemas import DRI, Submission, ModelEvaluation, ConfigChange
from .memory_manager import MemoryManager


def migrate_memory_json(
    json_path: Path,
    project_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Migrate an existing memory.json file to the new MemoryManager format.

    This function extracts:
    - dont_repeat_incidents -> DRI objects
    - submissions -> Submission objects
    - model_evaluations -> ModelEvaluation objects

    Args:
        json_path: Path to the existing memory.json file
        project_path: Path to the project directory (default: parent of json_path)
        dry_run: If True, just analyze without writing

    Returns:
        Dictionary with migration statistics:
        {
            "dris_found": int,
            "dris_migrated": int,
            "submissions_found": int,
            "submissions_migrated": int,
            "evaluations_found": int,
            "evaluations_migrated": int,
            "errors": list[str],
        }
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Memory file not found: {json_path}")

    if project_path is None:
        # Assume structure: project/.evolve-sdk/problem/memory.json
        project_path = json_path.parent.parent.parent

    # Load existing JSON
    with open(json_path) as f:
        data = json.load(f)

    stats = {
        "dris_found": 0,
        "dris_migrated": 0,
        "submissions_found": 0,
        "submissions_migrated": 0,
        "evaluations_found": 0,
        "evaluations_migrated": 0,
        "errors": [],
    }

    # Initialize memory manager (if not dry run)
    memory = None
    if not dry_run:
        memory = MemoryManager(project_path)

    # Migrate DRIs
    dris_data = data.get("dont_repeat_incidents", [])
    stats["dris_found"] = len(dris_data)

    for i, dri_data in enumerate(dris_data):
        try:
            dri = _convert_dri(dri_data, i + 1)
            if memory:
                memory.add_dri(dri)
            stats["dris_migrated"] += 1
        except Exception as e:
            stats["errors"].append(f"DRI {i}: {e}")

    # Migrate submissions
    submissions_data = data.get("submissions", [])
    stats["submissions_found"] = len(submissions_data)

    for i, sub_data in enumerate(submissions_data):
        try:
            submission = _convert_submission(sub_data)
            if memory:
                memory.add_submission(submission)
            stats["submissions_migrated"] += 1
        except Exception as e:
            stats["errors"].append(f"Submission {i}: {e}")

    # Migrate model evaluations
    evals_data = data.get("model_evaluations", {})
    if isinstance(evals_data, dict):
        # Old format: {"model_name": {...}}
        stats["evaluations_found"] = len(evals_data)
        for model_name, eval_data in evals_data.items():
            try:
                evaluation = _convert_evaluation(model_name, eval_data)
                if memory:
                    memory.add_evaluation(evaluation)
                stats["evaluations_migrated"] += 1
            except Exception as e:
                stats["errors"].append(f"Evaluation {model_name}: {e}")
    elif isinstance(evals_data, list):
        # New format: [{...}, {...}]
        stats["evaluations_found"] = len(evals_data)
        for i, eval_data in enumerate(evals_data):
            try:
                model_name = eval_data.get("model_name", f"model_{i}")
                evaluation = _convert_evaluation(model_name, eval_data)
                if memory:
                    memory.add_evaluation(evaluation)
                stats["evaluations_migrated"] += 1
            except Exception as e:
                stats["errors"].append(f"Evaluation {i}: {e}")

    if memory:
        memory.close()

    return stats


def _convert_dri(data: dict, index: int) -> DRI:
    """Convert a DRI dictionary to a DRI object."""
    # Handle various ID formats
    dri_id = data.get("id") or data.get("dri_id") or f"DRI-{index:03d}"

    # Ensure proper format
    if not dri_id.startswith("DRI-"):
        dri_id = f"DRI-{dri_id}"

    # Handle root_cause - can be string or list
    root_cause = data.get("root_cause", data.get("cause", data.get("why", "Unknown")))
    if isinstance(root_cause, list):
        root_cause = "; ".join(root_cause)

    # Handle rules_to_follow - can be list or dict
    rules = data.get("rules_to_follow", data.get("rules", []))
    if isinstance(rules, dict):
        # Flatten dict values into a list of rules
        flat_rules = []
        for key, value in rules.items():
            if isinstance(value, list):
                flat_rules.extend(value)
            else:
                flat_rules.append(f"{key}: {value}")
        rules = flat_rules
    elif not isinstance(rules, list):
        rules = [str(rules)] if rules else []

    # Handle verification_commands - can be string or list
    verify = data.get("verification_commands", data.get("verify", data.get("verification_code", [])))
    if isinstance(verify, str):
        verify = [verify] if verify else []
    elif not isinstance(verify, list):
        verify = []

    return DRI(
        id=dri_id,
        title=data.get("title", data.get("incident", "Unknown incident")),
        root_cause=root_cause,
        date=data.get("date", datetime.now().isoformat()),
        description=data.get("description", ""),
        submissions_wasted=data.get("submissions_wasted", data.get("wasted", 0)),
        time_wasted_hours=data.get("time_wasted_hours", 0.0),
        rules_to_follow=rules,
        verification_commands=verify,
        related_actions=data.get("related_actions", []),
        tags=data.get("tags", []),
    )


def _convert_submission(data: dict) -> Submission:
    """Convert a submission dictionary to a Submission object."""
    # Determine status
    status = data.get("status", "unknown").lower()
    if "success" in status or "completed" in status:
        status = "succeeded"
    elif "fail" in status or "error" in status:
        status = "failed"
    elif "timeout" in status:
        status = "timeout"
    elif "pending" in status or "running" in status:
        status = "pending"

    # Extract metrics
    metrics = data.get("metrics", {})
    if not metrics:
        # Try to extract common metrics
        for key in ["acpl", "score", "accuracy", "loss"]:
            if key in data:
                metrics[key] = data[key]

    return Submission(
        id=str(data.get("submission_id", data.get("id", "unknown"))),
        model_path=data.get("model", data.get("model_path", "")),
        status=status,
        date=data.get("submitted_at", data.get("date", datetime.now().isoformat())),
        metrics=metrics,
        submission_url=data.get("submission_url", ""),
        failure_reason=data.get("error", data.get("failure_reason")),
        fixes_applied=data.get("fixes_applied", []),
        verification_done=data.get("verification", {}).get("done", False),
        notes=data.get("notes", ""),
        tags=data.get("tags", []),
    )


def _convert_evaluation(model_name: str, data: dict) -> ModelEvaluation:
    """Convert an evaluation dictionary to a ModelEvaluation object."""
    # Extract metrics
    metrics = data.get("metrics", {})
    if not metrics:
        # Try to extract common metrics from flat structure
        for key in ["acpl", "score", "accuracy", "loss", "win_rate"]:
            if key in data:
                metrics[key] = data[key]

    # Determine sample size
    sample_size = data.get("sample_size", data.get("n_samples", data.get("samples", 0)))
    if sample_size == 0:
        # Try to infer from data
        sample_size = data.get("positions_evaluated", data.get("games", 100))

    return ModelEvaluation(
        model_name=model_name,
        metrics=metrics,
        sample_size=sample_size,
        methodology=data.get("methodology", data.get("method", "Unknown")),
        verdict=data.get("verdict", data.get("conclusion", "No verdict")),
        eval_date=data.get("eval_date", data.get("date", datetime.now().isoformat())),
        config=data.get("config", {}),
        notes=data.get("notes", ""),
        tags=data.get("tags", []),
    )


def analyze_memory_json(json_path: Path) -> dict[str, Any]:
    """
    Analyze an existing memory.json file without migrating.

    Useful for understanding what will be migrated.

    Args:
        json_path: Path to the memory.json file

    Returns:
        Analysis dictionary with counts and sample data
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Memory file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    # Calculate file size
    file_size = json_path.stat().st_size
    token_estimate = file_size // 4  # Rough estimate

    analysis = {
        "file_path": str(json_path),
        "file_size_bytes": file_size,
        "estimated_tokens": token_estimate,
        "top_level_keys": list(data.keys()),
        "dris": {
            "count": 0,
            "sample_ids": [],
        },
        "submissions": {
            "count": 0,
            "succeeded": 0,
            "failed": 0,
        },
        "evaluations": {
            "count": 0,
            "model_names": [],
        },
        "other_data": [],
    }

    # Analyze DRIs
    dris = data.get("dont_repeat_incidents", [])
    analysis["dris"]["count"] = len(dris)
    analysis["dris"]["sample_ids"] = [
        d.get("id", d.get("dri_id", f"DRI-{i}"))
        for i, d in enumerate(dris[:5])
    ]

    # Analyze submissions
    submissions = data.get("submissions", [])
    analysis["submissions"]["count"] = len(submissions)
    for sub in submissions:
        status = sub.get("status", "").lower()
        if "success" in status or "completed" in status:
            analysis["submissions"]["succeeded"] += 1
        elif "fail" in status or "error" in status:
            analysis["submissions"]["failed"] += 1

    # Analyze evaluations
    evals = data.get("model_evaluations", {})
    if isinstance(evals, dict):
        analysis["evaluations"]["count"] = len(evals)
        analysis["evaluations"]["model_names"] = list(evals.keys())[:5]
    elif isinstance(evals, list):
        analysis["evaluations"]["count"] = len(evals)
        analysis["evaluations"]["model_names"] = [
            e.get("model_name", f"model_{i}")
            for i, e in enumerate(evals[:5])
        ]

    # Note other data that won't be migrated
    migrated_keys = {"dont_repeat_incidents", "submissions", "model_evaluations"}
    analysis["other_data"] = [k for k in data.keys() if k not in migrated_keys]

    return analysis


def print_migration_report(stats: dict[str, Any], analysis: dict[str, Any] | None = None):
    """Print a formatted migration report."""
    print("=" * 60)
    print("MIGRATION REPORT")
    print("=" * 60)

    if analysis:
        print(f"\nSource: {analysis['file_path']}")
        print(f"Size: {analysis['file_size_bytes']:,} bytes (~{analysis['estimated_tokens']:,} tokens)")
        print(f"Top-level keys: {', '.join(analysis['top_level_keys'])}")

    print("\nMigration Results:")
    print("-" * 40)
    print(f"  DRIs:        {stats['dris_migrated']}/{stats['dris_found']} migrated")
    print(f"  Submissions: {stats['submissions_migrated']}/{stats['submissions_found']} migrated")
    print(f"  Evaluations: {stats['evaluations_migrated']}/{stats['evaluations_found']} migrated")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    if analysis and analysis["other_data"]:
        print(f"\nNot migrated (manual review needed): {', '.join(analysis['other_data'])}")

    print("\n" + "=" * 60)


# CLI integration
def migrate_cli(args):
    """CLI handler for migration command."""
    json_path = Path(args.json_path)
    project_path = Path(args.project) if args.project else None

    # Analyze first
    print("Analyzing memory file...")
    analysis = analyze_memory_json(json_path)

    print(f"\nFound:")
    print(f"  - {analysis['dris']['count']} DRIs")
    print(f"  - {analysis['submissions']['count']} submissions")
    print(f"  - {analysis['evaluations']['count']} evaluations")
    print(f"  - Estimated tokens: {analysis['estimated_tokens']:,}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")

    # Confirm if not dry run
    if not args.dry_run and not args.yes:
        response = input("\nProceed with migration? [y/N] ")
        if response.lower() != "y":
            print("Migration cancelled.")
            return

    # Run migration
    print("\nMigrating...")
    stats = migrate_memory_json(json_path, project_path, dry_run=args.dry_run)

    # Print report
    print_migration_report(stats, analysis)
