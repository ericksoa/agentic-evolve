#!/usr/bin/env python3
"""
Memory-Test: Persistence Verification

This script verifies that data written by scenario_evolution_run.py
persists after the process exits. This is critical - if memvid isn't
working properly, data would be lost between runs.

Run this AFTER scenario_evolution_run.py to verify persistence.

Usage:
    python verify_persistence.py
"""

import sys
from pathlib import Path

# Add the SDK to path
sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

from evolve_sdk.memory import MemoryManager, MEMVID_AVAILABLE


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_header("PERSISTENCE VERIFICATION")

    print(f"\nMemvid-SDK Available: {MEMVID_AVAILABLE}")

    test_dir = Path(__file__).parent / "test_project"
    if not test_dir.exists():
        print(f"\nERROR: Test directory not found: {test_dir}")
        print("Run scenario_evolution_run.py first!")
        return False

    # Check if store file exists
    store_file = test_dir / ".evolve-sdk" / "evolution_run" / "project_memory.mv2"
    json_fallback = test_dir / ".evolve-sdk" / "evolution_run" / "project_memory.json"

    if store_file.exists():
        print(f"\n[OK] Memvid store file exists: {store_file}")
        print(f"     Size: {store_file.stat().st_size:,} bytes")
    elif json_fallback.exists():
        print(f"\n[WARN] Using JSON fallback: {json_fallback}")
        print(f"       Size: {json_fallback.stat().st_size:,} bytes")
    else:
        print("\nERROR: No store file found!")
        return False

    # Open memory manager (NEW PROCESS - this proves persistence)
    print_header("Loading Data in New Process")

    memory = MemoryManager(test_dir, problem_id="evolution_run")
    stats = memory.stats()

    print(f"Backend: {stats['backend']}")
    print(f"Total records: {stats['total_records']}")
    print(f"Counts: {stats['counts']}")

    # Verify expected data
    expected = {
        "dris": 5,
        "submissions": 5,
        "evaluations": 2,
    }

    all_pass = True
    print_header("Verification Results")

    for type_name, expected_count in expected.items():
        actual_count = stats['counts'].get(type_name, 0)
        if actual_count == expected_count:
            print(f"[PASS] {type_name}: {actual_count} (expected {expected_count})")
        else:
            print(f"[FAIL] {type_name}: {actual_count} (expected {expected_count})")
            all_pass = False

    # Verify we can read specific data
    print_header("Data Integrity Check")

    dris = memory.get_all_dris()
    print(f"\nDRIs loaded: {len(dris)}")
    for dri in dris[:3]:
        print(f"  - {dri.id}: {dri.title[:50]}...")

    submissions = memory.get_all_submissions()
    print(f"\nSubmissions loaded: {len(submissions)}")
    successful = memory.get_successful_submissions()
    failed = memory.get_failed_submissions()
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    evaluations = memory.get_all_evaluations()
    print(f"\nEvaluations loaded: {len(evaluations)}")
    for eval_record in evaluations:
        print(f"  - {eval_record.model_name}: accuracy={eval_record.metrics.get('accuracy')}")

    # Verify semantic search still works
    print_header("Semantic Search Verification")

    print("\nSearching for 'gradient explosion'...")
    results = memory.get_relevant_dris("gradient explosion", limit=3)
    if results:
        print(f"  Found {len(results)} results:")
        for dri in results:
            print(f"    - {dri.id}: {dri.title}")
        if any("divergence" in dri.title.lower() or "learning rate" in dri.title.lower() for dri in results):
            print("  [PASS] Found semantically related DRI")
        else:
            print("  [WARN] Results may not be semantically related")
    else:
        print("  [FAIL] No results found!")
        all_pass = False

    memory.close()

    print_header("PERSISTENCE TEST RESULT")
    if all_pass:
        print("\n[PASS] All persistence checks passed!")
        print("Data successfully survives process restarts.")
    else:
        print("\n[FAIL] Some persistence checks failed!")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
