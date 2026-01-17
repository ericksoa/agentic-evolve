#!/usr/bin/env python3
"""
Memory-Test: Realistic Evolution Run Scenario

This script simulates a realistic evolution run that uses the evolve-sdk
memory system. It demonstrates:

1. Recording DRIs (Don't Repeat Incidents) as mistakes happen
2. Semantic search to find relevant past incidents
3. Recording submissions and their outcomes
4. Recording model evaluations with proper sample sizes
5. Pre-action safety checks before critical operations
6. Persistence - data survives process restarts

Run this script and observe the outputs to verify memvid integration works.

Usage:
    python scenario_evolution_run.py

The script will create a .evolve-sdk/ directory with project_memory.mv2
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add the SDK to path
sdk_path = Path(__file__).parent.parent / "evolve_sdk"
sys.path.insert(0, str(sdk_path.parent))

from evolve_sdk.memory import (
    MemoryManager,
    DRI,
    Submission,
    ModelEvaluation,
    MEMVID_AVAILABLE,
)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def main():
    print_header("MEMORY-TEST: Realistic Evolution Run Scenario")

    # Check memvid availability
    print(f"\nMemvid-SDK Available: {MEMVID_AVAILABLE}")
    if not MEMVID_AVAILABLE:
        print("WARNING: memvid-sdk not installed, falling back to JSON backend")
        print("Install with: pip install memvid-sdk")

    # Setup: Clean previous test data
    test_dir = Path(__file__).parent / "test_project"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    print(f"\nTest project directory: {test_dir}")

    # Initialize memory manager
    print_subheader("Initializing MemoryManager")
    memory = MemoryManager(test_dir, problem_id="evolution_run")

    stats = memory.stats()
    print(f"Store path: {stats['store_path']}")
    print(f"Backend: {stats['backend']}")
    print(f"Initial record counts: {stats['counts']}")

    # Verify we're using memvid (not JSON fallback)
    if stats['backend'] != 'memvid':
        print("\n!!! CRITICAL: Using JSON fallback, not memvid !!!")
        print("This test requires memvid-sdk to be installed.")
    else:
        print("\n[OK] Using memvid backend")

    # =========================================================================
    # PHASE 1: Simulate an evolution run with mistakes (DRIs)
    # =========================================================================
    print_header("PHASE 1: Recording Incidents (DRIs)")

    # DRI 1: A common ML mistake
    dri1 = DRI(
        id="DRI-001",
        title="Learning rate too high caused divergence",
        root_cause="Used lr=0.1 without warmup, gradients exploded after 100 steps",
        description="Model loss went to NaN during generation 5. Had to restart from checkpoint.",
        submissions_wasted=2,
        time_wasted_hours=3.5,
        rules_to_follow=[
            "Always use learning rate warmup for first 10% of steps",
            "Start with lr=1e-4 and increase gradually",
            "Monitor gradient norms during training",
        ],
        verification_commands=[
            "grep 'grad_norm' training.log | tail -20",
            "python check_gradients.py --model $MODEL",
        ],
        related_actions=["training", "hyperparameter_tuning"],
        tags=["training", "learning_rate", "divergence"],
    )
    memory.add_dri(dri1)
    print(f"Added: {dri1.id} - {dri1.title}")

    # DRI 2: A submission mistake
    dri2 = DRI(
        id="DRI-002",
        title="Submitted wrong model checkpoint",
        root_cause="Script grabbed latest checkpoint instead of best checkpoint by validation score",
        description="Submitted gen15 checkpoint but gen12 had better validation. Wasted a submission.",
        submissions_wasted=1,
        time_wasted_hours=0.5,
        rules_to_follow=[
            "Always verify checkpoint is best by validation metric before submission",
            "Keep a manifest of checkpoint -> validation scores",
            "Double check submission file matches intended model",
        ],
        verification_commands=[
            "python verify_checkpoint.py --expected-score $BEST_SCORE",
        ],
        related_actions=["submission", "model_selection"],
        tags=["submission", "checkpoint", "validation"],
    )
    memory.add_dri(dri2)
    print(f"Added: {dri2.id} - {dri2.title}")

    # DRI 3: Data preprocessing issue
    dri3 = DRI(
        id="DRI-003",
        title="Training data had duplicate entries",
        root_cause="Augmentation script ran twice, creating 2x duplicates in dataset",
        description="Model overfit to duplicated examples. Validation looked good but test was poor.",
        submissions_wasted=1,
        time_wasted_hours=5.0,
        rules_to_follow=[
            "Always check for duplicates after data preprocessing",
            "Hash examples to detect accidental duplication",
            "Run dedup script as part of data pipeline",
        ],
        verification_commands=[
            "python check_duplicates.py --dataset $DATASET_PATH",
        ],
        related_actions=["data_preprocessing", "training"],
        tags=["data", "duplicates", "overfitting"],
    )
    memory.add_dri(dri3)
    print(f"Added: {dri3.id} - {dri3.title}")

    # DRI 4: Evaluation mistake
    dri4 = DRI(
        id="DRI-004",
        title="Evaluation used wrong metric normalization",
        root_cause="Metric was per-batch instead of per-sample, larger batches looked better",
        description="Thought model improved 20% but it was batch size artifact. Real improvement was 3%.",
        submissions_wasted=0,
        time_wasted_hours=2.0,
        rules_to_follow=[
            "Always normalize metrics per-sample, not per-batch",
            "Verify metric calculation with known test cases",
            "Compare multiple batch sizes to detect normalization bugs",
        ],
        verification_commands=[
            "python test_metric.py --batch-sizes 1,8,32,128",
        ],
        related_actions=["evaluation", "metrics"],
        tags=["evaluation", "metrics", "batch_size"],
    )
    memory.add_dri(dri4)
    print(f"Added: {dri4.id} - {dri4.title}")

    # DRI 5: Huggingface upload issue (similar to real chess project issue)
    dri5 = DRI(
        id="DRI-005",
        title="Huggingface upload corrupted tokenizer config",
        root_cause="Used wrong template when uploading, overwrote special tokens",
        description="Model generated garbage because eos_token was wrong after upload.",
        submissions_wasted=3,
        time_wasted_hours=8.0,
        rules_to_follow=[
            "Always verify tokenizer config after upload",
            "Download and test model from HF before submission",
            "Compare local vs uploaded tokenizer.json byte-for-byte",
        ],
        verification_commands=[
            "python verify_hf_upload.py --repo $HF_REPO --local-model $LOCAL_PATH",
            "diff <(cat local/tokenizer.json) <(huggingface-cli download $REPO tokenizer.json)",
        ],
        related_actions=["huggingface_upload", "submission", "tokenizer"],
        tags=["huggingface", "upload", "tokenizer", "corruption"],
    )
    memory.add_dri(dri5)
    print(f"Added: {dri5.id} - {dri5.title}")

    print(f"\nTotal DRIs recorded: {len(memory.get_all_dris())}")

    # =========================================================================
    # PHASE 2: Test Semantic Search
    # =========================================================================
    print_header("PHASE 2: Testing Semantic Search")

    # Test 1: Search for training issues (should find DRI-001, DRI-003)
    print_subheader("Search: 'model training issues gradient'")
    results = memory.get_relevant_dris("model training issues gradient", limit=5)
    for dri in results:
        print(f"  Found: {dri.id} - {dri.title}")

    # Test 2: Search for upload issues (should find DRI-005)
    print_subheader("Search: 'uploading model to cloud'")
    results = memory.get_relevant_dris("uploading model to cloud", limit=5)
    for dri in results:
        print(f"  Found: {dri.id} - {dri.title}")

    # Test 3: Search for submission issues (should find DRI-002, DRI-005)
    print_subheader("Search: 'preparing submission package'")
    results = memory.get_relevant_dris("preparing submission package", limit=5)
    for dri in results:
        print(f"  Found: {dri.id} - {dri.title}")

    # Test 4: Search for data quality issues (should find DRI-003)
    print_subheader("Search: 'dataset quality problems'")
    results = memory.get_relevant_dris("dataset quality problems", limit=5)
    for dri in results:
        print(f"  Found: {dri.id} - {dri.title}")

    # Test 5: Search with different wording (semantic, not keyword)
    print_subheader("Search: 'loss became infinity' (semantic for 'divergence')")
    results = memory.get_relevant_dris("loss became infinity", limit=5)
    for dri in results:
        print(f"  Found: {dri.id} - {dri.title}")

    # =========================================================================
    # PHASE 3: Record Submissions
    # =========================================================================
    print_header("PHASE 3: Recording Submissions")

    base_date = datetime.now() - timedelta(days=10)

    submissions = [
        Submission(
            id="sub-001",
            model_path="models/gen5_v1",
            status="failed",
            date=(base_date + timedelta(days=1)).isoformat(),
            metrics={"accuracy": 0.45, "loss": 2.3},
            failure_reason="Model diverged during evaluation",
            notes="Learning rate was too high",
            tags=["gen5"],
        ),
        Submission(
            id="sub-002",
            model_path="models/gen8_v1",
            status="failed",
            date=(base_date + timedelta(days=3)).isoformat(),
            metrics={"accuracy": 0.62, "loss": 1.1},
            failure_reason="Wrong checkpoint submitted",
            notes="Grabbed latest instead of best",
            tags=["gen8"],
        ),
        Submission(
            id="sub-003",
            model_path="models/gen12_v2",
            status="succeeded",
            date=(base_date + timedelta(days=5)).isoformat(),
            metrics={"accuracy": 0.78, "loss": 0.45},
            notes="Best model so far after fixing checkpoint selection",
            tags=["gen12", "best"],
        ),
        Submission(
            id="sub-004",
            model_path="models/gen15_v1",
            status="failed",
            date=(base_date + timedelta(days=7)).isoformat(),
            metrics={"accuracy": 0.71, "loss": 0.52},
            failure_reason="Tokenizer corrupted during HF upload",
            notes="Wasted submission due to DRI-005 issue",
            tags=["gen15"],
        ),
        Submission(
            id="sub-005",
            model_path="models/gen18_v3",
            status="succeeded",
            date=(base_date + timedelta(days=9)).isoformat(),
            metrics={"accuracy": 0.82, "loss": 0.38},
            notes="New best! Verified upload before submission.",
            tags=["gen18", "best", "verified"],
        ),
    ]

    for sub in submissions:
        memory.add_submission(sub)
        status_icon = "[OK]" if sub.status == "succeeded" else "[FAIL]"
        print(f"{status_icon} {sub.id}: {sub.model_path} - acc={sub.metrics.get('accuracy', 'N/A')}")

    print(f"\nTotal submissions: {len(memory.get_all_submissions())}")
    print(f"Successful: {len(memory.get_successful_submissions())}")
    print(f"Failed: {len(memory.get_failed_submissions())}")

    # =========================================================================
    # PHASE 4: Record Model Evaluations
    # =========================================================================
    print_header("PHASE 4: Recording Model Evaluations")

    evaluations = [
        ModelEvaluation(
            model_name="gen12_v2",
            metrics={"accuracy": 0.78, "f1": 0.75, "precision": 0.80, "recall": 0.71},
            sample_size=500,
            methodology="5-fold cross-validation on held-out test set",
            verdict="Significant improvement over baseline (0.65 -> 0.78)",
            eval_date=(base_date + timedelta(days=5)).isoformat(),
            config={"batch_size": 32, "eval_steps": 100},
            notes="First model to exceed 0.75 accuracy threshold",
            tags=["milestone", "gen12"],
        ),
        ModelEvaluation(
            model_name="gen18_v3",
            metrics={"accuracy": 0.82, "f1": 0.80, "precision": 0.83, "recall": 0.77},
            sample_size=500,
            methodology="5-fold cross-validation on held-out test set",
            verdict="New best model. Ready for final submission.",
            eval_date=(base_date + timedelta(days=9)).isoformat(),
            config={"batch_size": 32, "eval_steps": 100},
            notes="Achieved target accuracy. Verified on multiple runs.",
            tags=["best", "final", "gen18"],
        ),
    ]

    for eval_record in evaluations:
        memory.add_evaluation(eval_record)
        print(f"Recorded: {eval_record.model_name}")
        print(f"  Metrics: {eval_record.metrics}")
        print(f"  Sample size: {eval_record.sample_size}")
        print(f"  Verdict: {eval_record.verdict}")

    # Try to add an evaluation with insufficient sample size (should fail)
    print_subheader("Testing sample size validation (should fail)")
    try:
        bad_eval = ModelEvaluation(
            model_name="gen20_untested",
            metrics={"accuracy": 0.90},
            sample_size=10,  # Too small!
            methodology="Quick spot check",
            verdict="Looks great!",
        )
        memory.add_evaluation(bad_eval)
        print("ERROR: Should have rejected insufficient sample size!")
    except ValueError as e:
        print(f"[OK] Correctly rejected: {e}")

    # =========================================================================
    # PHASE 5: Pre-Action Safety Checks
    # =========================================================================
    print_header("PHASE 5: Pre-Action Safety Checks")

    print_subheader("Check before 'submission' action")
    check = memory.pre_action_check("submission", "uploading final model to competition")
    print(f"Relevant DRIs: {len(check['relevant_dris'])}")
    print(f"Warnings: {len(check['warnings'])}")
    for warning in check['warnings']:
        print(f"  ! {warning}")
    print(f"Safe to proceed: {check['proceed']}")

    print_subheader("Check before 'huggingface_upload' action")
    check = memory.pre_action_check("huggingface_upload", "pushing model to hub")
    print(f"Relevant DRIs: {len(check['relevant_dris'])}")
    for warning in check['warnings']:
        print(f"  ! {warning}")

    print_subheader("Check before 'training' action")
    check = memory.pre_action_check("training", "starting new training run")
    print(f"Relevant DRIs: {len(check['relevant_dris'])}")
    for warning in check['warnings']:
        print(f"  ! {warning}")

    # =========================================================================
    # PHASE 6: Final Statistics
    # =========================================================================
    print_header("PHASE 6: Final Statistics")

    stats = memory.stats()
    print(f"Store path: {stats['store_path']}")
    print(f"Backend: {stats['backend']}")
    print(f"Total records: {stats['total_records']}")
    print(f"Counts by type:")
    for type_name, count in stats['counts'].items():
        print(f"  {type_name}: {count}")

    # Close memory manager
    memory.close()

    print_header("TEST COMPLETE")
    print(f"\nMemvid backend: {stats['backend']}")
    print(f"Store file: {stats['store_path']}")
    print("\nTo verify persistence, run: python verify_persistence.py")

    return stats['backend'] == 'memvid'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
