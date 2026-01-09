#!/usr/bin/env python3
"""
Fetch published AutoML baseline scores for comparison.

This script queries OpenML for existing runs on our target datasets
and/or runs the AMLB framework to get comparison scores.

Strategy:
1. Query OpenML API for existing runs on target datasets
2. If needed, run AMLB locally for specific frameworks
3. Save comparison baselines for our benchmark
"""

import json
import openml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Our pilot dataset IDs
PILOT_DATASETS = {
    31: "credit-g",
    37: "diabetes",
    54: "vehicle",
    36: "segment",
    1067: "kc1"
}

# Known baseline scores from literature/testing
# Format: {dataset_id: {model: {metric: score}}}
KNOWN_BASELINES = {
    37: {  # diabetes
        "LogisticRegression_default": {"accuracy": 0.77, "source": "sklearn docs"},
        "RandomForest_default": {"accuracy": 0.786, "source": "sklearn docs"},
        "DecisionTree_default": {"accuracy": 0.714, "source": "sklearn docs"},
    }
}

# AMLB reported relative performance (from 2023 benchmark)
# These are win rates - percentage of datasets where framework beats the baseline
AMLB_RELATIVE_PERFORMANCE = {
    "autogluon": {"avg_rank": 1.95, "top1_rate": 0.63},
    "lightautoml": {"avg_rank": 4.78, "top1_rate": 0.12},
    "h2o": {"avg_rank": 4.98, "top1_rate": 0.01},
    "flaml": {"avg_rank": 5.33, "top1_rate": 0.02},
    "autosklearn2": {"avg_rank": 5.58, "top1_rate": 0.05},
    "xgboost_default": {"avg_rank": 8.86, "top1_rate": 0.02},
    "randomforest_default": {"avg_rank": 9.78, "top1_rate": 0.01},
}


def fetch_openml_runs(dataset_id: int, limit: int = 100) -> List[Dict]:
    """
    Fetch existing runs from OpenML for a dataset.

    OpenML stores all experimental runs, which can include
    Auto-sklearn, scikit-learn, etc.
    """
    try:
        # Get tasks for this dataset
        tasks = openml.tasks.list_tasks(
            data_id=dataset_id,
            task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            output_format='dataframe'
        )

        if tasks.empty:
            return []

        # Get runs for the first task
        task_id = tasks.index[0]
        runs = openml.runs.list_runs(
            task=[task_id],
            size=limit,
            output_format='dataframe'
        )

        if runs.empty:
            return []

        # Extract relevant info
        results = []
        for _, run in runs.iterrows():
            results.append({
                'run_id': run.name if hasattr(run, 'name') else run.get('run_id'),
                'flow_name': run.get('flow_name', 'unknown'),
                'accuracy': run.get('predictive_accuracy'),
                'auc': run.get('area_under_roc_curve'),
            })

        return results

    except Exception as e:
        print(f"Error fetching runs for dataset {dataset_id}: {e}")
        return []


def get_comparison_targets() -> Dict[str, Dict]:
    """
    Get target scores we need to beat for each tier.

    Based on AMLB rankings:
    - Tier 1 (World-class): Beat AutoGluon (avg rank ~2)
    - Tier 2 (Excellent): Beat FLAML/Auto-sklearn (avg rank ~5)
    - Tier 3 (Good): Beat XGBoost/RF defaults (avg rank ~9)
    """
    return {
        "tier1_worldclass": {
            "description": "Beat AutoGluon - top AutoML performance",
            "target_rank": 2.0,
            "comparison": "autogluon",
        },
        "tier2_excellent": {
            "description": "Beat FLAML/Auto-sklearn - production AutoML",
            "target_rank": 5.0,
            "comparison": "flaml",
        },
        "tier3_good": {
            "description": "Beat tuned XGBoost - strong baseline",
            "target_rank": 8.0,
            "comparison": "xgboost_default",
        },
        "tier4_baseline": {
            "description": "Beat RandomForest default - minimum bar",
            "target_rank": 9.0,
            "comparison": "randomforest_default",
        }
    }


def estimate_target_score(our_baseline_f1: float, target_tier: str = "tier2_excellent") -> float:
    """
    Estimate what F1 score we need to beat a given tier.

    Based on AMLB data:
    - AutoGluon improves over XGBoost by ~23%
    - FLAML improves over XGBoost by ~17%
    - Auto-sklearn improves over XGBoost by ~15%

    So if our XGBoost baseline is X, we need:
    - To beat AutoGluon: X * 1.23
    - To beat FLAML: X * 1.17
    - To beat Auto-sklearn: X * 1.15
    """
    improvements = {
        "tier1_worldclass": 1.23,  # AutoGluon
        "tier2_excellent": 1.17,   # FLAML
        "tier3_good": 1.10,        # Tuned XGBoost
        "tier4_baseline": 1.0,     # RF default
    }

    multiplier = improvements.get(target_tier, 1.0)
    return our_baseline_f1 * multiplier


def create_comparison_report(our_results: Dict[int, Dict]) -> pd.DataFrame:
    """
    Create a comparison report showing where we stand vs AutoML tools.
    """
    rows = []

    for dataset_id, result in our_results.items():
        our_f1 = result.get('best_f1', 0)
        our_model = result.get('best_baseline', 'unknown')

        # Estimate targets
        target_flaml = estimate_target_score(our_f1, "tier2_excellent")
        target_autogluon = estimate_target_score(our_f1, "tier1_worldclass")

        rows.append({
            'dataset': result.get('dataset_name', str(dataset_id)),
            'our_best_f1': our_f1,
            'our_model': our_model,
            'est_flaml_target': target_flaml,
            'est_autogluon_target': target_autogluon,
            'need_to_beat_flaml': f"+{(target_flaml - our_f1):.3f}",
            'need_to_beat_autogluon': f"+{(target_autogluon - our_f1):.3f}",
        })

    return pd.DataFrame(rows)


def main():
    """Print comparison targets and fetch available baselines."""
    print("=" * 60)
    print("AutoML Comparison Targets")
    print("=" * 60)

    print("\nTier Definitions (from AMLB 2023):")
    for tier, info in get_comparison_targets().items():
        print(f"  {tier}: {info['description']}")

    print("\n" + "=" * 60)
    print("Estimated Improvement Needed")
    print("=" * 60)
    print("""
Based on AMLB data, typical improvements over RF/XGBoost defaults:
  - AutoGluon:    +23% (avg rank 1.95)
  - LightAutoML:  +17% (avg rank 4.78)
  - FLAML:        +17% (avg rank 5.33)
  - Auto-sklearn: +15% (avg rank 5.58)

So if our XGBoost baseline gets F1=0.60:
  - To claim "beats FLAML": need F1 >= 0.70
  - To claim "beats AutoGluon": need F1 >= 0.74

Note: These are estimates. For rigorous comparison, run AMLB directly.
""")

    print("\n" + "=" * 60)
    print("Fetching OpenML runs for pilot datasets...")
    print("=" * 60)

    for dataset_id, name in PILOT_DATASETS.items():
        print(f"\n{name} (ID={dataset_id}):")
        runs = fetch_openml_runs(dataset_id, limit=10)
        if runs:
            for run in runs[:5]:
                print(f"  {run['flow_name'][:40]}: acc={run.get('accuracy', 'N/A')}")
        else:
            print("  No runs found")


if __name__ == "__main__":
    main()
