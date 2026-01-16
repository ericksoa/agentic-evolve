"""
gen0_e.py - Size-aware batching strategy

Approach: Use size-homogeneous batching to group similar-sized LPs
for better GPU utilization. Moderate transforms.

Strategy: Size-based batching, geometric scaling, moderate tolerance
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_e_size_aware",
    "description": "Size-aware: batching by problem size, moderate transforms",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-6,
        "max_iterations": 15000,
        "presolve": True,
        "scaling_method": "geometric",
        "threads": 4,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [75, 500, 5000],  # Different size buckets
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 32,
        "strategy": "size_homogeneous",
    },
}


def get_policy() -> Policy:
    """Return the policy for this solution."""
    return Policy.from_dict(POLICY_DATA)


def get_policy_data() -> dict:
    """Return raw policy data."""
    return POLICY_DATA.copy()


if __name__ == "__main__":
    policy = get_policy()
    print(f"Policy: {policy.name}")
    print(json.dumps(policy.to_dict(), indent=2))
