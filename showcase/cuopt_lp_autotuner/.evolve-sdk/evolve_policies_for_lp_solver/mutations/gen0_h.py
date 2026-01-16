"""
gen0_h.py - Minimal transforms with bound tightening

Approach: Use only essential transforms - bound tightening can reduce
problem size without numerical risk. Fast preprocessing.

Strategy: Minimal transforms, bound tightening only, fast iterations
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_h_minimal_bounds",
    "description": "Minimal: bound tightening + basic scaling",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": False,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": True,  # Key transform - reduces search space
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-5,  # Moderate tolerance
        "max_iterations": 8000,
        "presolve": True,  # Let solver do more work
        "scaling_method": "geometric",
        "threads": 6,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [100, 800, 8000],
        "density_thresholds": [0.015, 0.12],
        "max_batch_size": 48,
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
