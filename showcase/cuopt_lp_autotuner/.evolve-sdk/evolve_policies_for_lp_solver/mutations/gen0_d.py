"""
gen0_d.py - Precision-optimized with tight tolerance

Approach: Maximize solution quality with tight tolerances, all transforms,
and small batches for better individual handling.

Strategy: Tight tolerance, all transforms, small batches
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_d_precision_optimized",
    "description": "Precision: tight tolerance, all transforms, small batches",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": True,
        "enable_col_reorder": True,
        "enable_rhs_normalize": True,
        "enable_bound_tighten": True,
        "row_reorder_method": "rhs_ascending",  # Different ordering
        "col_reorder_method": "bounds_first",
    },

    "solver_params": {
        "tolerance": 1e-9,  # Very tight tolerance
        "max_iterations": 100000,  # Many iterations allowed
        "presolve": True,
        "scaling_method": "equilibration",  # Alternative scaling method
        "threads": 2,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [200, 2000, 20000],
        "density_thresholds": [0.005, 0.05],
        "max_batch_size": 8,  # Small batches for precision
        "strategy": "mixed",
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
