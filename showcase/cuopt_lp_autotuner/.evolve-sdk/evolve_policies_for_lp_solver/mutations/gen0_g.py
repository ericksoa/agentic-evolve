"""
gen0_g.py - Scaling-heavy with equilibration

Approach: Focus on numerical conditioning through multiple scaling
approaches. Uses both transform-level and solver-level scaling.

Strategy: Heavy scaling, equilibration method, moderate batching
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_g_scaling_heavy",
    "description": "Scaling-heavy: equilibration + transform scaling",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": False,
        "enable_col_reorder": True,
        "enable_rhs_normalize": True,  # Normalize RHS too
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "bounds_first",  # Bounded vars first
    },

    "solver_params": {
        "tolerance": 1e-7,  # Moderate-tight tolerance
        "max_iterations": 20000,
        "presolve": True,
        "scaling_method": "equilibration",  # Solver-level scaling
        "threads": 4,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [150, 1500, 15000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 16,
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
