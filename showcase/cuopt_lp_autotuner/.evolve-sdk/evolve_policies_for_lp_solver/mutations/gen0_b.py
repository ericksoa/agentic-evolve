"""
gen0_b.py - Aggressive transforms with all scaling

Approach: Enable all transforms for maximum numerical stability.
Uses geometric scaling on both rows and columns, plus reordering
and bound tightening.

Strategy: All transforms enabled, tight tolerance, moderate threads
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_b_aggressive_transforms",
    "description": "Aggressive: all transforms, tight tolerance",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": True,
        "enable_col_reorder": True,
        "enable_rhs_normalize": True,
        "enable_bound_tighten": True,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-8,  # Tight tolerance for accuracy
        "max_iterations": 50000,  # More iterations allowed
        "presolve": True,
        "scaling_method": "geometric",  # Use solver's built-in scaling too
        "threads": 4,
    },

    "batching": {
        "enabled": False,
        "size_thresholds": [100, 1000, 10000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 1,
        "strategy": "none",
    },
}


def get_policy():
    """Return the policy for this solution."""
    import sys
    from pathlib import Path
    cuopt_dir = Path(__file__).parent.parent.parent.parent
    if str(cuopt_dir) not in sys.path:
        sys.path.insert(0, str(cuopt_dir))
    from policy import Policy
    return Policy.from_dict(POLICY_DATA)


def get_policy_data() -> dict:
    """Return raw policy data."""
    return POLICY_DATA.copy()


if __name__ == "__main__":
    print(f"Policy: {POLICY_DATA['name']}")
    print(json.dumps(POLICY_DATA, indent=2))
