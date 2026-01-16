"""
gen0_a.py - Conservative baseline with minor tweaks

Approach: Start from baseline, add only row scaling for numerical stability.
This is a safe, conservative policy that slightly improves conditioning
without risking regressions.

Strategy: Minimal transforms, conservative parameters, no batching
"""

import json

# Policy definition - conservative baseline with row scaling only
POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_a_conservative",
    "description": "Conservative: baseline + row scaling only",

    "transforms": {
        "enable_row_scale": True,  # Only change from baseline
        "enable_col_scale": False,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-6,
        "max_iterations": 10000,
        "presolve": True,
        "scaling_method": "none",
        "threads": 1,
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
    # Import here to avoid module-level dependency
    import sys
    from pathlib import Path
    # Add parent dirs to find policy module
    cuopt_dir = Path(__file__).parent.parent.parent.parent
    if str(cuopt_dir) not in sys.path:
        sys.path.insert(0, str(cuopt_dir))
    from policy import Policy
    return Policy.from_dict(POLICY_DATA)


def get_policy_data() -> dict:
    """Return raw policy data."""
    return POLICY_DATA.copy()


if __name__ == "__main__":
    # Print policy for verification
    print(f"Policy: {POLICY_DATA['name']}")
    print(json.dumps(POLICY_DATA, indent=2))
