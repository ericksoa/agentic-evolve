"""
gen0_c.py - Speed-optimized with loose tolerance

Approach: Minimize solve time by using loose tolerances and minimal
transforms. Trade precision for throughput.

Strategy: Loose tolerance, minimal transforms, large batches
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_c_speed_optimized",
    "description": "Speed: loose tolerance, minimal transforms, batching",

    "transforms": {
        "enable_row_scale": False,
        "enable_col_scale": False,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-4,  # Loose tolerance for speed
        "max_iterations": 5000,  # Fewer iterations
        "presolve": True,
        "scaling_method": "none",
        "threads": 8,  # Max threads
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [50, 500, 5000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 64,  # Large batches for GPU efficiency
        "strategy": "size_homogeneous",
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
