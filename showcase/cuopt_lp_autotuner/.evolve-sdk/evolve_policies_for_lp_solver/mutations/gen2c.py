"""
gen2c.py - Relaxed tolerance mutation of gen0_d

Parent: gen0_d (precision-optimized with tight tolerance)
Mutation: Relaxed tolerance from 1e-9 to 1e-7

Hypothesis: The extremely tight tolerance (1e-9) in the parent may cause
excessive iterations for marginal quality improvement. A moderately relaxed
tolerance (1e-7) should allow faster convergence while maintaining good
solution quality - a better precision/performance tradeoff. Combined with
the transforms still enabled, this should improve throughput.
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen2c_relaxed_tolerance",
    "description": "Relaxed tolerance (1e-7) with all transforms for better performance",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": True,
        "enable_col_reorder": True,
        "enable_rhs_normalize": True,
        "enable_bound_tighten": True,
        "row_reorder_method": "rhs_ascending",
        "col_reorder_method": "bounds_first",
    },

    "solver_params": {
        "tolerance": 1e-7,  # MUTATION: Relaxed from 1e-9 to 1e-7
        "max_iterations": 100000,  # Keep high max iterations
        "presolve": True,
        "scaling_method": "equilibration",
        "threads": 2,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [200, 2000, 20000],
        "density_thresholds": [0.005, 0.05],
        "max_batch_size": 8,
        "strategy": "mixed",
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
    print(f"Policy: {POLICY_DATA['name']}")
    print(json.dumps(POLICY_DATA, indent=2))
