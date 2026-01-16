"""
gen1b.py - Thread parallelism increase mutation

Parent: gen0_e (size-aware batching, geometric scaling)
Mutation: Increase threads from 4 to 8 for maximum parallelism

Hypothesis: Doubling thread count may improve throughput on
multi-core systems by better utilizing available parallelism,
especially when combined with batched execution.
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen1b_max_threads",
    "description": "Size-aware batching with maximum thread parallelism (8 threads)",

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
        "threads": 8,  # MUTATION: Increased from 4 to 8 (max allowed)
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [75, 500, 5000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 32,
        "strategy": "size_homogeneous",
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
