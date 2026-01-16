"""
gen1c.py - Hybrid dispatch mutation of gen0_d

Approach: Use conditional rules to adapt solver parameters based on problem size.
Small problems get fewer max_iterations (faster), large problems get more.

Strategy: Hybrid dispatch - different solver configs based on input size
Parent: gen0_d (precision-optimized with tight tolerance)
Mutation: Changed max_iterations from fixed 100000 to conditional rule based on size_bucket
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen1c_hybrid_dispatch",
    "description": "Hybrid dispatch: adaptive max_iterations based on problem size",

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
        "tolerance": 1e-9,  # Keep tight tolerance from parent
        # MUTATION: Use conditional rule for max_iterations based on problem size
        "max_iterations": {
            "type": "conditional",
            "feature": "size_bucket",
            "conditions": [
                {"match": "tiny", "value": 5000},      # Fast for tiny problems
                {"match": "small", "value": 20000},    # Moderate for small
                {"match": "medium", "value": 50000},   # More for medium
                {"match": "large", "value": 100000},   # Full iterations for large
                {"default": 100000}
            ]
        },
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
