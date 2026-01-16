"""
gen2a.py - Tighter tolerance mutation of gen0_a

Parent: gen0_a (conservative baseline with row scaling only, fitness=0)
Mutation type: Parameter tweak
Change: tolerance: 1e-6 -> 1e-8

Hypothesis: The parent's fitness of 0 may indicate convergence issues or
solution inaccuracy. Using a tighter tolerance (1e-8) forces the solver
to converge more precisely, which could improve solution quality metrics.
Row scaling is retained to maintain numerical stability while the tighter
tolerance ensures higher precision in the final solution.
"""

import json

# Policy definition - conservative baseline with tighter tolerance
POLICY_DATA = {
    "version": "1.0",
    "name": "gen2a_tight_tolerance",
    "description": "Mutation of gen0_a: Tighter solver tolerance (1e-8) for improved precision",

    "transforms": {
        "enable_row_scale": True,   # Keep from parent
        "enable_col_scale": False,
        "enable_row_reorder": False,
        "enable_col_reorder": False,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_ascending",
        "col_reorder_method": "degree_descending",
    },

    "solver_params": {
        "tolerance": 1e-8,          # MUTATION: 1e-6 -> 1e-8 (tighter tolerance)
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
