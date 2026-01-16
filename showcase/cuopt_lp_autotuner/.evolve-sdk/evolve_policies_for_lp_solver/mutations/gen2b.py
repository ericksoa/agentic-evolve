"""
gen2b.py - Mixed batching strategy with maximum batch size

Parent: gen0_e (size-aware batching, size_homogeneous strategy, batch_size=32)
Mutation: Change batching strategy from "size_homogeneous" to "mixed" and
          increase max_batch_size from 32 to 64 (maximum allowed)

Hypothesis: Mixed batching strategy considers both problem size AND density
characteristics when grouping problems, which may lead to better GPU utilization
by ensuring batched problems have similar computational profiles. Combined with
maximum batch size of 64, this should better amortize kernel launch overhead
and improve overall throughput for workloads with many LP problems.
"""

import json

POLICY_DATA = {
    "version": "1.0",
    "name": "gen2b_mixed_batch_64",
    "description": "Mixed batching strategy with maximum batch size (64)",

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
        "threads": 4,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [75, 500, 5000],
        "density_thresholds": [0.01, 0.1],
        "max_batch_size": 64,    # MUTATION: Increased from 32 to 64 (max)
        "strategy": "mixed",      # MUTATION: Changed from "size_homogeneous" to "mixed"
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
