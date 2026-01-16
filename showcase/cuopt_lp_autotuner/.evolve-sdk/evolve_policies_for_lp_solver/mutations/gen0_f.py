"""
gen0_f.py - Density-aware batching strategy

Approach: Use density-homogeneous batching to group LPs by sparsity
pattern. This can improve cache efficiency for sparse matrices.

Strategy: Density-based batching, row reordering for cache locality
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen0_f_density_aware",
    "description": "Density-aware: batching by sparsity, cache-friendly reordering",

    "transforms": {
        "enable_row_scale": True,
        "enable_col_scale": False,
        "enable_row_reorder": True,  # Key for cache locality
        "enable_col_reorder": True,
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        "row_reorder_method": "degree_descending",  # Dense rows first
        "col_reorder_method": "degree_ascending",   # Sparse cols first
    },

    "solver_params": {
        "tolerance": 1e-6,
        "max_iterations": 12000,
        "presolve": True,
        "scaling_method": "none",
        "threads": 4,
    },

    "batching": {
        "enabled": True,
        "size_thresholds": [100, 1000, 10000],
        "density_thresholds": [0.02, 0.15],  # Different density splits
        "max_batch_size": 24,
        "strategy": "density_homogeneous",
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
