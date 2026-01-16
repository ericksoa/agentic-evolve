"""
gen1x.py - Crossover hybrid from gen0_a, gen0_e, gen0_d

Approach: Balanced hybrid combining:
- Conservative numerical stability from gen0_a (row scaling foundation)
- Size-aware batching for GPU utilization from gen0_e (geometric scaling, size_homogeneous)
- Selective transforms from gen0_d (row/col reorder for better structure)

Strategy: Moderate transforms, geometric scaling, size-based batching, balanced tolerance
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen1x_balanced_hybrid",
    "description": "Hybrid: size-aware batching + selective transforms + geometric scaling",

    "transforms": {
        # From gen0_a/gen0_e: both scaling for numerical stability
        "enable_row_scale": True,
        "enable_col_scale": True,
        # From gen0_d: row reordering for better structure (but not col to avoid overhead)
        "enable_row_reorder": True,
        "enable_col_reorder": False,
        # Conservative: avoid aggressive transforms that may cause issues
        "enable_rhs_normalize": False,
        "enable_bound_tighten": False,
        # From gen0_d: use rhs_ascending for better numerical ordering
        "row_reorder_method": "rhs_ascending",
        "col_reorder_method": "degree_descending",  # From gen0_a baseline
    },

    "solver_params": {
        # Balanced tolerance - tighter than baseline but not extreme like gen0_d
        "tolerance": 1e-7,
        # Moderate iterations - between gen0_a's 10k and gen0_d's 100k
        "max_iterations": 25000,
        "presolve": True,
        # From gen0_e: geometric scaling for better performance
        "scaling_method": "geometric",
        # From gen0_e: multi-threaded for performance
        "threads": 4,
    },

    "batching": {
        # From gen0_e: enable size-aware batching
        "enabled": True,
        # Hybrid thresholds - blend between gen0_e and gen0_d
        "size_thresholds": [100, 1000, 10000],
        "density_thresholds": [0.01, 0.1],
        # From gen0_e: larger batch for GPU utilization
        "max_batch_size": 24,
        # From gen0_e: size-homogeneous for better GPU efficiency
        "strategy": "size_homogeneous",
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
