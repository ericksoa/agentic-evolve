"""
gen2x.py - Crossover hybrid: Full transforms + Size-aware batching

Approach: Combine the best aspects of three parent strategies:
- Full transform suite from gen0_d for numerical conditioning
- Size-homogeneous batching from gen0_e for GPU efficiency
- Balanced parameters bridging conservative and aggressive approaches

Strategy: Full transforms, geometric scaling, size-aware batching, moderate parallelism
Parents: gen0_a, gen0_e, gen0_d
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from showcase.cuopt_lp_autotuner.policy import Policy

POLICY_DATA = {
    "version": "1.0",
    "name": "gen2x_hybrid_crossover",
    "description": "Crossover: full transforms (from d) + size batching (from e) + balanced params",

    "transforms": {
        # Full transform suite from gen0_d for best conditioning
        "enable_row_scale": True,
        "enable_col_scale": True,
        "enable_row_reorder": True,
        "enable_col_reorder": True,
        "enable_rhs_normalize": True,
        "enable_bound_tighten": True,
        # Ordering methods from gen0_d (precision-focused)
        "row_reorder_method": "rhs_ascending",
        "col_reorder_method": "bounds_first",
    },

    "solver_params": {
        # Balanced tolerance: tighter than baseline but not as extreme as gen0_d
        "tolerance": 1e-8,
        # More iterations than baseline, less than precision-focused
        "max_iterations": 25000,
        "presolve": True,
        # Geometric scaling from gen0_e (good for general use)
        "scaling_method": "geometric",
        # Multi-threading from gen0_e for performance
        "threads": 4,
    },

    "batching": {
        # Batching enabled from gen0_e
        "enabled": True,
        # Size thresholds: blend of gen0_e and gen0_d
        "size_thresholds": [100, 750, 7500],
        # Density thresholds from gen0_d (tighter for precision)
        "density_thresholds": [0.005, 0.075],
        # Moderate batch size: between gen0_e (32) and gen0_d (8)
        "max_batch_size": 16,
        # Size-homogeneous strategy from gen0_e for GPU efficiency
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
