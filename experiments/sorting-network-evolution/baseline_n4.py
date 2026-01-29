"""
Baseline 4-input sorting network.

This is a simple starting point for evolution experiments.
The optimal 4-input sorting network uses exactly 5 comparators.

Known bounds for n=4:
- Optimal comparators: 5
- Optimal depth: 3

This baseline is optimal, but evolution might find alternative
optimal networks with different structures.
"""

# Number of inputs to sort
N = 4

# Optimal 4-input sorting network (5 comparators, depth 3)
NETWORK = [
    # Stage 1 (depth 1): Sort pairs
    (0, 1), (2, 3),

    # Stage 2 (depth 2): Compare across
    (0, 2), (1, 3),

    # Stage 3 (depth 3): Final comparison
    (1, 2),
]

assert len(NETWORK) == 5, f"Expected 5 comparators, got {len(NETWORK)}"
