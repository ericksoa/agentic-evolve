"""
Baseline 8-input sorting network using Batcher's odd-even mergesort.

This is a well-known construction that produces valid but non-optimal networks.
For n=8, this gives 19 comparators (which happens to be optimal for comparator count).

The goal of evolution is to find networks that:
1. Still correctly sort (mandatory)
2. Use fewer comparators (primary objective)
3. Have lower depth (secondary objective)

Known bounds for n=8:
- Optimal comparators: 19
- Optimal depth: 6

This baseline achieves 19 comparators but may not have optimal depth.
Evolution should try to match or beat this while potentially finding
novel network structures.
"""

# Number of inputs to sort
N = 8

# Batcher's odd-even mergesort for n=8
# Format: list of (i, j) comparators where i < j
# A comparator swaps positions i and j if arr[i] > arr[j]
NETWORK = [
    # Stage 1: Compare adjacent pairs
    (0, 1), (2, 3), (4, 5), (6, 7),

    # Stage 2: Compare with distance 2
    (0, 2), (1, 3), (4, 6), (5, 7),

    # Stage 3: Odd-even merge step
    (1, 2), (5, 6),

    # Stage 4: Compare across halves
    (0, 4), (1, 5), (2, 6), (3, 7),

    # Stage 5: Merge step
    (2, 4), (3, 5),

    # Stage 6: Final cleanup
    (1, 2), (3, 4), (5, 6),
]

# Verification that this is 19 comparators
assert len(NETWORK) == 19, f"Expected 19 comparators, got {len(NETWORK)}"
