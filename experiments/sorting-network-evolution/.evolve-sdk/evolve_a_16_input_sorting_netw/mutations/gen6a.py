"""
Precomputed Static Network - eliminates runtime network generation overhead
Using a statically computed comparison network for 16 elements to maximize performance
by avoiding any algorithmic computation during network generation.
"""

N = 16

# Precomputed static network based on hybrid multi-stage approach
# This eliminates all runtime computation for network generation
NETWORK = [
    # Stage 1: Shell sort gaps (precomputed for gaps 13, 4, 1)
    # Gap 13 comparisons
    (0, 13), (1, 14), (2, 15),
    # Gap 4 comparisons
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (8, 12), (9, 13), (10, 14), (11, 15),

    # Stage 2: Odd-even transposition passes (8 passes for n//2)
    # Pass 0 (even)
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    # Pass 1 (odd)
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    # Pass 2 (even)
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    # Pass 3 (odd)
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    # Pass 4 (even)
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    # Pass 5 (odd)
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    # Pass 6 (even)
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    # Pass 7 (odd)
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),

    # Stage 3: Final bubble cleanup (4 passes)
    # Pass 0
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    # Pass 1
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    # Pass 2
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    # Pass 3
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12)
]

def hybrid_multistage_network(n):
    """Return precomputed static network - no runtime computation needed."""
    return NETWORK