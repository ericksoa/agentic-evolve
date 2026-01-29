"""
Hybrid Multi-Stage Network - Lookup Table Optimized Version
Stage 1: Precomputed shell sort gaps for initial long-distance ordering
Stage 2: Precomputed odd-even passes with eliminated branching
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

# Precomputed lookup tables for performance optimization
_SHELL_GAPS = [13, 4]  # Precomputed Knuth gaps >= 4 for N=16
_EVEN_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
_ODD_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
_MAX_PASSES = 8  # N // 2

# Precomputed shell sort network for gaps [13, 4]
_SHELL_NETWORK = [
    # Gap 13 comparisons
    (0, 13), (1, 14), (2, 15),
    # Gap 4 comparisons
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (8, 12), (9, 13), (10, 14), (11, 15), (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14), (11, 15)
]

# Precomputed odd-even network for 8 passes
_ODDEVEN_NETWORK = [
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
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)
]

# Precomputed final bubble sort cleanup (4 passes)
_BUBBLE_NETWORK = [
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
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (12, 13)
]

def hybrid_multistage_network(n):
    """Generate hybrid network using precomputed lookup tables for maximum performance."""
    # Concatenate precomputed networks - no runtime computation needed
    return _SHELL_NETWORK + _ODDEVEN_NETWORK + _BUBBLE_NETWORK

NETWORK = hybrid_multistage_network(N)