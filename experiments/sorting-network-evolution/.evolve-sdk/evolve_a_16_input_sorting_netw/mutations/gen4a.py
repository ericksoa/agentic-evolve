"""
Lookup Table Optimized Network - precomputed patterns for maximum performance
All gap sequences, pair patterns, and comparison sequences are precomputed
Eliminates all runtime calculations and branching for optimal cache performance
"""

N = 16

# Precomputed lookup tables for N=16 (avoiding runtime calculations)
SHELL_GAPS = [13, 4]  # Knuth sequence gaps >= 4 for N=16
SHELL_COMPARISONS = [
    # Gap 13 comparisons (precomputed)
    (0, 13), (1, 14), (2, 15),
    # Gap 4 comparisons (precomputed in optimal order)
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 12), (1, 13), (2, 14), (3, 15), (4, 8), (5, 9), (6, 10), (7, 11)
]

# Precomputed odd-even patterns (8 passes for N=16//2)
ODD_EVEN_PATTERNS = [
    # Even passes: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    # Odd passes: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14)
    [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)],
    # Repeat pattern for 8 total passes
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)],
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)],
    [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
]

# Precomputed final cleanup patterns (4 passes as in original)
CLEANUP_PATTERNS = [
    # Pass 0: bubble all adjacent pairs except last 0
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)],
    # Pass 1: bubble all adjacent pairs except last 1
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)],
    # Pass 2: bubble all adjacent pairs except last 2
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
    # Pass 3: bubble all adjacent pairs except last 3
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
]

def lookup_table_network(n):
    """Generate network using precomputed lookup tables - zero runtime calculations."""
    network = []

    # Stage 1: Shell sort using precomputed comparisons
    network.extend(SHELL_COMPARISONS)

    # Stage 2: Odd-even transposition using precomputed patterns
    for pattern in ODD_EVEN_PATTERNS:
        network.extend(pattern)

    # Stage 3: Final cleanup using precomputed patterns
    for pattern in CLEANUP_PATTERNS:
        network.extend(pattern)

    return network

NETWORK = lookup_table_network(N)