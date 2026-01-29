"""
Hybrid Multi-Stage Network - Precomputed Lookup Table Version
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes with eliminated branching
Stage 3: Bubble sort cleanup for guaranteed completion
MUTATION: Precomputed lookup tables for all comparison patterns
"""

N = 16

# MUTATION: Precompute all comparison patterns as lookup tables
# This eliminates runtime computation of gaps, pairs, and loop iterations

# Precomputed Shell sort gap sequence for N=16 (Knuth sequence, gaps >= 4)
SHELL_GAPS = [13, 4]  # Computed once: gaps that are >= 4 for n=16

# Precomputed Shell sort comparisons for each gap
SHELL_COMPARISONS = []
for gap in SHELL_GAPS:
    gap_comparisons = []
    for i in range(gap, N):
        j = i
        while j >= gap:
            gap_comparisons.append((j - gap, j))
            j -= gap
    SHELL_COMPARISONS.append(gap_comparisons)

# Precomputed odd-even patterns
EVEN_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
ODD_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]

# Precomputed odd-even sequence for N=16 (max_passes = N//2 = 8)
ODDEVEN_SEQUENCE = []
for pass_num in range(N // 2):
    if pass_num & 1 == 0:
        ODDEVEN_SEQUENCE.extend(EVEN_PAIRS)
    else:
        ODDEVEN_SEQUENCE.extend(ODD_PAIRS)

# Precomputed final bubble sort cleanup (final_passes = min(4, N//4) = 4)
BUBBLE_CLEANUP = []
for i in range(4):
    for j in range(N - 1 - i):
        BUBBLE_CLEANUP.append((j, j + 1))

def hybrid_multistage_network(n):
    """Generate hybrid network using precomputed lookup tables."""
    network = []

    # Stage 1: Shell sort using precomputed comparisons
    for gap_comparisons in SHELL_COMPARISONS:
        network.extend(gap_comparisons)

    # Stage 2: Odd-even transposition using precomputed sequence
    network.extend(ODDEVEN_SEQUENCE)

    # Stage 3: Final cleanup using precomputed bubble comparisons
    network.extend(BUBBLE_CLEANUP)

    return network

NETWORK = hybrid_multistage_network(N)