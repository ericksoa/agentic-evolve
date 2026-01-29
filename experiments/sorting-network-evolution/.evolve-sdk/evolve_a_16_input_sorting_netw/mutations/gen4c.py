"""
Hybrid Multi-Stage Network - loop unrolling optimization
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Fully unrolled odd-even passes (explicit pairs, no loops)
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with fully unrolled odd-even stage."""
    network = []

    # Stage 1: Shell sort with gap sequence for initial ordering
    # Use Knuth's gap sequence but only larger gaps to avoid redundancy
    gaps = []
    k = 1
    while k < n:
        gaps.append(k)
        k = 3 * k + 1
    gaps.reverse()

    # Only use gaps >= 4 to focus on long-distance comparisons
    large_gaps = [gap for gap in gaps if gap >= 4]

    for gap in large_gaps:
        # For each gap, compare all pairs at gap distance
        for i in range(gap, n):
            j = i
            while j >= gap:
                network.append((j - gap, j))
                j -= gap

    # Stage 2: Fully unrolled odd-even transposition
    # Eliminate all loops by explicitly listing all pairs for each pass
    max_passes = n // 2  # Reduced from n to n//2 due to shell pre-sorting

    # Pre-generate all pairs for maximum unrolling
    even_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]
    odd_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14)]

    # Fully unroll passes - explicit pass enumeration
    for pass_num in range(max_passes):
        if (pass_num & 1) == 0:
            # Even pass - fully unrolled
            network.append((0,1))
            network.append((2,3))
            network.append((4,5))
            network.append((6,7))
            network.append((8,9))
            network.append((10,11))
            network.append((12,13))
            network.append((14,15))
        else:
            # Odd pass - fully unrolled
            network.append((1,2))
            network.append((3,4))
            network.append((5,6))
            network.append((7,8))
            network.append((9,10))
            network.append((11,12))
            network.append((13,14))

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n // 4)  # Much fewer passes than full bubble sort

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)