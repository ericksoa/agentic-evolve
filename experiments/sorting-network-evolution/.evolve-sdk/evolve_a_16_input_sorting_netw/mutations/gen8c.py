"""
Hybrid Multi-Stage Network - loop unrolling optimization
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Fully unrolled odd-even passes (loop unrolling)
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with unrolled odd-even stage."""
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
    # Completely unroll the passes for maximum performance
    max_passes = n // 2  # Reduced from n to n//2 due to shell pre-sorting

    # Unroll all passes explicitly for N=16 (8 passes)
    if n == 16:
        # Pass 0 (even): pairs at 0,2,4,6,8,10,12,14
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            network.append((i, i+1))
        # Pass 1 (odd): pairs at 1,3,5,7,9,11,13
        for i in [1, 3, 5, 7, 9, 11, 13]:
            network.append((i, i+1))
        # Pass 2 (even)
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            network.append((i, i+1))
        # Pass 3 (odd)
        for i in [1, 3, 5, 7, 9, 11, 13]:
            network.append((i, i+1))
        # Pass 4 (even)
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            network.append((i, i+1))
        # Pass 5 (odd)
        for i in [1, 3, 5, 7, 9, 11, 13]:
            network.append((i, i+1))
        # Pass 6 (even)
        for i in [0, 2, 4, 6, 8, 10, 12, 14]:
            network.append((i, i+1))
        # Pass 7 (odd)
        for i in [1, 3, 5, 7, 9, 11, 13]:
            network.append((i, i+1))
    else:
        # Fallback for other sizes (keep original logic)
        even_pairs = list(range(0, n-1, 2))
        odd_pairs = list(range(1, n-1, 2))

        for pass_num in range(max_passes):
            pairs = even_pairs if (pass_num & 1) == 0 else odd_pairs
            for i in pairs:
                network.append((i, i+1))

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n // 4)

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)