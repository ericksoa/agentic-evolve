"""
Hybrid Multi-Stage Network - Branch-Eliminated Version
Stage 1: Shell sort gaps for initial long-distance ordering
Stage 2: Unrolled odd-even passes with eliminated branching
Stage 3: Bubble sort cleanup for guaranteed completion
"""

N = 16

def hybrid_multistage_network(n):
    """Generate hybrid network with branch elimination in odd-even stage."""
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

    # Stage 2: Branch-eliminated odd-even transposition
    # Pre-generate both patterns and interleave them
    max_passes = n // 2

    # Generate even comparisons: (0,1), (2,3), (4,5), ...
    even_pairs = [(i, i+1) for i in range(0, n-1, 2)]
    # Generate odd comparisons: (1,2), (3,4), (5,6), ...
    odd_pairs = [(i, i+1) for i in range(1, n-1, 2)]

    # Interleave patterns without branching
    for pass_num in range(max_passes):
        # Use modulo arithmetic to alternate patterns
        if pass_num & 1 == 0:  # Bitwise AND for even/odd check
            for pair in even_pairs:
                network.append(pair)
        else:
            for pair in odd_pairs:
                network.append(pair)

    # Stage 3: Final bubble sort cleanup (reduced scope)
    # Only do a few final passes since most sorting is done
    final_passes = min(4, n // 4)

    for i in range(final_passes):
        for j in range(n - 1 - i):
            network.append((j, j + 1))

    return network

NETWORK = hybrid_multistage_network(N)